"""Tests for circuit identity, seeding, and frozen split membership."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from configs import EnvironmentConfig
from training import (
    CIRCUIT_IDENTITY_LIMIT,
    CircuitSplit,
    TrainingCircuitSchedule,
    circuit_track_seed,
)
from training.circuits import (
    HELD_OUT_SPLITS,
    SPLIT_NAMESPACES,
    load_split_circuits,
)
from utils.random import SeedNamespace


def test_a_circuit_identity_names_the_same_seed_every_time() -> None:
    """
    The identity is the name, so it must not depend on when it is resolved.
    """
    first = circuit_track_seed(SeedNamespace.EXPERIMENT_2_VALIDATION_TRACK, 3)
    second = circuit_track_seed(SeedNamespace.EXPERIMENT_2_VALIDATION_TRACK, 3)

    assert first == second
    # The same identity in another split is a different circuit.
    assert first != circuit_track_seed(SeedNamespace.EXPERIMENT_2_TEST_TRACK, 3)


def test_circuit_identities_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        circuit_track_seed(SeedNamespace.EXPERIMENT_2_TEST_TRACK, -1)


def test_the_schedule_draws_identities_and_their_seeds_together() -> None:
    schedule = TrainingCircuitSchedule()
    generator = np.random.default_rng(0)

    identity, seed = schedule.draw(generator)

    assert 0 <= identity < CIRCUIT_IDENTITY_LIMIT
    assert seed == circuit_track_seed(schedule.namespace, identity)
    # Drawing again consumes the stream rather than repeating.
    assert schedule.draw(generator)[0] != identity


def test_held_out_splits_never_share_a_namespace_with_training() -> None:
    """
    A held-out circuit must be unreachable from the training schedule.

    The training reference is the deliberate exception: it *is* a training
    circuit, revisited at the end, and the gap between it and the test split is
    what the experiment calls generalization.
    """
    training = SPLIT_NAMESPACES[CircuitSplit.TRAINING]
    held_out = [SPLIT_NAMESPACES[split] for split in HELD_OUT_SPLITS]

    assert len(set(held_out)) == len(held_out)
    assert training not in held_out
    assert SPLIT_NAMESPACES[CircuitSplit.DEVELOPMENT] not in held_out
    assert SPLIT_NAMESPACES[CircuitSplit.TRAINING_REFERENCE] == training


def test_a_frozen_split_is_rejected_when_its_geometry_no_longer_matches(
    tmp_path: Path,
) -> None:
    """
    Circuits are rebuilt from the generator, not stored, so drift must be loud.

    A split records what each identity looked like when it was frozen. If the
    generator or its configuration changes, the same identity yields different
    geometry, and every held-out result would silently describe other circuits.
    """
    manifest = {
        "splits": {
            "validation": {
                "namespace": SeedNamespace.EXPERIMENT_2_VALIDATION_TRACK.name,
                "circuits": [
                    {"identity": 0, "geometry_checksum": "not-the-frozen-geometry"}
                ],
            }
        }
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no longer matches the frozen split"):
        load_split_circuits(
            path,
            CircuitSplit.VALIDATION,
            environment_config=EnvironmentConfig(),
        )


def test_the_committed_split_manifest_still_describes_its_circuits() -> None:
    """
    The manifest in the repository must match what the generator produces now.
    """
    root = Path(__file__).parents[2]
    manifest_path = root / "tracks" / "experiment_2_splits.json"

    circuits = load_split_circuits(
        manifest_path,
        CircuitSplit.VALIDATION,
        environment_config=EnvironmentConfig(),
    )

    assert len(circuits) == 16
    assert [circuit.identity for circuit in circuits] == [
        str(index) for index in range(16)
    ]
    assert {circuit.split for circuit in circuits} == {CircuitSplit.VALIDATION}
