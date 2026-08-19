"""Tests for circuit identity, seeding, and frozen split membership."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from circuits import (
    HELD_OUT_SPLITS,
    SPLIT_NAMESPACES,
    load_split_circuits,
)
from configs import EnvironmentConfig
from envs.tracks import TrackWithGeometry
from training import (
    CIRCUIT_IDENTITY_LIMIT,
    CircuitSplit,
    TrainingCircuitSchedule,
    circuit_track_seed,
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
                "circuits": [{"identity": 0, "track_length": 123.4}],
            }
        }
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="track_length was 123.4"):
        load_split_circuits(
            path,
            CircuitSplit.VALIDATION,
            environment_config=EnvironmentConfig(),
        )


def test_frozen_geometry_survives_a_last_bit_difference_between_machines() -> None:
    """
    The check must tolerate a rebuild on another platform, and nothing more.

    Generation runs through `math.cos` and `math.sin`, whose last bit is a
    property of the platform's libm rather than of the circuit. Comparing raw
    coordinates byte for byte therefore failed on Linux for circuits identical
    to within a picometre, while a real change to the generator moves this
    geometry by metres.
    """
    from circuits import (
        circuit_geometry_fingerprint,
        verify_circuit_geometry,
    )

    configuration = EnvironmentConfig()
    track = TrackWithGeometry.generate(
        circuit_track_seed(SeedNamespace.EXPERIMENT_2_VALIDATION_TRACK, 0),
        track_config=configuration.track,
        vehicle_config=configuration.vehicle,
    )
    frozen = circuit_geometry_fingerprint(track)

    # A last-bit disagreement, of the size a different libm produces.
    nudged = {
        name: value * (1.0 + 1e-12) if value else value
        for name, value in frozen.items()
    }
    verify_circuit_geometry(track, nudged, description="circuit")

    # A change a hundred times smaller than the narrowest real circuit feature
    # is still caught, so the tolerance buys robustness and not blindness.
    with pytest.raises(ValueError, match="no longer matches"):
        verify_circuit_geometry(
            track,
            {**frozen, "track_length": frozen["track_length"] + 0.01},
            description="circuit",
        )


def test_the_committed_split_manifest_still_describes_its_circuits() -> None:
    """
    The manifest in the repository must match what the generator produces now.
    """
    root = Path(__file__).parents[1]
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
