"""Circuit identity, seeding, and split membership shared by training and evaluation.

A circuit is named by a *logical identity* inside a split namespace, and the
track generator's seed is derived from that pair. The indirection is what lets
two runs agree on which circuit they mean: paired observation conditions draw
the same ordered identities from identically seeded streams, and a saved
validation or test circuit keeps its identity even though the integer the
generator consumed is an implementation detail recorded beside it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np

from configs import EnvironmentConfig, StartStateConfig
from envs.racing import RacingEnv
from envs.tracks import TrackWithGeometry
from utils.random import RunSeedStreams, SeedNamespace, SeedStream

# Circuit identities are drawn from this half-open range. It is far larger than
# any run consumes, so repeats within a run are effectively absent, and it stays
# inside the signed 32-bit values NumPy indexes comfortably.
CIRCUIT_IDENTITY_LIMIT = 2**31


class CircuitSplit(StrEnum):
    """
    Name the role a circuit plays, which decides what may be learned from it.
    """

    DEVELOPMENT = "development"
    TRAINING = "training"
    TRAINING_REFERENCE = "training_reference"
    VALIDATION = "validation"
    TEST = "test"


def circuit_track_seed(namespace: SeedNamespace, circuit_identity: int) -> int:
    """
    Return the generator seed a circuit identity denotes inside its split.

    The identity is the local identity of its split namespace, and the seed is
    the first value of that namespace's track-generation stream.
    """
    if circuit_identity < 0:
        raise ValueError("Circuit identities cannot be negative.")
    return RunSeedStreams(namespace, circuit_identity).integer_seed(
        SeedStream.TRACK_GENERATION
    )


@dataclass(frozen=True, slots=True)
class TrainingCircuitSchedule:
    """
    Draw the unbounded sequence of training circuits one worker will race.

    Each worker owns an identically seeded selection stream, so worker `w` meets
    the same circuit on its `k`-th episode in every run sharing the root. That is
    what pairs two observation conditions whose episodes end at different times:
    pairing by worker and per-worker episode count survives the difference,
    while pairing by a global episode index would not.

    Fields:
        * namespace: Split namespace the drawn identities belong to.
    """

    namespace: SeedNamespace = SeedNamespace.EXPERIMENT_2_TRAINING_TRACK

    def draw(self, generator: np.random.Generator) -> tuple[int, int]:
        """
        Consume one identity from a worker's selection stream and seed it.
        """
        identity = int(generator.integers(0, CIRCUIT_IDENTITY_LIMIT))
        return identity, circuit_track_seed(self.namespace, identity)


@dataclass(frozen=True, slots=True)
class EvaluationCircuit:
    """
    Bind one deterministic evaluation circuit to the split it is drawn from.

    Fields:
        * identity: Logical circuit identity recorded with every outcome.
        * split: Role deciding whether results may influence training.
        * factory: Builds a fresh environment on this circuit.
    """

    identity: str
    split: CircuitSplit | None
    factory: Callable[[], RacingEnv]


SPLIT_NAMESPACES: dict[CircuitSplit, SeedNamespace] = {
    CircuitSplit.DEVELOPMENT: SeedNamespace.MULTI_CIRCUIT_DEVELOPMENT,
    CircuitSplit.TRAINING: SeedNamespace.EXPERIMENT_2_TRAINING_TRACK,
    # A training reference is a circuit the run actually trained on, revisited
    # deterministically at the end. It is drawn from the training namespace by
    # construction, and it is the only split that is deliberately in-sample:
    # the gap between it and the test split is what generalization means here.
    CircuitSplit.TRAINING_REFERENCE: SeedNamespace.EXPERIMENT_2_TRAINING_TRACK,
    CircuitSplit.VALIDATION: SeedNamespace.EXPERIMENT_2_VALIDATION_TRACK,
    CircuitSplit.TEST: SeedNamespace.EXPERIMENT_2_TEST_TRACK,
}

HELD_OUT_SPLITS = (CircuitSplit.VALIDATION, CircuitSplit.TEST)


def circuit_geometry_checksum(track: TrackWithGeometry) -> str:
    """
    Fingerprint one circuit's sampled geometry.

    A circuit is named by an identity and rebuilt from the generator on demand
    rather than stored, which is only safe while the generator is frozen. The
    checksum turns a change in the generator into a loud mismatch instead of a
    silent change of what "validation circuit 3" means.
    """
    sampled = track.track
    digest = sha256()
    for values in (sampled.x, sampled.y, sampled.curvature):
        digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    digest.update(np.float64(sampled.width).tobytes())
    return digest.hexdigest()


def generated_evaluation_circuits(
    identities: Iterable[int],
    *,
    split: CircuitSplit,
    environment_config: EnvironmentConfig,
    namespace: SeedNamespace | None = None,
    expected_checksums: dict[str, str] | None = None,
) -> tuple[EvaluationCircuit, ...]:
    """
    Prepare one evaluation circuit per identity in a split.

    Each circuit is generated once here rather than inside its factory: a run
    rebuilds every one of them at each of forty checkpoints, and regenerating
    identical geometry that many times would cost more than the evaluation.

    Evaluation always launches from the canonical start line, so the returned
    environments never sample a start pose regardless of the training setting.
    """
    resolved_namespace = namespace or SPLIT_NAMESPACES[split]
    evaluation_config = replace(
        environment_config, start=StartStateConfig(randomized=False)
    )
    circuits: list[EvaluationCircuit] = []
    for identity in identities:
        track = TrackWithGeometry.generate(
            circuit_track_seed(resolved_namespace, identity),
            track_config=environment_config.track,
            vehicle_config=environment_config.vehicle,
        )
        if expected_checksums is not None:
            expected = expected_checksums.get(str(identity))
            actual = circuit_geometry_checksum(track)
            if expected != actual:
                raise ValueError(
                    f"{split.value} circuit {identity} no longer matches the frozen "
                    "split: the generator or its configuration has changed."
                )
        circuits.append(
            EvaluationCircuit(
                identity=str(identity),
                split=split,
                factory=partial(RacingEnv, track, config=evaluation_config),
            )
        )
    return tuple(circuits)


def load_split_circuits(
    manifest_path: str | Path,
    split: CircuitSplit,
    *,
    environment_config: EnvironmentConfig,
) -> tuple[EvaluationCircuit, ...]:
    """
    Rebuild one frozen split, refusing to proceed if its geometry has drifted.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entry = manifest["splits"].get(split.value)
    if entry is None:
        raise ValueError(f"The split manifest does not declare {split.value}.")
    return generated_evaluation_circuits(
        (int(circuit["identity"]) for circuit in entry["circuits"]),
        split=split,
        environment_config=environment_config,
        namespace=SeedNamespace[entry["namespace"]],
        expected_checksums={
            str(circuit["identity"]): str(circuit["geometry_checksum"])
            for circuit in entry["circuits"]
        },
    )
