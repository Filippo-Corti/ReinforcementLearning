"""Circuit identity, seeding, and split membership shared by training and evaluation.

A circuit is named by a *logical identity* inside a split namespace, and the
track generator's seed is derived from that pair. The indirection is what lets
two runs agree on which circuit they mean: paired observation conditions draw
the same ordered identities from identically seeded streams, and a saved
validation or test circuit keeps its identity even though the integer the
generator consumed is an implementation detail recorded beside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial

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


def generated_evaluation_circuits(
    identities: Iterable[int],
    *,
    namespace: SeedNamespace,
    split: CircuitSplit,
    environment_config: EnvironmentConfig,
) -> tuple[EvaluationCircuit, ...]:
    """
    Prepare one evaluation circuit per identity in a split.

    Each circuit is generated once here rather than inside its factory: a run
    rebuilds every one of them at each of forty checkpoints, and regenerating
    identical geometry that many times would cost more than the evaluation.

    Evaluation always launches from the canonical start line, so the returned
    environments never sample a start pose regardless of the training setting.
    """
    evaluation_config = replace(
        environment_config, start=StartStateConfig(randomized=False)
    )
    circuits: list[EvaluationCircuit] = []
    for identity in identities:
        track = TrackWithGeometry.generate(
            circuit_track_seed(namespace, identity),
            track_config=environment_config.track,
            vehicle_config=environment_config.vehicle,
        )
        circuits.append(
            EvaluationCircuit(
                identity=str(identity),
                split=split,
                factory=partial(RacingEnv, track, config=evaluation_config),
            )
        )
    return tuple(circuits)
