"""Immutable configuration matrices for the two reported experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .choices import Algorithm, ObservationRepresentation
from .serialization import SerializableConfig
from .training import (
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    ActorConfig,
)


@dataclass(frozen=True, slots=True)
class ExperimentMatricesConfig(SerializableConfig):
    """
    Collect the machine-readable matrices for both reported experiments.

    Learning-rate calibration is intentionally absent: the calibration runner
    will try the candidates declared by each algorithm without needing a second
    configuration object or a separate training loop.

    Fields:
        * experiment_1: Fixed-circuit actor-size matrix.
        * experiment_2: PPO observation-generalization matrix.
    """

    experiment_1: Experiment1MatrixConfig = field(
        default_factory=lambda: Experiment1MatrixConfig()
    )
    experiment_2: Experiment2MatrixConfig = field(
        default_factory=lambda: Experiment2MatrixConfig()
    )


@dataclass(frozen=True, slots=True)
class Experiment1MatrixConfig(SerializableConfig):
    """
    Configure the fixed-circuit actor-size experiment matrix.

    Fields:
        * algorithms: Algorithms occupying the matrix rows.
        * actors: Named actor configurations occupying the columns.
        * root_identities: Paired reported-run identities.
        * track_path: Saved fixed circuit used by every run.
        * observation: Observation representation used by every run.
    """

    algorithms: tuple[Algorithm, ...] = (
        Algorithm.REINFORCE,
        Algorithm.A2C,
        Algorithm.PPO,
    )
    actors: tuple[ActorConfig, ...] = (
        SMALL_ACTOR_CONFIG,
        MEDIUM_ACTOR_CONFIG,
        LARGE_ACTOR_CONFIG,
    )
    root_identities: tuple[int, ...] = (0, 1, 2, 3, 4)
    track_path: str = "tracks/experiment_1.json"
    observation: ObservationRepresentation = ObservationRepresentation.FRENET


@dataclass(frozen=True, slots=True)
class Experiment2MatrixConfig(SerializableConfig):
    """
    Configure the paired PPO observation-generalization experiment matrix.

    Fields:
        * algorithm: Fixed learning algorithm.
        * actor_selection_rule: Predeclared source of the shared actor architecture.
        * observations: Paired observation conditions.
        * root_identities: Paired reported-run identities.
        * development_circuit_count: Circuits used only before reported runs.
        * training_reference_circuit_count: In-sample circuits revisited at the end.
        * validation_circuit_count: Fixed validation-circuit count.
        * test_circuit_count: Fixed held-out test-circuit count.
    """

    algorithm: Algorithm = Algorithm.PPO
    actor_selection_rule: Literal["experiment_1_ppo_parsimony_rule"] = (
        "experiment_1_ppo_parsimony_rule"
    )
    observations: tuple[ObservationRepresentation, ...] = (
        ObservationRepresentation.FRENET,
        ObservationRepresentation.LIDAR,
    )
    root_identities: tuple[int, ...] = (0, 1, 2, 3, 4)
    development_circuit_count: int = 8
    # Matched to the validation count so that the training-reference and
    # validation summaries rest on the same denominator: their gaps to the test
    # split are then comparable to each other rather than to their own sizes.
    training_reference_circuit_count: int = 16
    validation_circuit_count: int = 16
    test_circuit_count: int = 32
