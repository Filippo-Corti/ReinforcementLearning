"""Immutable configuration matrices for the two reported experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .serialization import SerializableConfig
from .training import (
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    ActorConfig,
)


@dataclass(frozen=True, slots=True)
class LearningRateCalibrationConfig(SerializableConfig):
    """
    Fixed protocol for resolving documented learning-rate candidates.

    Fields:
        * interactions_per_candidate: Training budget for each candidate/root run.
        * root_identities: Dedicated root identities used by calibration.
        * actor: Actor architecture fixed during calibration.
    """

    interactions_per_candidate: int = 250_000
    root_identities: tuple[int, ...] = (0, 1, 2)
    actor: ActorConfig = field(default_factory=lambda: MEDIUM_ACTOR_CONFIG)


@dataclass(frozen=True, slots=True)
class Experiment1MatrixConfig(SerializableConfig):
    """
    Complete reported-run matrix for the fixed-circuit actor-size experiment.

    Fields:
        * algorithms: Algorithms occupying the matrix rows.
        * actors: Named actor configurations occupying the matrix columns.
        * root_identities: Paired reported-run identities.
        * track_path: Saved fixed circuit used by every run.
        * observation: Observation representation for every run.
    """

    algorithms: tuple[Literal["reinforce", "a2c", "ppo"], ...] = (
        "reinforce",
        "a2c",
        "ppo",
    )
    actors: tuple[ActorConfig, ...] = (
        SMALL_ACTOR_CONFIG,
        MEDIUM_ACTOR_CONFIG,
        LARGE_ACTOR_CONFIG,
    )
    root_identities: tuple[int, ...] = (0, 1, 2, 3, 4)
    track_path: str = "tracks/experiment_1.json"
    observation: Literal["frenet"] = "frenet"


@dataclass(frozen=True, slots=True)
class Experiment2MatrixConfig(SerializableConfig):
    """
    Complete reported-run matrix for PPO observation generalization.

    Fields:
        * algorithm: The fixed learning algorithm.
        * actor_selection_rule: Predeclared source of the shared actor architecture.
        * observations: Paired observation conditions.
        * root_identities: Paired reported-run identities.
        * development_circuit_count: Circuits used only before reported runs.
        * validation_circuit_count: Fixed validation-circuit count.
        * test_circuit_count: Fixed held-out test-circuit count.
    """

    algorithm: Literal["ppo"] = "ppo"
    actor_selection_rule: Literal["experiment_1_ppo_parsimony_rule"] = (
        "experiment_1_ppo_parsimony_rule"
    )
    observations: tuple[Literal["frenet", "lidar"], ...] = ("frenet", "lidar")
    root_identities: tuple[int, ...] = (0, 1, 2, 3, 4)
    development_circuit_count: int = 8
    validation_circuit_count: int = 16
    test_circuit_count: int = 32


@dataclass(frozen=True, slots=True)
class ExperimentMatricesConfig(SerializableConfig):
    """
    All machine-readable experiment matrices and calibration protocol.

    Fields:
        * learning_rate_calibration: Protocol for unresolved learning rates.
        * experiment_1: Fixed-circuit actor-size matrix.
        * experiment_2: PPO observation-generalization matrix.
    """

    learning_rate_calibration: LearningRateCalibrationConfig = field(
        default_factory=LearningRateCalibrationConfig
    )
    experiment_1: Experiment1MatrixConfig = field(
        default_factory=Experiment1MatrixConfig
    )
    experiment_2: Experiment2MatrixConfig = field(
        default_factory=Experiment2MatrixConfig
    )
