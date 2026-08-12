"""Tests for immutable learning and experiment configuration contracts."""

from __future__ import annotations

import json

import pytest

from configs import (
    FIXED_CRITIC_CONFIG,
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    A2CConfig,
    Algorithm,
    ExperimentMatricesConfig,
    ObservationRepresentation,
    PPOConfig,
    ReinforceConfig,
    TrainingConfig,
    physical_cpu_count,
)


def test_training_configuration_serializes_stably_with_literal_actor_widths() -> None:
    configuration = TrainingConfig(actor=SMALL_ACTOR_CONFIG)

    first = configuration.to_dict()
    second = configuration.to_dict()

    assert first == second
    assert first["actor"]["hidden_sizes"] == [32, 32]
    assert first["actor"]["learning_rate"] is None
    assert first["critic"]["hidden_sizes"] == [64, 64]
    assert first["critic"]["hidden_initialization_gain"] == 2**0.5
    assert first["training_interaction_budget"] == 2_000_000
    assert first["checkpoint_interval"] == 250_000
    assert first["evaluation"]["evaluation_interval"] == 50_000
    assert first["execution"]["environment_workers"] == physical_cpu_count()
    assert not first["ppo"]["learning_rate_scheduler_enabled"]
    assert json.loads(json.dumps(first, sort_keys=True)) == first


def test_configurations_are_immutable() -> None:
    configuration = TrainingConfig(actor=MEDIUM_ACTOR_CONFIG)

    with pytest.raises(AttributeError):
        configuration.actor = LARGE_ACTOR_CONFIG  # type: ignore[misc]


def test_agent_configs_expose_only_documented_learning_rate_candidates() -> None:
    reinforce = ReinforceConfig()
    a2c = A2CConfig()
    ppo = PPOConfig()

    documented_pairs = (
        (1e-4, 3e-4),
        (3e-4, 1e-3),
        (3e-4, 3e-3),
        (3e-4, 1e-2),
    )

    assert reinforce.actor_learning_rate_candidates == (1e-4, 3e-4, 1e-3)
    # A2C and PPO are offered the same grid so that neither is denied an option
    # the other is given; each still selects its own pair.
    assert a2c.learning_rate_candidates == documented_pairs
    assert ppo.learning_rate_candidates == documented_pairs
    assert not hasattr(reinforce, "actor_learning_rate")
    assert not hasattr(a2c, "actor_learning_rate")
    assert not hasattr(ppo, "critic_learning_rate")


def test_actor_learning_rate_is_unset_in_size_presets() -> None:
    assert SMALL_ACTOR_CONFIG.learning_rate is None


def test_experiment_matrices_cover_approved_choices() -> None:
    matrices = ExperimentMatricesConfig()
    experiment_1 = matrices.experiment_1
    experiment_2 = matrices.experiment_2

    assert experiment_1.algorithms == (
        Algorithm.REINFORCE,
        Algorithm.A2C,
        Algorithm.PPO,
    )
    assert tuple(actor.hidden_sizes for actor in experiment_1.actors) == (
        (32, 32),
        (64, 64),
        (256, 256),
    )
    assert experiment_1.root_identities == (0, 1, 2, 3, 4)
    assert experiment_2.observations == (
        ObservationRepresentation.FRENET,
        ObservationRepresentation.LIDAR,
    )
    assert experiment_2.root_identities == (0, 1, 2, 3, 4)
    assert experiment_2.validation_circuit_count == 16
    assert experiment_2.test_circuit_count == 32
    assert not hasattr(matrices, "learning_rate_calibration")
    assert FIXED_CRITIC_CONFIG.hidden_sizes == (64, 64)
