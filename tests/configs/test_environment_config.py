"""Tests for environment configuration defaults and validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from configs import (
    EnvironmentConfig,
    FrenetObservationConfig,
    RewardConfig,
    SimulationConfig,
    TrackGenerationConfig,
    VehicleConfig,
)


def test_defaults_match_the_environment_specification() -> None:
    config = EnvironmentConfig()

    assert config.simulation == SimulationConfig(
        agent_timestep_s=0.04,
        physics_timestep_s=0.01,
        physics_substeps=4,
        max_episode_steps=5_000,
    )
    assert config.vehicle == VehicleConfig(
        wheelbase_m=3.6,
        max_acceleration_m_per_s2=9.26,
        max_steering_angle_deg=30.0,
        max_speed_m_per_s=70.0,
    )
    assert config.track == TrackGenerationConfig(
        n_checkpoints=12,
        base_radius_m=250.0,
        radial_jitter_fraction=0.25,
        angular_jitter_sectors=0.25,
        sample_spacing_m=0.5,
        width_m=12.0,
        max_attempts=100,
        min_length_m=1_000.0,
        max_length_m=3_000.0,
        nonlocal_centerline_margin_m=2.0,
    )
    assert config.reward == RewardConfig(
        finish_reward=10.0,
        crash_penalty=20.0,
        time_penalty_rate_per_s=0.05,
        progress_coefficient=1.0,
    )
    assert config.observation == FrenetObservationConfig(
        lookahead_base_m=5.0,
        lookahead_speed_factor_s=0.7,
    )


def test_timestep_relationship_is_enforced() -> None:
    with pytest.raises(
        ValueError,
        match=r"agent_timestep_s must equal physics_timestep_s \* physics_substeps",
    ):
        SimulationConfig(
            agent_timestep_s=0.04,
            physics_timestep_s=0.02,
            physics_substeps=4,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SimulationConfig(agent_timestep_s=0.0), "agent_timestep_s"),
        (
            lambda: SimulationConfig(physics_timestep_s=float("nan")),
            "physics_timestep_s",
        ),
        (lambda: SimulationConfig(physics_substeps=0), "physics_substeps"),
        (lambda: SimulationConfig(max_episode_steps=True), "max_episode_steps"),
        (lambda: VehicleConfig(wheelbase_m=0.0), "wheelbase_m"),
        (
            lambda: VehicleConfig(max_acceleration_m_per_s2=float("inf")),
            "max_acceleration_m_per_s2",
        ),
        (
            lambda: VehicleConfig(max_steering_angle_deg=90.0),
            "max_steering_angle_deg",
        ),
        (lambda: VehicleConfig(max_speed_m_per_s=-1.0), "max_speed_m_per_s"),
        (lambda: TrackGenerationConfig(n_checkpoints=2), "n_checkpoints"),
        (lambda: TrackGenerationConfig(base_radius_m=0.0), "base_radius_m"),
        (
            lambda: TrackGenerationConfig(radial_jitter_fraction=1.0),
            "radial_jitter_fraction",
        ),
        (
            lambda: TrackGenerationConfig(angular_jitter_sectors=0.5),
            "angular_jitter_sectors",
        ),
        (lambda: TrackGenerationConfig(sample_spacing_m=0.0), "sample_spacing_m"),
        (lambda: TrackGenerationConfig(width_m=-1.0), "width_m"),
        (lambda: TrackGenerationConfig(max_attempts=False), "max_attempts"),
        (lambda: TrackGenerationConfig(min_length_m=0.0), "min_length_m"),
        (lambda: TrackGenerationConfig(max_length_m=float("nan")), "max_length_m"),
        (
            lambda: TrackGenerationConfig(
                min_length_m=2_000.0,
                max_length_m=2_000.0,
            ),
            "min_length_m",
        ),
        (
            lambda: TrackGenerationConfig(nonlocal_centerline_margin_m=-1.0),
            "nonlocal_centerline_margin_m",
        ),
        (lambda: RewardConfig(finish_reward=0.0), "finish_reward"),
        (lambda: RewardConfig(crash_penalty=float("inf")), "crash_penalty"),
        (
            lambda: RewardConfig(time_penalty_rate_per_s=-0.01),
            "time_penalty_rate_per_s",
        ),
        (
            lambda: RewardConfig(progress_coefficient=float("nan")),
            "progress_coefficient",
        ),
        (
            lambda: FrenetObservationConfig(lookahead_base_m=0.0),
            "lookahead_base_m",
        ),
        (
            lambda: FrenetObservationConfig(lookahead_speed_factor_s=-0.1),
            "lookahead_speed_factor_s",
        ),
    ],
)
def test_invalid_values_raise_clear_errors(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_configuration_is_immutable() -> None:
    config = EnvironmentConfig()

    with pytest.raises(FrozenInstanceError):
        config.simulation.agent_timestep_s = 0.08  # type: ignore[misc]


def test_serialization_is_plain_deterministic_and_json_compatible() -> None:
    config = EnvironmentConfig()

    first = config.to_dict()
    second = config.to_dict()

    assert first == second
    assert type(first) is dict
    assert all(type(section) is dict for section in first.values())
    assert json.loads(json.dumps(first, sort_keys=True)) == first
