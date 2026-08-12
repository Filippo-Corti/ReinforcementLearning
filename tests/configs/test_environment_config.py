"""Tests for the environment configuration values."""

from __future__ import annotations

import json

from configs import (
    CarConfig,
    EnvironmentConfig,
    FrenetObservationConfig,
    RewardConfig,
    SimulationConfig,
    StartStateConfig,
    TrackGenerationConfig,
)


def test_defaults_match_the_environment_specification() -> None:
    config = EnvironmentConfig()

    assert config.simulation == SimulationConfig(
        agent_timestep=0.04,
        physics_timestep=0.01,
        physics_substeps=4,
        max_episode_steps=1_000,
        stall_time=3.0,
        stall_progress=1.0,
    )
    assert config.vehicle == CarConfig(
        wheelbase=3.6,
        max_acceleration=9.26,
        max_braking=20.0,
        max_steering_angle=30.0,
        max_steering_rate=180.0,
        max_lateral_acceleration=20.0,
        max_speed=70.0,
    )
    assert config.track == TrackGenerationConfig(
        n_checkpoints=12,
        base_radius=50.0,
        radial_jitter=0.25,
        angular_jitter=0.25,
        sample_spacing=0.5,
        width=12.0,
        max_attempts=100,
        min_length=200.0,
        max_length=600.0,
        nonlocal_centerline_margin=2.0,
    )
    assert config.reward == RewardConfig(
        finish_reward=100.0,
        lap_time_bonus=100.0,
        crash_penalty=5.0,
        time_penalty_rate=1.0,
        progress_coefficient=100.0,
    )
    assert config.observation == FrenetObservationConfig(
        lookahead_base=5.0,
        lookahead_speed_factor=0.7,
    )
    assert config.start == StartStateConfig(
        randomized=True,
        lateral_fraction=0.5,
        heading_error=0.2,
        speed_fraction=0.15,
    )


def test_serialization_is_plain_and_json_compatible() -> None:
    serialized = EnvironmentConfig().to_dict()

    assert serialized["simulation"]["physics_substeps"] == 4
    assert serialized["vehicle"]["max_speed"] == 70.0
    assert json.loads(json.dumps(serialized)) == serialized
