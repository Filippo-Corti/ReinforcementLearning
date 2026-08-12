"""Integration tests for the Gymnasium racing environment."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from configs import EnvironmentConfig, SimulationConfig, StartStateConfig
from envs import RacingEnv, TrackWithGeometry


def _environment(seed: int, **kwargs) -> RacingEnv:
    """
    Build an environment from a track prepared outside its constructor.
    """
    return RacingEnv(TrackWithGeometry.generate(seed), **kwargs)


def test_racing_env_passes_gymnasium_checker() -> None:
    """
    The assembled environment satisfies Gymnasium's API contract.
    """
    check_env(_environment(0), skip_render_check=True)


def test_seeded_action_sequences_are_reproducible() -> None:
    """
    Matching reset seeds and actions produce matching trajectories.
    """
    actions = [
        np.asarray([1.0, 0.1], dtype=np.float32),
        np.asarray([0.4, -0.2], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    ]
    first = _environment(12)
    second = _environment(12)
    first_observation, first_info = first.reset(seed=12)
    second_observation, second_info = second.reset(seed=12)

    assert np.array_equal(first_observation, second_observation)
    assert first_info["track_seed"] == second_info["track_seed"] == 12
    for action in actions:
        first_result = first.step(action)
        second_result = second.step(action)
        assert np.array_equal(first_result[0], second_result[0])
        assert first_result[1:] == second_result[1:]


def test_reset_clears_episode_state_and_observation_is_in_space() -> None:
    """
    Reset clears lifecycle counters and returns a declared-space observation.
    """
    environment = _environment(2)
    observation, info = environment.reset()
    environment.step(np.asarray([1.0, 0.0], dtype=np.float32))
    reset_observation, reset_info = environment.reset()

    assert environment.observation_space.contains(observation)
    assert environment.observation_space.contains(reset_observation)
    assert info["elapsed_time"] == 0.0
    assert reset_info["elapsed_time"] == 0.0
    assert reset_info["episode_progress"] == 0.0


def test_terminal_observation_and_time_limit_flags_are_valid() -> None:
    """
    A truncated transition returns a valid terminal observation.
    """
    environment = RacingEnv(
        TrackWithGeometry.generate(3),
        config=EnvironmentConfig(
            simulation=SimulationConfig(max_episode_steps=1),
            start=StartStateConfig(randomized=False),
        ),
    )
    environment.reset()

    observation, _, terminated, truncated, info = environment.step(
        np.zeros(2, dtype=np.float32)
    )

    assert environment.observation_space.contains(observation)
    assert not terminated
    assert truncated
    assert info["elapsed_time"] == 0.04


def test_saved_track_can_be_loaded() -> None:
    """
    A persisted circuit can be prepared before environment construction.
    """
    track_path = Path(__file__).parents[1] / "fixtures" / "tracks" / "valid_circle.json"
    environment = RacingEnv(TrackWithGeometry.load(track_path))

    observation, info = environment.reset(seed=99)

    assert environment.observation_space.contains(observation)
    assert info["track_seed"] == environment.track.generation.seed


def test_direct_construction_is_a_gymnasium_environment() -> None:
    """
    The public class can be used without environment registration.
    """
    assert isinstance(_environment(4), gym.Env)
