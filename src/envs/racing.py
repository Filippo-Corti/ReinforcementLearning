"""Gymnasium environment that combines track, dynamics and lifecycle rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from configs import EnvironmentConfig

from .dynamics import NormalizedAction, VehicleState, transition
from .geometry import TrackGeometry
from .lifecycle import EpisodeLifecycle
from .track import Track
from .track_generation import generate_track


class RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Continuous-control Formula 1 racing environment with Frenet observations.
    It generates or loads one track, advances the kinematic vehicle state and
    delegates episode outcomes to the lifecycle component.

    Fields:
        * config: The immutable configuration governing environment behaviour.
        * track: The loaded or generated circuit used by the current environment.
        * geometry: Runtime geometry derived from the current track.
        * action_space: Normalized throttle/brake and steering controls.
        * observation_space: Frenet observations in float32 physical units.
        * state: The current kinematic vehicle state.
    """

    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        *,
        config: EnvironmentConfig | None = None,
        track: Track | None = None,
        track_path: str | Path | None = None,
        track_seed: int | None = None,
    ) -> None:
        super().__init__()
        if track is not None and track_path is not None:
            raise ValueError("provide either track or track_path, not both.")
        if track is not None and track_seed is not None:
            raise ValueError("track_seed cannot be combined with an explicit track.")
        if track_path is not None and track_seed is not None:
            raise ValueError("track_seed cannot be combined with track_path.")

        self.config = config or EnvironmentConfig()
        if track_path is not None:
            track = Track.load(
                track_path,
                vehicle_config=self.config.vehicle,
                track_config=self.config.track,
            )
        if track is None and track_seed is not None:
            track = generate_track(
                track_seed,
                track_config=self.config.track,
                vehicle_config=self.config.vehicle,
            )

        self.track = track
        self.geometry: TrackGeometry | None = None
        self.state: VehicleState | None = None
        self._lifecycle: EpisodeLifecycle | None = None
        self._track_seed = None if track is None else track.generation.seed
        self._generated_from_reset_seed = track is None
        self._episode_finished = False

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-np.inf, -np.pi, 0.0, -np.inf], dtype=np.float32),
            high=np.asarray(
                [np.inf, np.pi, self.config.vehicle.max_speed, np.inf], dtype=np.float32
            ),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset at the canonical start line with zero speed.
        """
        super().reset(seed=seed)
        self._ensure_track(seed)
        if self.track is None:
            raise RuntimeError("track initialization failed.")

        self.geometry = TrackGeometry(self.track)
        start_s = self.track.s[self.track.start_index]
        start_position = self.geometry.position(float(start_s))
        self.state = VehicleState(
            x=float(start_position[0]),
            y=float(start_position[1]),
            heading=self.geometry.heading(float(start_s)),
            speed=0.0,
        )
        self._lifecycle = EpisodeLifecycle(
            self.geometry,
            simulation_config=self.config.simulation,
            vehicle_config=self.config.vehicle,
            reward_config=self.config.reward,
        )
        self._lifecycle.reset(self.state)
        self._episode_finished = False
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Apply one normalized control action and return the Gymnasium transition.
        """
        if self.state is None or self._lifecycle is None:
            raise RuntimeError("reset must be called before stepping the environment.")
        if self._episode_finished:
            raise RuntimeError(
                "reset must be called after a terminated or truncated episode."
            )
        action_values = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action_values):
            raise ValueError(
                "action must be a float32 array with shape (2,) in [-1, 1]."
            )

        dynamics = transition(
            self.state,
            NormalizedAction(
                throttle=float(action_values[0]),
                steering=float(action_values[1]),
            ),
            simulation_config=self.config.simulation,
            vehicle_config=self.config.vehicle,
        )
        outcome = self._lifecycle.advance(dynamics)
        if outcome.termination_substep is None:
            self.state = dynamics.state
        else:
            self.state = dynamics.substep_states[outcome.termination_substep - 1]
        self._episode_finished = outcome.terminated or outcome.truncated
        return (
            self._observation(),
            outcome.reward,
            outcome.terminated,
            outcome.truncated,
            self._info(outcome),
        )

    def close(self) -> None:
        """
        Release environment resources.
        """

    def _ensure_track(self, seed: int | None) -> None:
        """
        Generate a track when one was not supplied by the constructor.
        """
        if self.track is not None and not (
            self._generated_from_reset_seed and seed is not None
        ):
            return
        if seed is not None:
            self._track_seed = seed
        elif self._track_seed is None:
            self._track_seed = int(self.np_random.integers(np.iinfo(np.int64).max))
        self.track = generate_track(
            self._track_seed,
            track_config=self.config.track,
            vehicle_config=self.config.vehicle,
        )

    def _observation(self) -> np.ndarray:
        """
        Build the current Frenet observation in the declared dtype.
        """
        if self.state is None or self._lifecycle is None:
            raise RuntimeError("environment state is not initialized.")
        observation, _ = self._lifecycle.projector.observation(
            np.asarray([self.state.x, self.state.y], dtype=np.float64),
            vehicle_heading=self.state.heading,
            speed=self.state.speed,
        )
        return observation.astype(np.float32)

    def _info(self, outcome: Any | None = None) -> dict[str, Any]:
        """
        Return diagnostics shared by reset and step results.
        """
        if self._lifecycle is None:
            raise RuntimeError("environment lifecycle is not initialized.")
        return {
            "wrapped_progress": self._lifecycle.wrapped_progress,
            "episode_progress": self._lifecycle.episode_progress,
            "collision": False if outcome is None else outcome.collision,
            "lap_completed": False if outcome is None else outcome.lap_completed,
            "elapsed_time": self._lifecycle.agent_steps
            * self.config.simulation.agent_timestep,
            "track_seed": self._track_seed,
            "collision_substep": None if outcome is None else outcome.collision_substep,
        }
