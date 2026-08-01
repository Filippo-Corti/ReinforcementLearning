"""Gymnasium environment combining a prepared track and racing dynamics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from configs import EnvironmentConfig

from ..tracks import Track, TrackWithGeometry
from ..vehicle import NormalizedAction, VehicleState, transition
from .lifecycle import ActionOutcome, EpisodeLifecycle
from .rendering import RacingPygameRenderer

ActionType = NDArray[np.float32]
ObservationType = NDArray[np.float32]


class RacingEnv(gym.Env[ObservationType, ActionType]):
    """
    Expose continuous racing controls and Frenet observations through Gymnasium.

    Fields:
        * config: The environment behaviour configuration.
        * track: The immutable sampled circuit data.
        * track_with_geometry: The track's interpolation, boundaries, and indexes.
        * action_space: Normalized throttle/brake and steering controls.
        * observation_space: Frenet observations in float32 physical units.
        * state: The current kinematic vehicle state.
    """

    # Gymnasium and its wrappers inspect this class-level mapping to discover
    # which render modes the environment supports. It does not affect dynamics.
    metadata = {"render_modes": ["human", "rgb_array"]}  # noqa: RUF012

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        config: EnvironmentConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'.")

        self.config = config or EnvironmentConfig()
        self.track_with_geometry = track
        self.track: Track = track.track
        self.state: VehicleState | None = None
        self._lifecycle: EpisodeLifecycle | None = None
        self._episode_finished = False
        self.render_mode = render_mode
        self._renderer: RacingPygameRenderer | None = None

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-np.inf, -np.pi, 0.0, -np.inf], dtype=np.float32),
            high=np.asarray(
                [np.inf, np.pi, self.config.vehicle.max_speed, np.inf],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """
        Reset at the prepared track's canonical start line with zero speed.
        """
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        start_s = self.track.s[self.track.start_index]
        start_position = self.track_with_geometry.position(float(start_s))
        self.state = VehicleState(
            x=float(start_position[0]),
            y=float(start_position[1]),
            heading=self.track_with_geometry.heading(float(start_s)),
            speed=0.0,
        )
        self._lifecycle = EpisodeLifecycle(
            self.track_with_geometry,
            simulation_config=self.config.simulation,
            vehicle_config=self.config.vehicle,
            reward_config=self.config.reward,
        )
        self._lifecycle.reset(self.state)
        self._episode_finished = False
        return self._observe(), self._info()

    def step(
        self,
        action: ActionType,
    ) -> tuple[ObservationType, float, bool, bool, dict[str, Any]]:
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

        vehicle_transition = transition(
            self.state,
            NormalizedAction(
                throttle=float(action_values[0]),
                steering=float(action_values[1]),
            ),
            simulation_config=self.config.simulation,
            vehicle_config=self.config.vehicle,
        )
        outcome = self._lifecycle.process_transition(vehicle_transition)
        if outcome.termination_substep is None:
            self.state = vehicle_transition.state
        else:
            self.state = vehicle_transition.substep_states[
                outcome.termination_substep - 1
            ]
        self._episode_finished = outcome.terminated or outcome.truncated
        return (
            self._observe(),
            outcome.reward,
            outcome.terminated,
            outcome.truncated,
            self._info(outcome),
        )

    def render(self) -> NDArray[np.uint8] | None:
        """
        Render the current track and vehicle state in the configured mode.
        """
        if self.render_mode is None:
            return None
        if self.state is None:
            raise RuntimeError("reset must be called before rendering the environment.")
        if self._renderer is None:
            self._renderer = RacingPygameRenderer(
                self.track_with_geometry,
                render_mode=self.render_mode,
                image_size=(800, 800),
            )
        return self._renderer.render(self.state)

    def close(self) -> None:
        """
        Release environment rendering resources.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observe(self) -> ObservationType:
        """
        Build the current Frenet observation in the declared Gymnasium dtype.
        """
        if self.state is None or self._lifecycle is None:
            raise RuntimeError("environment state is not initialized.")
        observation, _ = self._lifecycle.observer.observe(
            self.state,
            config=self.config.observation,
        )
        return observation.as_array().astype(np.float32)

    def _info(self, outcome: ActionOutcome | None = None) -> dict[str, Any]:
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
            "track_seed": self.track.generation.seed,
            "collision_substep": None if outcome is None else outcome.collision_substep,
        }
