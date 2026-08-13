"""Gymnasium environment combining a prepared track and racing dynamics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from configs import EnvironmentConfig, ObservationRepresentation, RenderStyle

from ..observations import LidarObserver
from ..tracks import Track, TrackWithGeometry
from ..vehicle import NormalizedAction, VehicleState, transition
from .lifecycle import ActionOutcome, EpisodeLifecycle, EpisodeLifecycleState
from .rendering import RacingPygameRenderer, RenderFrame

ActionType = NDArray[np.float32]
ObservationType = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RacingEnvState:
    """
    Store focused non-rendering environment state for exact collection resume.

    Fields:
        * vehicle_state: Current kinematic state, or none before reset.
        * lifecycle_state: Mutable progress state paired with the vehicle state.
        * episode_finished: Whether reset is required before another action.
        * numpy_random_state: Gymnasium reset-generator state.
    """

    vehicle_state: VehicleState | None
    lifecycle_state: EpisodeLifecycleState | None
    episode_finished: bool
    numpy_random_state: dict[str, Any]


class RacingEnv(gym.Env[ObservationType, ActionType]):
    """
    Expose continuous racing controls and Frenet observations through Gymnasium.

    Fields:
        * config: The environment behaviour configuration.
        * track: The immutable sampled circuit data.
        * track_with_geometry: The track's interpolation, boundaries, and indexes.
        * action_space: Normalized throttle/brake and steering controls.
        * observation_space: The selected observation in float32 units.
        * lidar_observer: Ray caster, present only under LiDAR observation.
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
        render_style: RenderStyle = RenderStyle.BROADCAST,
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
        self.render_style = RenderStyle(render_style)
        self._renderer: RacingPygameRenderer | None = None
        # Kept for drawing only. The simulation never reads it back: an action
        # is applied when it arrives and nothing later depends on remembering it.
        self._last_action = np.zeros(2, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.lidar_observer = (
            LidarObserver(
                track,
                observation_config=self.config.lidar,
                vehicle_config=self.config.vehicle,
            )
            if self.config.observation_type is ObservationRepresentation.LIDAR
            else None
        )
        self.observation_space = self._observation_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """
        Reset at a sampled start pose, or the canonical start line when fixed.
        """
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        self.state = self._sample_start_state()
        self._last_action = np.zeros(2, dtype=np.float32)
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
        self._last_action = action_values.copy()
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
        if self.state is None or self._lifecycle is None:
            raise RuntimeError("reset must be called before rendering the environment.")
        if self._renderer is None:
            self._renderer = RacingPygameRenderer(
                self.track_with_geometry,
                render_mode=self.render_mode,
                image_size=(800, 800),
                style=self.render_style,
                vehicle_config=self.config.vehicle,
            )
        return self._renderer.render(
            RenderFrame(
                state=self.state,
                gate_s=self._lifecycle.gate_s,
                elapsed_time=self._lifecycle.agent_steps
                * self.config.simulation.agent_timestep,
                progress=self._lifecycle.episode_progress / self.track.track_length,
                throttle=float(self._last_action[0]),
                steering=float(self._last_action[1]),
            )
        )

    def close(self) -> None:
        """
        Release environment rendering resources.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def snapshot(self) -> RacingEnvState:
        """
        Return a focused dynamics snapshot without including rendering resources.
        """
        if (self.state is None) != (self._lifecycle is None):
            raise RuntimeError(
                "RacingEnv state and lifecycle must be initialized together."
            )
        return RacingEnvState(
            vehicle_state=self.state,
            lifecycle_state=(
                None if self._lifecycle is None else self._lifecycle.state()
            ),
            episode_finished=self._episode_finished,
            numpy_random_state=deepcopy(dict(self.np_random.bit_generator.state)),
        )

    def restore(self, state: RacingEnvState) -> None:
        """
        Restore a focused dynamics snapshot without recreating rendering state.
        """
        if (state.vehicle_state is None) != (state.lifecycle_state is None):
            raise ValueError(
                "RacingEnv snapshots require vehicle and lifecycle together."
            )
        generator = np.random.default_rng()
        generator.bit_generator.state = state.numpy_random_state
        self.np_random = generator
        self.state = state.vehicle_state
        self._episode_finished = state.episode_finished
        if state.vehicle_state is None:
            self._lifecycle = None
            return
        self._lifecycle = EpisodeLifecycle(
            self.track_with_geometry,
            simulation_config=self.config.simulation,
            vehicle_config=self.config.vehicle,
            reward_config=self.config.reward,
        )
        if state.lifecycle_state is None:
            raise ValueError("RacingEnv snapshot lifecycle state is missing.")
        self._lifecycle.restore(state.lifecycle_state)

    def _sample_start_state(self) -> VehicleState:
        """
        Draw the pose the episode begins from using only this env's generator.

        The lap always spans a full circuit from wherever the car is placed, so
        the sampled arc length moves the finish line with the start rather than
        shortening or lengthening the lap.
        """
        start = self.config.start
        geometry = self.track_with_geometry
        if not start.randomized:
            arc_length = float(self.track.s[self.track.start_index])
            return VehicleState(
                x=float(geometry.position(arc_length)[0]),
                y=float(geometry.position(arc_length)[1]),
                heading=geometry.heading(arc_length),
                speed=0.0,
            )

        arc_length = float(self.np_random.uniform(0.0, self.track.track_length))
        lateral_limit = start.lateral_fraction * self.track.width / 2.0
        lateral_offset = float(self.np_random.uniform(-lateral_limit, lateral_limit))
        position = geometry.position(arc_length) + lateral_offset * geometry.normal(
            arc_length
        )
        return VehicleState(
            x=float(position[0]),
            y=float(position[1]),
            heading=geometry.heading(arc_length)
            + float(self.np_random.uniform(-start.heading_error, start.heading_error)),
            speed=float(
                self.np_random.uniform(
                    0.0, start.speed_fraction * self.config.vehicle.max_speed
                )
            ),
        )

    def _observation_space(self) -> gym.spaces.Box:
        """
        Declare the bounds of whichever representation the agent observes.
        """
        steering_limit = np.radians(self.config.vehicle.max_steering_angle)
        max_speed = self.config.vehicle.max_speed
        if self.lidar_observer is not None:
            ray_count = self.lidar_observer.dimensions - 2
            return gym.spaces.Box(
                low=np.asarray(
                    [0.0, -steering_limit] + [0.0] * ray_count, dtype=np.float32
                ),
                high=np.asarray(
                    [max_speed, steering_limit] + [1.0] * ray_count, dtype=np.float32
                ),
                dtype=np.float32,
            )
        return gym.spaces.Box(
            low=np.asarray(
                [-np.inf, -np.pi, 0.0, -steering_limit, -np.inf], dtype=np.float32
            ),
            high=np.asarray(
                [np.inf, np.pi, max_speed, steering_limit, np.inf],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def _observe(self) -> ObservationType:
        """
        Build the selected observation in the declared Gymnasium dtype.

        The Frenet observer runs either way inside the lifecycle, because
        progress and collision are defined by the centerline projection. What
        the observation type decides is only what the agent is shown.
        """
        if self.state is None or self._lifecycle is None:
            raise RuntimeError("environment state is not initialized.")
        if self.lidar_observer is not None:
            return self.lidar_observer.observe(self.state).as_array().astype(np.float32)
        observation, _ = self._lifecycle.observer.observe(
            self.state,
            previous_segment_index=self._lifecycle.current_segment_index,
            config=self.config.frenet,
        )
        return observation.as_array().astype(np.float32)

    def _info(self, outcome: ActionOutcome | None = None) -> dict[str, Any]:
        """
        Return diagnostics shared by reset and step results.
        """
        if self._lifecycle is None or self.state is None:
            raise RuntimeError("environment lifecycle is not initialized.")
        return {
            # Reported here rather than read out of the observation vector,
            # which places speed at a different index in each representation.
            "speed": self.state.speed,
            "wrapped_progress": self._lifecycle.wrapped_progress,
            "episode_progress": self._lifecycle.episode_progress,
            "collision": False if outcome is None else outcome.collision,
            "lap_completed": False if outcome is None else outcome.lap_completed,
            "stalled": False if outcome is None else outcome.stalled,
            "elapsed_time": self._lifecycle.agent_steps
            * self.config.simulation.agent_timestep,
            "track_seed": self.track.generation.seed,
            "collision_substep": None if outcome is None else outcome.collision_substep,
        }
