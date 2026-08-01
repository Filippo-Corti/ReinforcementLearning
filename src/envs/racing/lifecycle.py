"""Episode lifecycle rules independent of the Gymnasium environment shell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs import CarConfig, RewardConfig, SimulationConfig

from ..geometry import segments_intersect
from ..observations import FrenetObserver, FrenetProjection, signed_progress
from ..tracks import TrackWithGeometry
from ..vehicle import KinematicTransition, VehicleState


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """
    Store lifecycle, reward, and termination results for one agent action.

    Fields:
        * reward: The reward selected by the crash, finish, or shaped branch.
        * terminated: Whether this action crashed or completed the lap.
        * truncated: Whether this action reached the episode time limit.
        * collision: Whether a physics substep left the track.
        * lap_completed: Whether a valid finish-gate crossing occurred.
        * progress_delta: The signed progress accumulated during this action.
        * wrapped_progress: The final projected position along the track.
        * episode_progress: The signed progress accumulated since reset.
        * collision_substep: The one-based substep where collision occurred.
        * termination_substep: The one-based substep where termination occurred.
    """

    reward: float
    terminated: bool
    truncated: bool
    collision: bool
    lap_completed: bool
    progress_delta: float
    wrapped_progress: float
    episode_progress: float
    collision_substep: int | None
    termination_substep: int | None


class EpisodeLifecycle:
    """
    Track progress, terminal outcomes, and reward for one fixed-start episode.

    Fields:
        * track: The sampled track and derived geometry used by the episode.
        * observer: The Frenet observer used for coherent centerline projections.
        * simulation: Simulation timing and time-limit configuration.
        * vehicle: Vehicle limits used to derive finish tolerance.
        * reward_config: Reward coefficients for the documented branches.
        * wrapped_progress: The previous projected position along the centerline.
        * episode_progress: The signed progress accumulated since reset.
        * agent_steps: The number of processed agent actions.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        simulation_config: SimulationConfig | None = None,
        vehicle_config: CarConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.track = track
        self.simulation = simulation_config or SimulationConfig()
        self.vehicle = vehicle_config or CarConfig()
        self.reward_config = reward_config or RewardConfig()
        self.observer = FrenetObserver(
            track,
            simulation_config=self.simulation,
            vehicle_config=self.vehicle,
        )
        self.wrapped_progress = 0.0
        self.episode_progress = 0.0
        self.agent_steps = 0
        self._previous_position = np.zeros(2, dtype=np.float64)
        self._previous_segment_index: int | None = None

    def reset(self, state: VehicleState) -> FrenetProjection:
        """
        Reset episode counters and return the state's centerline projection.
        """
        _, projection = self.observer.observe(state)
        self.wrapped_progress = projection.s
        self.episode_progress = 0.0
        self.agent_steps = 0
        self._previous_position = state.position()
        self._previous_segment_index = projection.segment_index
        return projection

    def process_transition(self, transition: KinematicTransition) -> ActionOutcome:
        """
        Process one action's physics states and return its lifecycle outcome.
        """
        if self._previous_segment_index is None:
            raise RuntimeError("reset must be called before processing a transition.")

        progress_delta = 0.0
        collision_substep: int | None = None
        termination_substep: int | None = None
        lap_completed = False
        for substep_index, state in enumerate(transition.substep_states, start=1):
            position = state.position()
            _, projection = self.observer.observe(
                state,
                previous_segment_index=self._previous_segment_index,
            )
            delta = signed_progress(
                self.wrapped_progress,
                projection.s,
                self.track.track.track_length,
            )
            progress_delta += delta
            self.episode_progress += delta
            crossing = self._crosses_finish_gate(self._previous_position, position)
            collision = abs(projection.lateral_distance) >= self.track.track.width / 2

            self.wrapped_progress = projection.s
            self._previous_position = position
            self._previous_segment_index = projection.segment_index
            if collision:
                collision_substep = substep_index
                termination_substep = substep_index
                break
            if crossing and self.episode_progress >= self._finish_progress_requirement:
                lap_completed = True
                termination_substep = substep_index
                break

        self.agent_steps += 1
        collision = collision_substep is not None
        terminated = collision or lap_completed
        truncated = (
            not terminated and self.agent_steps >= self.simulation.max_episode_steps
        )
        reward = self._reward(
            progress_delta,
            collision=collision,
            lap_completed=lap_completed,
        )
        return ActionOutcome(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            collision=collision,
            lap_completed=lap_completed,
            progress_delta=progress_delta,
            wrapped_progress=self.wrapped_progress,
            episode_progress=self.episode_progress,
            collision_substep=collision_substep,
            termination_substep=termination_substep,
        )

    @property
    def _finish_progress_requirement(self) -> float:
        """
        Return the progress required before a gate crossing can finish the lap.
        """
        finish_tolerance = max(
            2.0 * self.track.track.sample_spacing,
            self.vehicle.max_speed * self.simulation.agent_timestep,
        )
        return self.track.track.track_length - finish_tolerance

    def _crosses_finish_gate(
        self,
        previous_position: np.ndarray,
        current_position: np.ndarray,
    ) -> bool:
        """
        Return whether forward motion crosses the finite start/finish gate.
        """
        raw_track = self.track.track
        start_s = raw_track.start_index * raw_track.sample_spacing
        gate_center = self.track.position(start_s)
        normal = self.track.normal(start_s)
        tangent = np.asarray(
            [
                np.cos(self.track.heading(start_s)),
                np.sin(self.track.heading(start_s)),
            ]
        )
        gate_start = gate_center - raw_track.width / 2.0 * normal
        gate_end = gate_center + raw_track.width / 2.0 * normal
        previous_longitudinal = float(np.dot(previous_position - gate_center, tangent))
        current_longitudinal = float(np.dot(current_position - gate_center, tangent))
        return (
            previous_longitudinal < 0.0 <= current_longitudinal
            and segments_intersect(
                previous_position,
                current_position,
                gate_start,
                gate_end,
            )
        )

    def _reward(
        self,
        progress_delta: float,
        *,
        collision: bool,
        lap_completed: bool,
    ) -> float:
        """
        Select the documented crash, finish, or shaped reward branch.
        """
        if collision:
            return -self.reward_config.crash_penalty
        if lap_completed:
            return self.reward_config.finish_reward
        return (
            -self.reward_config.time_penalty_rate * self.simulation.agent_timestep
            + self.reward_config.progress_coefficient
            * progress_delta
            / self.track.track.track_length
        )
