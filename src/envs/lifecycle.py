"""Episode lifecycle rules independent of the Gymnasium environment shell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs import CarConfig, RewardConfig, SimulationConfig

from .dynamics import DynamicsTransition, VehicleState
from .geometry import TrackGeometry
from .observations import FrenetProjection, FrenetProjector, signed_progress


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    """
    Lifecycle outcome for one agent action.

    Fields:
        * reward: The reward selected by the crash, finish, or shaped branch.
        * terminated: Whether the transition reached a crash or completed lap.
        * truncated: Whether the time limit was reached without termination.
        * collision: Whether the car left the track during a physics substep.
        * lap_completed: Whether the car validly crossed the finish gate.
        * progress_delta: The signed progress accumulated during this action.
        * wrapped_progress: The final projected position along the track.
        * episode_progress: The signed progress accumulated since reset.
        * collision_substep: The one-based physics substep where collision occurred.
        * termination_substep: The one-based physics substep where the episode terminated.
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
    Track progress, terminal outcomes and reward for one fixed-start episode.
    The caller supplies dynamics transitions, leaving Gymnasium state handling and
    rendering outside this focused lifecycle component.

    Fields:
        * geometry: The track geometry used for projections, collision and finish checks.
        * projector: The Frenet projector that maintains temporally coherent projections.
        * simulation: Simulation timing and time-limit configuration.
        * vehicle: Vehicle limits used to derive finish tolerance.
        * reward_config: Reward coefficients for the documented branches.
        * wrapped_progress: The previous projected position along the centerline.
        * episode_progress: The signed progress accumulated since reset.
        * agent_steps: The number of completed agent actions.
    """

    def __init__(
        self,
        geometry: TrackGeometry,
        *,
        simulation_config: SimulationConfig | None = None,
        vehicle_config: CarConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.geometry = geometry
        self.simulation = simulation_config or SimulationConfig()
        self.vehicle = vehicle_config or CarConfig()
        self.reward_config = reward_config or RewardConfig()
        self.projector = FrenetProjector(
            geometry,
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
        Initialize lifecycle progress from a reset state.
        """
        projection = self.projector.project(_position(state))
        self.wrapped_progress = projection.s
        self.episode_progress = 0.0
        self.agent_steps = 0
        self._previous_position = _position(state)
        self._previous_segment_index = projection.segment_index
        return projection

    def advance(self, dynamics: DynamicsTransition) -> EpisodeTransition:
        """
        Evaluate each physics substep and return the agent-action outcome.
        """
        if self._previous_segment_index is None:
            raise RuntimeError("reset must be called before advancing an episode.")

        progress_delta = 0.0
        collision_substep: int | None = None
        termination_substep: int | None = None
        lap_completed = False
        for substep_index, state in enumerate(dynamics.substep_states, start=1):
            position = _position(state)
            projection = self.projector.project(
                position,
                previous_segment_index=self._previous_segment_index,
            )
            delta = signed_progress(
                self.wrapped_progress,
                projection.s,
                self.geometry.track.track_length,
            )
            progress_delta += delta
            self.episode_progress += delta
            crossing = self._crosses_finish_gate(self._previous_position, position)
            collision = (
                abs(projection.lateral_distance) >= self.geometry.track.width / 2
            )

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
        return EpisodeTransition(
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
        finish_tolerance = max(
            2.0 * self.geometry.track.sample_spacing,
            self.vehicle.max_speed * self.simulation.agent_timestep,
        )
        return self.geometry.track.track_length - finish_tolerance

    def _crosses_finish_gate(
        self,
        previous_position: np.ndarray,
        current_position: np.ndarray,
    ) -> bool:
        start_s = self.geometry.track.start_index * self.geometry.track.sample_spacing
        gate_center = self.geometry.position(start_s)
        normal = self.geometry.normal(start_s)
        tangent = np.asarray(
            [
                np.cos(self.geometry.heading(start_s)),
                np.sin(self.geometry.heading(start_s)),
            ]
        )
        gate_start = gate_center - self.geometry.track.width / 2.0 * normal
        gate_end = gate_center + self.geometry.track.width / 2.0 * normal
        previous_longitudinal = float(np.dot(previous_position - gate_center, tangent))
        current_longitudinal = float(np.dot(current_position - gate_center, tangent))
        return (
            previous_longitudinal < 0.0 <= current_longitudinal
            and _segments_intersect(
                previous_position, current_position, gate_start, gate_end
            )
        )

    def _reward(
        self,
        progress_delta: float,
        *,
        collision: bool,
        lap_completed: bool,
    ) -> float:
        if collision:
            return -self.reward_config.crash_penalty
        if lap_completed:
            return self.reward_config.finish_reward
        return (
            -self.reward_config.time_penalty_rate * self.simulation.agent_timestep
            + self.reward_config.progress_coefficient
            * progress_delta
            / self.geometry.track.track_length
        )


def _position(state: VehicleState) -> np.ndarray:
    """
    Return a state position as a float64 Cartesian vector.
    """
    return np.asarray([state.x, state.y], dtype=np.float64)


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    """
    Return whether two closed two-dimensional line segments intersect.
    """
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)
    return (
        first_orientation * second_orientation <= 0.0
        and third_orientation * fourth_orientation <= 0.0
    )


def _orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    """
    Return the signed two-dimensional orientation of a point around a segment.
    """
    displacement = end - start
    relative_point = point - start
    return float(
        displacement[0] * relative_point[1] - displacement[1] * relative_point[0]
    )
