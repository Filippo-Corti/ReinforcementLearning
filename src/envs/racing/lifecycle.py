"""Episode lifecycle rules independent of the Gymnasium environment shell."""

from __future__ import annotations

from collections import deque
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
        * stalled: Whether the car failed to make progress for the stall window.
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
    stalled: bool
    progress_delta: float
    wrapped_progress: float
    episode_progress: float
    collision_substep: int | None
    termination_substep: int | None


@dataclass(frozen=True, slots=True)
class FinishGate:
    """
    Store the fixed finish-line segment an episode must cross to complete a lap.

    The gate spans the track width at one arc length and never moves during an
    episode, so its geometry is resolved once instead of at every physics
    substep.

    Fields:
        * center: The centerline point at the gate arc length.
        * tangent: The unit forward direction at the gate.
        * start: The gate endpoint right of forward travel.
        * end: The gate endpoint left of forward travel.
    """

    center: np.ndarray
    tangent: np.ndarray
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True, slots=True)
class EpisodeLifecycleState:
    """
    Store the mutable episode progress needed to resume racing dynamics exactly.

    Fields:
        * wrapped_progress: Previous centerline position modulo track length.
        * episode_progress: Signed progress accumulated since reset.
        * agent_steps: Number of processed agent actions.
        * previous_position: Cartesian point used by finish-gate detection.
        * previous_segment_index: Projection hint used by the Frenet observer.
        * gate_s: Arc length of this episode's finish gate.
        * recent_progress: Progress of the actions inside the stall window.
    """

    wrapped_progress: float
    episode_progress: float
    agent_steps: int
    previous_position: tuple[float, float]
    previous_segment_index: int | None
    gate_s: float
    recent_progress: tuple[float, ...]


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
        self._gate_s = float(track.track.start_index * track.track.sample_spacing)
        self._gate = self._build_gate(self._gate_s)
        self._stall_steps = max(
            1, round(self.simulation.stall_time / self.simulation.agent_timestep)
        )
        self._recent_progress: deque[float] = deque(maxlen=self._stall_steps)

    def reset(self, state: VehicleState) -> FrenetProjection:
        """
        Reset episode counters and return the state's centerline projection.

        The finish gate is placed where the car starts, so a lap is always one
        full circuit from the sampled start pose rather than a partial run to a
        fixed line somewhere else on the track.
        """
        _, projection = self.observer.observe(state)
        self.wrapped_progress = projection.s
        self.episode_progress = 0.0
        self.agent_steps = 0
        self._previous_position = state.position()
        self._previous_segment_index = projection.segment_index
        self._gate_s = projection.s
        self._gate = self._build_gate(projection.s)
        self._recent_progress.clear()
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
        self._recent_progress.append(progress_delta)
        collision = collision_substep is not None
        stalled = not collision and not lap_completed and self._is_stalled()
        terminated = collision or lap_completed or stalled
        truncated = (
            not terminated and self.agent_steps >= self.simulation.max_episode_steps
        )
        reward = self._reward(
            progress_delta,
            collision=collision,
            lap_completed=lap_completed,
            stalled=stalled,
        )
        return ActionOutcome(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            collision=collision,
            lap_completed=lap_completed,
            stalled=stalled,
            progress_delta=progress_delta,
            wrapped_progress=self.wrapped_progress,
            episode_progress=self.episode_progress,
            collision_substep=collision_substep,
            termination_substep=termination_substep,
        )

    @property
    def gate_s(self) -> float:
        """
        Return the arc length this episode's lap ends at.

        The gate is placed where the car started, so a drawing that marks the
        canonical start line would be marking the wrong place on every episode
        that begins somewhere else.
        """
        return self._gate_s

    @property
    def current_segment_index(self) -> int | None:
        """
        Return the centerline segment holding the most recent projection.
        """
        return self._previous_segment_index

    def state(self) -> EpisodeLifecycleState:
        """
        Return the semantic mutable state needed for a mid-episode restore.
        """
        return EpisodeLifecycleState(
            wrapped_progress=self.wrapped_progress,
            episode_progress=self.episode_progress,
            agent_steps=self.agent_steps,
            previous_position=(
                float(self._previous_position[0]),
                float(self._previous_position[1]),
            ),
            previous_segment_index=self._previous_segment_index,
            gate_s=self._gate_s,
            recent_progress=tuple(self._recent_progress),
        )

    def restore(self, state: EpisodeLifecycleState) -> None:
        """
        Restore semantic episode progress after the owning environment restores pose.
        """
        self.wrapped_progress = state.wrapped_progress
        self.episode_progress = state.episode_progress
        self.agent_steps = state.agent_steps
        self._previous_position = np.asarray(state.previous_position, dtype=np.float64)
        self._previous_segment_index = state.previous_segment_index
        self._gate_s = state.gate_s
        self._gate = self._build_gate(state.gate_s)
        self._recent_progress.clear()
        self._recent_progress.extend(state.recent_progress)

    def _build_gate(self, gate_s: float) -> FinishGate:
        """
        Resolve the finish-line geometry at one arc length along the centerline.
        """
        half_width = self.track.track.width / 2.0
        center = self.track.position(gate_s)
        normal = self.track.normal(gate_s)
        heading = self.track.heading(gate_s)
        return FinishGate(
            center=center,
            tangent=np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64),
            start=center - half_width * normal,
            end=center + half_width * normal,
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
        gate = self._gate
        previous_longitudinal = float(
            np.dot(previous_position - gate.center, gate.tangent)
        )
        current_longitudinal = float(
            np.dot(current_position - gate.center, gate.tangent)
        )
        return (
            previous_longitudinal < 0.0 <= current_longitudinal
            and segments_intersect(
                previous_position,
                current_position,
                gate.start,
                gate.end,
            )
        )

    def _is_stalled(self) -> bool:
        """
        Return whether the car has failed to advance for the whole stall window.
        """
        return (
            len(self._recent_progress) == self._stall_steps
            and sum(self._recent_progress) < self.simulation.stall_progress
        )

    def _reward(
        self,
        progress_delta: float,
        *,
        collision: bool,
        lap_completed: bool,
        stalled: bool,
    ) -> float:
        """
        Select the documented crash, finish, stall, or shaped reward branch.

        A stall is charged the time penalty it would have paid by idling out the
        remaining episode. Ending early therefore saves the simulation but not
        the agent: standing still costs exactly what standing still has always
        cost, so the ordering that keeps driving preferable is unaffected.
        """
        step_cost = (
            self.reward_config.time_penalty_rate * self.simulation.agent_timestep
        )
        if collision:
            return -self.reward_config.crash_penalty
        if lap_completed:
            return self._finish_reward()
        if stalled:
            return -step_cost * (
                self.simulation.max_episode_steps - self.agent_steps + 1
            )
        return (
            -step_cost
            + self.reward_config.progress_coefficient
            * progress_delta
            / self.track.track.track_length
        )

    def _finish_reward(self) -> float:
        """
        Return the completion reward, increased for a faster lap.

        Progress and the per-step time penalty alone leave a slow lap worth
        almost as much as a fast one. Scaling part of the completion reward by
        the unused share of the episode clock makes lap time, the actual racing
        objective, a visible fraction of the return without ever making a crash
        preferable to finishing.
        """
        unused_clock = 1.0 - self.agent_steps / self.simulation.max_episode_steps
        return (
            self.reward_config.finish_reward
            + self.reward_config.lap_time_bonus * unused_clock
        )
