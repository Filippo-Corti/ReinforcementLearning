"""Per-worker accumulation of in-flight episodes into finished episode records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from configs import ObservationRepresentation
from recording.records import (
    CircuitGeometrySummaryRecord,
    EpisodeOutcome,
    EpisodeRecord,
    MetricScope,
    RunCategory,
)
from utils.statistics import scalar_summary


@dataclass(slots=True)
class ActiveEpisode:
    """
    Accumulate semantic metrics for one worker's active training episode.

    An episode carries its own circuit's seed and geometry because a run that
    changes circuit at every reset cannot read either from a single prototype
    environment: eight workers race eight different circuits at once.

    Fields:
        * identity: Stable episode identity used by rollout records.
        * circuit_identity: Stable logical identity of the selected track.
        * circuit_seed: Generator seed of the circuit this episode races.
        * circuit_geometry: Frozen geometry summary of that circuit.
        * worker_index: Collection stream racing this episode.
        * worker_episode_index: Position of this episode within that stream.
        * step_index: Zero-based action position for the next transition.
        * undiscounted_return: Reward sum accumulated through the prior action.
        * maximum_progress: Greatest normalized progress observed so far.
        * speeds: Pre-action speed samples.
        * throttles: Applied throttle/brake samples.
        * absolute_steering: Absolute steering samples.
    """

    identity: int
    circuit_identity: str
    circuit_seed: int
    circuit_geometry: CircuitGeometrySummaryRecord
    worker_index: int
    worker_episode_index: int
    step_index: int = 0
    undiscounted_return: float = 0.0
    maximum_progress: float = 0.0
    speeds: list[float] = field(default_factory=list)
    throttles: list[float] = field(default_factory=list)
    absolute_steering: list[float] = field(default_factory=list)


class EpisodeCollector:
    """
    Track the episode every worker is racing, and close finished ones into records.

    This accumulates rather than logs. An episode is only summarizable once it
    ends, so each worker's speeds, actions and progress are held here across
    steps and turned into one `EpisodeRecord` at the boundary. Algorithms differ
    in when they collect and when they update, but not in what a finished
    episode means, so the manager stepping the workers owns exactly one of these
    and every engine sees the same episode records out of it.

    Speed is held here rather than read back out of the observation vector: each
    representation places it at a different index, and the info returned by a
    step describes the state *after* the action, while what is recorded is the
    speed the policy acted on.

    Fields:
        * run_category: Category carried by every emitted record.
        * observation_type: Representation the policy observed.
        * records: Episodes finished so far, in completion order.
    """

    def __init__(
        self,
        worker_count: int,
        *,
        run_category: RunCategory,
        observation_type: ObservationRepresentation,
        root_identity: int | None = None,
        circuit_split: str | None = None,
        near_saturated_steering_threshold: float | None = None,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("Episode recording requires at least one worker.")
        self.run_category = run_category
        self.observation_type = observation_type
        self.root_identity = root_identity
        self.circuit_split = circuit_split
        self.near_saturated_steering_threshold = near_saturated_steering_threshold
        self.records: list[EpisodeRecord] = []
        self._worker_count = worker_count
        self._active: list[ActiveEpisode | None] = [None] * worker_count
        self._worker_episode_counts = [0] * worker_count
        self._pending_speeds = np.zeros(worker_count, dtype=np.float64)
        self._next_identity = 0

    @property
    def next_episode_identity(self) -> int:
        """
        Return the identity the next opened episode will take.
        """
        return self._next_identity

    def active(self, worker_index: int) -> ActiveEpisode:
        """
        Return the episode a worker is currently racing.
        """
        episode = self._active[worker_index]
        if episode is None:
            raise RuntimeError(f"Worker {worker_index} has no active episode.")
        return episode

    def begin(self, worker_index: int, reset_info: dict[str, Any]) -> ActiveEpisode:
        """
        Open an episode on whichever circuit the worker just reset onto.

        The worker and its own episode count are carried through to the record
        because they are what pairs two runs: an episode identity is assigned in
        completion order and means something different in each run.
        """
        geometry = reset_info["circuit_geometry"]
        if not isinstance(geometry, CircuitGeometrySummaryRecord):
            raise TypeError("A worker reset must report its circuit geometry.")
        episode = ActiveEpisode(
            identity=self._next_identity,
            circuit_identity=str(reset_info["circuit_identity"]),
            circuit_seed=int(reset_info["track_seed"]),
            circuit_geometry=geometry,
            worker_index=worker_index,
            worker_episode_index=self._worker_episode_counts[worker_index],
        )
        self._next_identity += 1
        self._worker_episode_counts[worker_index] += 1
        self._active[worker_index] = episode
        self._pending_speeds[worker_index] = float(reset_info["speed"])
        return episode

    def add_step(
        self,
        worker_index: int,
        action: np.ndarray,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """
        Add one applied action to the episode a worker is racing.
        """
        episode = self.active(worker_index)
        episode.speeds.append(float(self._pending_speeds[worker_index]))
        episode.throttles.append(float(action[0]))
        episode.absolute_steering.append(abs(float(action[1])))
        episode.step_index += 1
        episode.undiscounted_return += reward
        progress = float(info["episode_progress"]) / float(info["track_length"])
        episode.maximum_progress = max(episode.maximum_progress, progress)
        self._pending_speeds[worker_index] = float(info["speed"])

    def finish(
        self,
        worker_index: int,
        *,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        training_interactions: int,
        evaluation_interactions: int,
        circuit_split: str | None = None,
    ) -> EpisodeRecord:
        """
        Close a worker's episode and append its record.
        """
        episode = self.active(worker_index)
        outcome = EpisodeOutcome.from_transition(
            terminated=terminated, truncated=truncated, info=info
        )
        record = EpisodeRecord(
            run_category=self.run_category,
            scope=MetricScope.TRAINING,
            episode_index=episode.identity,
            outcome=outcome,
            undiscounted_return=episode.undiscounted_return,
            training_target_total=None,
            interactions=episode.step_index,
            simulated_time=float(info["elapsed_time"]),
            final_progress=float(info["episode_progress"])
            / float(info["track_length"]),
            maximum_progress=episode.maximum_progress,
            lap_time=(
                float(info["elapsed_time"])
                if outcome is EpisodeOutcome.COMPLETED
                else None
            ),
            training_interactions=training_interactions,
            evaluation_interactions=evaluation_interactions,
            circuit_identity=episode.circuit_identity,
            root_identity=self.root_identity,
            observation_type=self.observation_type.value,
            circuit_seed=episode.circuit_seed,
            circuit_split=circuit_split or self.circuit_split,
            collection_worker=episode.worker_index,
            worker_episode_index=episode.worker_episode_index,
            speed=scalar_summary(episode.speeds),
            throttle=scalar_summary(episode.throttles),
            absolute_steering=scalar_summary(episode.absolute_steering),
            positive_throttle_fraction=float(
                np.mean(np.asarray(episode.throttles) > 0)
            ),
            braking_fraction=float(np.mean(np.asarray(episode.throttles) < 0)),
            near_saturated_steering_fraction=(
                None
                if self.near_saturated_steering_threshold is None
                else float(
                    np.mean(
                        np.asarray(episode.absolute_steering)
                        >= self.near_saturated_steering_threshold
                    )
                )
            ),
            circuit_geometry=episode.circuit_geometry,
        )
        self.records.append(record)
        self._active[worker_index] = None
        return record

    def state(self) -> dict[str, Any]:
        """
        Return everything needed to resume mid-episode accumulation exactly.
        """
        return {
            "active": list(self._active),
            "worker_episode_counts": list(self._worker_episode_counts),
            "pending_speeds": self._pending_speeds.tolist(),
            "next_identity": self._next_identity,
            "records": list(self.records),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """
        Resume from a checkpoint's accumulation, including partial episodes.
        """
        self._active = list(state["active"])
        self._worker_episode_counts = [
            int(value) for value in state["worker_episode_counts"]
        ]
        self._pending_speeds = np.asarray(state["pending_speeds"], dtype=np.float64)
        self._next_identity = int(state["next_identity"])
        records = state["records"]
        if not all(isinstance(record, EpisodeRecord) for record in records):
            raise TypeError("checkpoint episode records have invalid types.")
        self.records = list(records)
