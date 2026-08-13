"""One synchronous vector step, turned into transitions and episode bookkeeping.

Normalizing an observation, asking the agent to act, stepping the workers and
assembling a transition row is the same work whichever algorithm is learning.
It lives here so that each engine's own file is about what that algorithm does
with the transitions rather than about how they are obtained.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from agents.types import CollectedActionBatch, OnPolicyAgent

from ..buffers import TrainingTransition
from ..normalization import RunningObservationNormalizer
from ..vector_environment import PersistentRacingVectorEnv, vector_worker_info
from .episode_recording import EpisodeRecorder


@dataclass(frozen=True, slots=True)
class CollectedStep:
    """
    Report what one vector step produced, worker by worker.

    Fields:
        * transitions: One row per worker, or none where the worker was parked.
        * finished: Indices of workers whose episode ended on this step.
        * interactions: Transitions that count against the training budget.
    """

    transitions: list[TrainingTransition | None]
    finished: tuple[int, ...]
    interactions: int


class StepCollector:
    """
    Advance the environment workers and record what the policy did.

    Holds the observations the policy is about to act on, so an engine's loop
    can stay a loop: ask for a step, decide what to do with the transitions.

    Fields:
        * agent: Chooses actions and, where it has a critic, values them.
        * environments: Persistent worker pool being stepped.
        * normalizer: Updated on training observations, frozen for evaluation.
        * recorder: Accumulates per-worker episode metrics.
    """

    def __init__(
        self,
        agent: OnPolicyAgent,
        environments: PersistentRacingVectorEnv,
        normalizer: RunningObservationNormalizer,
        recorder: EpisodeRecorder,
        *,
        worker_count: int,
    ) -> None:
        self.agent = agent
        self.environments = environments
        self.normalizer = normalizer
        self.recorder = recorder
        self.worker_count = worker_count
        self.observations, reset_infos = environments.reset()
        for worker_index in range(worker_count):
            recorder.begin(worker_index, vector_worker_info(reset_infos, worker_index))

    def step(
        self,
        active: np.ndarray,
        *,
        training_interactions: int,
        evaluation_interactions: int,
    ) -> CollectedStep:
        """
        Step every enabled worker once and build one transition for each.

        The interaction counters are passed in because a finished episode is
        recorded against the run's totals, and the collector deliberately does
        not own them: what counts against a budget is the engine's business.
        """
        normalized = self.normalizer.update_and_normalize_batch(
            self.observations, active
        )
        active_indices = np.flatnonzero(active)
        decisions = self._collect_actions(normalized, active_indices)
        actions = np.zeros((self.worker_count, 2), dtype=np.float32)
        actions[active_indices] = decisions.env_actions
        next_observations, rewards, terminated, truncated, infos = (
            self.environments.step(actions, active)
        )
        next_normalized = np.stack(
            [
                self.normalizer.normalize(next_observations[index])
                for index in active_indices
            ]
        )
        next_values = self._bootstrap_values(next_normalized)

        transitions: list[TrainingTransition | None] = [None] * self.worker_count
        finished: list[int] = []
        interactions = 0
        for decision_index, worker_index_value in enumerate(active_indices):
            worker_index = int(worker_index_value)
            info = vector_worker_info(infos, worker_index)
            is_terminated = bool(terminated[worker_index])
            is_truncated = bool(truncated[worker_index])
            episode = self.recorder.active(worker_index)
            transitions[worker_index] = TrainingTransition(
                normalized_observation=normalized[worker_index],
                raw_action=decisions.raw_actions[decision_index],
                env_action=decisions.env_actions[decision_index],
                reward=float(rewards[worker_index]),
                behaviour_log_probability=float(
                    decisions.behaviour_log_probabilities[decision_index]
                ),
                current_value=(
                    None
                    if decisions.current_values is None
                    else float(decisions.current_values[decision_index])
                ),
                next_value=(
                    None
                    if next_values is None
                    else (0.0 if is_terminated else float(next_values[decision_index]))
                ),
                terminated=is_terminated,
                truncated=is_truncated,
                next_normalized_observation=next_normalized[decision_index],
                episode_identity=episode.identity,
                episode_step_index=episode.step_index,
                circuit_identity=episode.circuit_identity,
                environment_index=worker_index,
            )
            self.recorder.record_step(
                worker_index,
                decisions.env_actions[decision_index],
                float(rewards[worker_index]),
                info,
            )
            interactions += 1
            if is_terminated or is_truncated:
                self.recorder.finish(
                    worker_index,
                    terminated=is_terminated,
                    truncated=is_truncated,
                    info=info,
                    training_interactions=training_interactions + interactions,
                    evaluation_interactions=evaluation_interactions,
                )
                finished.append(worker_index)

        self.observations = next_observations
        return CollectedStep(
            transitions=transitions,
            finished=tuple(finished),
            interactions=interactions,
        )

    def reset_workers(self, reset_mask: np.ndarray) -> None:
        """
        Reset the selected workers and open a fresh episode on each.
        """
        observations, infos = self.environments.reset(reset_mask)
        self.observations = observations
        for worker_index_value in np.flatnonzero(reset_mask):
            worker_index = int(worker_index_value)
            self.recorder.begin(worker_index, vector_worker_info(infos, worker_index))

    def state(self) -> dict[str, Any]:
        """
        Return the observations the policy is about to act on.
        """
        return {"observations": self.observations.tolist()}

    def restore(self, state: dict[str, Any]) -> None:
        """
        Resume from the observations a checkpoint was taken at.
        """
        self.observations = np.asarray(state["observations"], dtype=np.float32)

    def _collect_actions(
        self, normalized: np.ndarray, active_indices: np.ndarray
    ) -> CollectedActionBatch:
        batch_method = getattr(self.agent, "collect_actions", None)
        if callable(batch_method):
            return cast(Callable[..., CollectedActionBatch], batch_method)(
                normalized[active_indices],
                environment_indices=[int(index) for index in active_indices],
            )
        rows = [
            self.agent.collect_action(normalized[index]) for index in active_indices
        ]
        return CollectedActionBatch(
            raw_actions=np.stack([row.raw_action for row in rows]),
            env_actions=np.stack([row.env_action for row in rows]),
            behaviour_log_probabilities=np.asarray(
                [row.behaviour_log_probability or 0.0 for row in rows],
                dtype=np.float32,
            ),
            current_values=(
                None
                if any(row.current_value is None for row in rows)
                else np.asarray([row.current_value for row in rows], dtype=np.float32)
            ),
        )

    def _bootstrap_values(self, next_normalized: np.ndarray) -> np.ndarray | None:
        batch_method = getattr(self.agent, "bootstrap_values", None)
        if callable(batch_method):
            return cast(Callable[[np.ndarray], np.ndarray | None], batch_method)(
                next_normalized
            )
        values = [self.agent.bootstrap_value(row) for row in next_normalized]
        if any(value is None for value in values):
            return None
        return np.asarray(values, dtype=np.float32)
