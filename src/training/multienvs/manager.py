"""One synchronous vector step, turned into transitions and episode bookkeeping.

Normalizing an observation, asking the agent to act, stepping the workers and
assembling a transition row is the same work whichever algorithm is learning.
It lives here so that each engine's own file is about what that algorithm does
with the transitions rather than about how they are obtained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.types import CollectedActionBatch, OnPolicyAgent
from circuits import CircuitSplit, TrainingCircuitSchedule
from configs import EnvironmentConfig, ExecutionConfig, ObservationRepresentation
from envs.tracks import TrackWithGeometry
from normalization import RunningObservationNormalizer
from recording.records import EpisodeRecord, RunCategory

from ..buffers import TrainingTransition
from .episodes import EpisodeCollector
from .vector_env import (
    PersistentRacingVectorEnv,
    VectorRacingState,
    vector_worker_info,
)


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


class MultiEnvironmentManager:
    """
    Own the persistent environment pool and turn a step of it into transitions.

    Holds the observations the policy is about to act on, so an engine's loop
    can stay a loop: ask for a step, decide what to do with the transitions.

    Both the worker pool and the episode collector are built here rather than
    handed in. They exist only to serve a step of these environments, and an
    engine that held all three side by side would be holding one idea under
    three names. What an engine wants from this is the transitions, and the
    finished episodes those transitions completed.

    Fields:
        * agent: Chooses actions and, where it has a critic, values them.
        * environments: Persistent worker pool being stepped.
        * normalizer: Updated on training observations, frozen for evaluation.
        * episode_collector: Accumulates in-flight episodes across steps.
    """

    def __init__(
        self,
        agent: OnPolicyAgent,
        normalizer: RunningObservationNormalizer,
        *,
        track: TrackWithGeometry,
        environment_config: EnvironmentConfig,
        execution_config: ExecutionConfig,
        reset_generators: tuple[np.random.Generator, ...],
        selection_generators: tuple[np.random.Generator, ...],
        training_circuit_schedule: TrainingCircuitSchedule | None,
        run_category: RunCategory,
        observation_type: ObservationRepresentation,
        root_identity: int | None = None,
        circuit_split: str | None = None,
        near_saturated_steering_threshold: float | None = None,
    ) -> None:
        self.agent = agent
        self.normalizer = normalizer
        self.worker_count = execution_config.environment_workers
        self.environments = PersistentRacingVectorEnv(
            (track,),
            environment_config,
            execution_config,
            reset_generators,
            selection_generators,
            training_circuit_schedule,
        )
        self.episode_collector = EpisodeCollector(
            self.worker_count,
            run_category=run_category,
            observation_type=observation_type,
            root_identity=root_identity,
            # A scheduled run races the training split by construction. Without
            # a schedule the split is whatever the caller named the circuit.
            circuit_split=(
                CircuitSplit.TRAINING.value
                if training_circuit_schedule is not None
                else circuit_split
            ),
            near_saturated_steering_threshold=near_saturated_steering_threshold,
        )
        self.observations, reset_infos = self.environments.reset()
        for worker_index in range(self.worker_count):
            self.episode_collector.begin(
                worker_index, vector_worker_info(reset_infos, worker_index)
            )

    @property
    def episode_records(self) -> list[EpisodeRecord]:
        """
        Return every finished episode, in completion order.
        """
        return self.episode_collector.records

    @property
    def next_episode_identity(self) -> int:
        """
        Return the identity the next opened episode will take.
        """
        return self.episode_collector.next_episode_identity

    def step(
        self,
        active_mask: np.ndarray,
        *,
        training_interactions: int,
        evaluation_interactions: int,
    ) -> CollectedStep:
        """
        Step every enabled worker once and build one transition for each.

        The interaction counters are passed in because a finished episode is
        recorded against the run's totals, and this manager deliberately does
        not own them: what counts against a budget is the engine's business.
        """
        normalized = self.normalizer.normalize_batch(
            self.observations, active_mask, update=True
        )
        active_indices = np.flatnonzero(active_mask)
        decisions = self._collect_actions(normalized, active_indices)
        actions = np.zeros((self.worker_count, 2), dtype=np.float32)
        actions[active_indices] = decisions.env_actions
        next_observations, rewards, terminated, truncated, infos = (
            self.environments.step(actions, active_mask)
        )
        next_normalized = self.normalizer.normalize_batch(
            next_observations[active_indices], update=False
        )
        next_values = self.agent.bootstrap_values(next_normalized)

        transitions: list[TrainingTransition | None] = [None] * self.worker_count
        finished: list[int] = []
        interactions = 0
        for decision_index, worker_index_value in enumerate(active_indices):
            worker_index = int(worker_index_value)
            info = vector_worker_info(infos, worker_index)
            is_terminated = bool(terminated[worker_index])
            is_truncated = bool(truncated[worker_index])
            episode = self.episode_collector.active(worker_index)
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
            self.episode_collector.add_step(
                worker_index,
                decisions.env_actions[decision_index],
                float(rewards[worker_index]),
                info,
            )
            interactions += 1
            if is_terminated or is_truncated:
                self.episode_collector.finish(
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
            self.episode_collector.begin(
                worker_index, vector_worker_info(infos, worker_index)
            )

    def state(self) -> dict[str, Any]:
        """
        Return the observations, worker pool, and in-flight episodes.
        """
        return {
            "observations": self.observations.tolist(),
            "vector_environment": self.environments.state(),
            "episodes": self.episode_collector.state(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """
        Resume the observations, worker pool, and in-flight episodes.
        """
        self.observations = np.asarray(state["observations"], dtype=np.float32)
        vector_state = state["vector_environment"]
        if not isinstance(vector_state, VectorRacingState):
            raise TypeError("checkpoint vector environment state is invalid.")
        self.environments.restore(vector_state)
        episodes = state["episodes"]
        if not isinstance(episodes, dict):
            raise TypeError("checkpoint episode state must be a dictionary.")
        self.episode_collector.restore(episodes)

    def close(self) -> None:
        """
        Close the persistent worker processes.
        """
        self.environments.close()

    def _collect_actions(
        self, normalized: np.ndarray, active_indices: np.ndarray
    ) -> CollectedActionBatch:
        return self.agent.collect_actions(
            normalized[active_indices],
            environment_indices=[int(index) for index in active_indices],
        )
