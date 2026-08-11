"""Readable parallel fixed-rollout training loop for A2C with GAE."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from agents import A2CAgent, AgentUpdateInput, CollectionMode
from configs import EnvironmentConfig, ExecutionConfig
from envs.tracks import TrackWithGeometry

from ..buffers import TrainingTransition, VectorRolloutBuffer
from ..normalization import RunningObservationNormalizer
from ..vector_environment import (
    PersistentRacingVectorEnv,
    vector_info,
    vector_worker_info,
)
from .records import (
    EducationalEpisodeRecord,
    EducationalTrainingHistory,
    EducationalUpdateRecord,
    racing_outcome,
)


class A2CTrainingEngine:
    """
    Train A2C from synchronous persistent environment workers.

    In accordance with `docs/LEARNING.md`, the pooled rollout contains 2048
    valid transitions and may cross episode boundaries. Its explicit
    `(time, environments, ...)` shape lets the agent compute detached GAE down
    each worker column without allowing recursion to cross environments. A
    final shorter pooled rollout is also used.

    Fields:
        * agent: A2C actor, critic, optimizers, and per-worker sampling streams.
        * tracks: Circuits available for selection at episode boundaries.
        * environment_config: Racing dynamics, reward, observation, and time limit.
        * execution_config: Persistent environment-worker execution settings.
        * normalizer: Running observation statistics updated only during training.
        * environments: Persistent process-based racing environment pool.
        * history: Episode and optimizer-update records collected so far.
    """

    def __init__(
        self,
        agent: A2CAgent,
        tracks: Sequence[TrackWithGeometry],
        environment_config: EnvironmentConfig,
        normalizer: RunningObservationNormalizer,
        environment_reset_generator: (
            np.random.Generator | Sequence[np.random.Generator]
        ),
        track_selection_generator: np.random.Generator | Sequence[np.random.Generator],
        *,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """
        Construct an educational A2C engine over one or more circuits.
        """
        if not tracks:
            raise ValueError("A2C training requires at least one circuit.")
        worker_count = len(agent.sampling_generators)
        self.execution_config = execution_config or ExecutionConfig(
            device="cpu", environment_workers=worker_count
        )
        if worker_count != self.execution_config.environment_workers:
            raise ValueError("A2C requires one policy stream per environment worker.")
        self.agent = agent
        self.tracks = tuple(tracks)
        self.environment_config = environment_config
        self.normalizer = normalizer
        self.history = EducationalTrainingHistory()
        self._rollout_buffer = VectorRolloutBuffer(agent.collection_size, worker_count)
        self.environments = PersistentRacingVectorEnv(
            self.tracks,
            environment_config,
            self.execution_config,
            _generator_tuple(environment_reset_generator, worker_count, "reset"),
            _generator_tuple(
                track_selection_generator, worker_count, "track-selection"
            ),
        )

    def train(
        self,
        episode_count: int,
        *,
        on_episode_end: (
            Callable[[EducationalEpisodeRecord, EducationalTrainingHistory], None]
            | None
        ) = None,
    ) -> EducationalTrainingHistory:
        """
        Collect exactly the requested episodes across synchronous CPU workers.
        """
        if episode_count <= 0:
            raise ValueError("Episode count must be positive.")

        worker_count = self.execution_config.environment_workers
        active = np.zeros(worker_count, dtype=np.bool_)
        initial_episodes = min(worker_count, episode_count)
        active[:initial_episodes] = True
        observations, reset_infos = self.environments.reset(active)
        next_episode_identity = len(self.history.episodes)
        episode_identities = np.full(worker_count, -1, dtype=np.int64)
        for environment_index in range(initial_episodes):
            episode_identities[environment_index] = next_episode_identity
            next_episode_identity += 1
        started_episodes = initial_episodes
        completed_episodes = 0
        episode_steps = np.zeros(worker_count, dtype=np.int64)
        episode_returns = np.zeros(worker_count, dtype=np.float64)
        maximum_progress = np.zeros(worker_count, dtype=np.float64)
        circuit_identities = ["" for _ in range(worker_count)]
        for environment_index in range(initial_episodes):
            circuit_identities[environment_index] = str(
                vector_info(reset_infos, "circuit_identity", environment_index)
            )

        progress_bar = tqdm(
            total=episode_count,
            desc="Training episodes",
            unit="episode",
        )
        try:
            while completed_episodes < episode_count:
                collection_active = active.copy()
                active_indices = np.flatnonzero(collection_active)
                remaining_capacity = self._rollout_buffer.remaining_capacity
                if active_indices.size > remaining_capacity:
                    collection_active[active_indices[remaining_capacity:]] = False
                    active_indices = active_indices[:remaining_capacity]

                normalized = self.normalizer.update_and_normalize_batch(
                    observations, collection_active
                )
                decisions = self.agent.collect_actions(
                    normalized[active_indices],
                    environment_indices=[int(index) for index in active_indices],
                )
                actions = np.zeros((worker_count, 2), dtype=np.float32)
                actions[active_indices] = decisions.env_actions
                next_observations, rewards, terminated, truncated, infos = (
                    self.environments.step(actions, collection_active)
                )
                next_normalized = np.stack(
                    [
                        self.normalizer.normalize(next_observations[index])
                        for index in active_indices
                    ]
                )
                next_values = self.agent.bootstrap_values(next_normalized)
                if decisions.current_values is None:
                    raise RuntimeError("A2C collection requires current critic values.")

                transition_step: list[TrainingTransition | None] = [None] * worker_count
                ended_records: list[tuple[int, EducationalEpisodeRecord]] = []
                for decision_index, environment_index_value in enumerate(
                    active_indices
                ):
                    environment_index = int(environment_index_value)
                    info = vector_worker_info(infos, environment_index)
                    is_terminated = bool(terminated[environment_index])
                    is_truncated = bool(truncated[environment_index])
                    transition_step[environment_index] = TrainingTransition(
                        normalized_observation=normalized[environment_index],
                        raw_action=decisions.raw_actions[decision_index],
                        env_action=decisions.env_actions[decision_index],
                        reward=float(rewards[environment_index]),
                        behaviour_log_probability=float(
                            decisions.behaviour_log_probabilities[decision_index]
                        ),
                        current_value=float(decisions.current_values[decision_index]),
                        next_value=(
                            0.0 if is_terminated else float(next_values[decision_index])
                        ),
                        terminated=is_terminated,
                        truncated=is_truncated,
                        next_normalized_observation=next_normalized[decision_index],
                        episode_identity=int(episode_identities[environment_index]),
                        episode_step_index=int(episode_steps[environment_index]),
                        circuit_identity=circuit_identities[environment_index],
                        environment_index=environment_index,
                    )
                    episode_steps[environment_index] += 1
                    episode_returns[environment_index] += rewards[environment_index]
                    progress = float(info["episode_progress"]) / float(
                        info["track_length"]
                    )
                    maximum_progress[environment_index] = max(
                        maximum_progress[environment_index], progress
                    )
                    if is_terminated or is_truncated:
                        ended_records.append(
                            (
                                environment_index,
                                EducationalEpisodeRecord(
                                    episode_index=int(
                                        episode_identities[environment_index]
                                    ),
                                    circuit_identity=circuit_identities[
                                        environment_index
                                    ],
                                    interactions=int(episode_steps[environment_index]),
                                    undiscounted_return=float(
                                        episode_returns[environment_index]
                                    ),
                                    outcome=racing_outcome(
                                        is_terminated, is_truncated, info
                                    ),
                                    final_progress=progress,
                                    maximum_progress=float(
                                        maximum_progress[environment_index]
                                    ),
                                ),
                            )
                        )

                self._rollout_buffer.append_step(transition_step)
                self.history.training_interactions += int(active_indices.size)
                observations = next_observations
                if self._rollout_buffer.transition_count == self.agent.collection_size:
                    self._apply_rollout_update()

                reset_mask = np.zeros(worker_count, dtype=np.bool_)
                for environment_index, episode_record in ended_records:
                    active[environment_index] = False
                    self.history.episodes.append(episode_record)
                    completed_episodes += 1
                    progress_bar.update(1)
                    if started_episodes < episode_count:
                        episode_identities[environment_index] = next_episode_identity
                        next_episode_identity += 1
                        started_episodes += 1
                        episode_steps[environment_index] = 0
                        episode_returns[environment_index] = 0.0
                        maximum_progress[environment_index] = 0.0
                        active[environment_index] = True
                        reset_mask[environment_index] = True
                    if on_episode_end is not None:
                        on_episode_end(episode_record, self.history)

                if np.any(reset_mask):
                    reset_observations, reset_infos = self.environments.reset(
                        reset_mask
                    )
                    observations = reset_observations
                    for environment_index_value in np.flatnonzero(reset_mask):
                        environment_index = int(environment_index_value)
                        circuit_identities[environment_index] = str(
                            vector_info(
                                reset_infos,
                                "circuit_identity",
                                environment_index,
                            )
                        )
        finally:
            progress_bar.close()

        if self._rollout_buffer.transition_count:
            self._apply_rollout_update()
        return self.history

    def close(self) -> None:
        """
        Close the persistent environment processes owned by this engine.
        """
        self.environments.close()

    def _apply_rollout_update(self) -> None:
        rollout = self._rollout_buffer.finalize()
        output = self.agent.update(
            AgentUpdateInput(mode=CollectionMode.FIXED_ROLLOUT, rollout=rollout)
        )
        self.history.updates.append(
            EducationalUpdateRecord(
                update_index=len(self.history.updates),
                final_episode_index=max(
                    transition.episode_identity for transition in rollout.transitions
                ),
                training_interactions=self.history.training_interactions,
                transition_count=len(rollout.transitions),
                diagnostics=output.diagnostics,
            )
        )


def _generator_tuple(
    generators: np.random.Generator | Sequence[np.random.Generator],
    worker_count: int,
    role: str,
) -> tuple[np.random.Generator, ...]:
    if isinstance(generators, np.random.Generator):
        values = (generators,)
    else:
        values = tuple(generators)
    if len(values) != worker_count:
        raise ValueError(f"One {role} generator is required per environment worker.")
    return values
