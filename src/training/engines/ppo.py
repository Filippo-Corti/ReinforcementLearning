"""Readable parallel fixed-rollout training loop for clipped PPO."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from agents import AgentUpdateInput, CollectionMode, PPOAgent
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


class PPOTrainingEngine:
    """
    Train clipped PPO from synchronous persistent environment workers.

    Collection stores a fixed behaviour log probability in an explicit
    `(time, environments, ...)` rollout. GAE is computed down each worker column,
    after which the 2048 valid transitions are flattened for the unchanged
    seeded PPO minibatch epochs described in `docs/LEARNING.md`. A final shorter
    pooled rollout is also used.

    Fields:
        * agent: PPO actor, critic, optimizers, and per-worker sampling streams.
        * tracks: Circuits available for selection at episode boundaries.
        * environment_config: Racing dynamics, reward, observation, and time limit.
        * execution_config: Persistent environment-worker execution settings.
        * normalizer: Running observation statistics updated only during training.
        * environments: Persistent process-based racing environment pool.
        * history: Episode and optimizer-update records collected so far.
    """

    def __init__(
        self,
        agent: PPOAgent,
        tracks: Sequence[TrackWithGeometry],
        environment_config: EnvironmentConfig,
        normalizer: RunningObservationNormalizer,
        environment_reset_generator: (
            np.random.Generator | Sequence[np.random.Generator]
        ),
        track_selection_generator: np.random.Generator | Sequence[np.random.Generator],
        *,
        track_seed_for_episode: Callable[[int], int] | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """
        Construct an educational PPO engine over one or more circuits.
        """
        if not tracks:
            raise ValueError("PPO training requires at least one circuit.")
        worker_count = len(agent.sampling_generators)
        self.execution_config = execution_config or ExecutionConfig(
            device="cpu", environment_workers=worker_count
        )
        if worker_count != self.execution_config.environment_workers:
            raise ValueError("PPO requires one policy stream per environment worker.")
        self.agent = agent
        self.tracks = tuple(tracks)
        self.environment_config = environment_config
        self.normalizer = normalizer
        self.track_seed_for_episode = track_seed_for_episode
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
        interaction_budget: int,
        *,
        on_episode_end: (
            Callable[[EducationalEpisodeRecord, EducationalTrainingHistory], None]
            | None
        ) = None,
    ) -> EducationalTrainingHistory:
        """
        Collect and optimize until the exact interaction budget is exhausted.
        """
        if interaction_budget <= 0:
            raise ValueError("Interaction budget must be positive.")
        if interaction_budget < self.history.training_interactions:
            raise ValueError(
                "Interaction budget cannot be below consumed interactions."
            )
        remaining_budget = interaction_budget - self.history.training_interactions
        if remaining_budget == 0:
            return self.history

        worker_count = self.execution_config.environment_workers
        active = np.zeros(worker_count, dtype=np.bool_)
        initial_episodes = min(worker_count, remaining_budget)
        active[:initial_episodes] = True
        next_episode_identity = len(self.history.episodes)
        episode_identities = np.full(worker_count, -1, dtype=np.int64)
        for environment_index in range(initial_episodes):
            episode_identities[environment_index] = next_episode_identity
            next_episode_identity += 1
        observations, reset_infos = self.environments.reset(
            active,
            track_seeds=self._track_seeds(episode_identities, active),
        )
        episode_steps = np.zeros(worker_count, dtype=np.int64)
        episode_returns = np.zeros(worker_count, dtype=np.float64)
        maximum_progress = np.zeros(worker_count, dtype=np.float64)
        speed_totals = np.zeros(worker_count, dtype=np.float64)
        throttle_totals = np.zeros(worker_count, dtype=np.float64)
        signed_throttle_totals = np.zeros(worker_count, dtype=np.float64)
        circuit_identities = ["" for _ in range(worker_count)]
        for environment_index in range(initial_episodes):
            circuit_identities[environment_index] = str(
                vector_info(reset_infos, "circuit_identity", environment_index)
            )

        progress_bar = tqdm(
            total=remaining_budget,
            desc="Training interactions",
            unit="interaction",
        )
        try:
            while self.history.training_interactions < interaction_budget:
                collection_active = active.copy()
                active_indices = np.flatnonzero(collection_active)
                maximum_rows = min(
                    interaction_budget - self.history.training_interactions,
                    self._rollout_buffer.remaining_capacity,
                )
                if active_indices.size > maximum_rows:
                    collection_active[active_indices[maximum_rows:]] = False
                    active_indices = active_indices[:maximum_rows]

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
                    raise RuntimeError("PPO collection requires current critic values.")

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
                    speed_totals[environment_index] += observations[
                        environment_index, 2
                    ]
                    throttle_totals[environment_index] += abs(
                        float(decisions.env_actions[decision_index, 0])
                    )
                    signed_throttle_totals[environment_index] += float(
                        decisions.env_actions[decision_index, 0]
                    )
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
                                    mean_speed=float(
                                        speed_totals[environment_index]
                                        / episode_steps[environment_index]
                                    ),
                                    mean_throttle=float(
                                        signed_throttle_totals[environment_index]
                                        / episode_steps[environment_index]
                                    ),
                                    mean_throttle_magnitude=float(
                                        throttle_totals[environment_index]
                                        / episode_steps[environment_index]
                                    ),
                                    lap_time=(
                                        float(info["elapsed_time"])
                                        if bool(info["lap_completed"])
                                        else None
                                    ),
                                ),
                            )
                        )

                self._rollout_buffer.append_step(transition_step)
                self.history.training_interactions += int(active_indices.size)
                progress_bar.update(int(active_indices.size))
                observations = next_observations
                if self._rollout_buffer.transition_count == self.agent.collection_size:
                    self._apply_rollout_update()

                reset_mask = np.zeros(worker_count, dtype=np.bool_)
                for environment_index, episode_record in ended_records:
                    active[environment_index] = False
                    self.history.episodes.append(episode_record)
                    remaining_interactions = (
                        interaction_budget - self.history.training_interactions
                    )
                    active_episode_count = int(np.count_nonzero(active))
                    if remaining_interactions > active_episode_count:
                        episode_identities[environment_index] = next_episode_identity
                        next_episode_identity += 1
                        episode_steps[environment_index] = 0
                        episode_returns[environment_index] = 0.0
                        maximum_progress[environment_index] = 0.0
                        speed_totals[environment_index] = 0.0
                        throttle_totals[environment_index] = 0.0
                        signed_throttle_totals[environment_index] = 0.0
                        active[environment_index] = True
                        reset_mask[environment_index] = True
                    if on_episode_end is not None:
                        on_episode_end(episode_record, self.history)

                if np.any(reset_mask):
                    reset_observations, reset_infos = self.environments.reset(
                        reset_mask,
                        track_seeds=self._track_seeds(episode_identities, reset_mask),
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

    def _track_seeds(
        self,
        episode_identities: np.ndarray,
        reset_mask: np.ndarray,
    ) -> list[int | None] | None:
        """
        Derive one procedural circuit seed for each episode selected to reset.
        """
        if self.track_seed_for_episode is None:
            return None
        return [
            (
                self.track_seed_for_episode(int(episode_identities[index]))
                if reset_mask[index]
                else None
            )
            for index in range(self.execution_config.environment_workers)
        ]

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
