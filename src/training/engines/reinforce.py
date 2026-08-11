"""Readable parallel complete-episode training loop for REINFORCE."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from agents import AgentUpdateInput, CollectionMode, ReinforceAgent
from configs import EnvironmentConfig, ExecutionConfig
from envs.tracks import TrackWithGeometry

from ..buffers import OnPolicyRollout, TrainingTransition
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


class ReinforceTrainingEngine:
    """
    Train REINFORCE from one concurrent complete episode per environment.

    Each trajectory remains separate because its return-to-go uses only its own
    rewards. The eight documented batch trajectories are collected concurrently
    in persistent CPU workers, then their trajectory-sum losses are averaged by
    the unchanged agent update. A partial final batch remains recorded but unused.

    Fields:
        * agent: REINFORCE policy, optimizer, and per-worker sampling streams.
        * tracks: Circuits available for selection at episode boundaries.
        * environment_config: Racing dynamics, reward, observation, and time limit.
        * execution_config: Persistent environment-worker execution settings.
        * normalizer: Running observation statistics updated only during training.
        * environments: Persistent process-based racing environment pool.
        * history: Episode and optimizer-update records collected so far.
    """

    def __init__(
        self,
        agent: ReinforceAgent,
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
        Construct an educational REINFORCE engine over one or more circuits.
        """
        if not tracks:
            raise ValueError("REINFORCE training requires at least one circuit.")
        worker_count = len(agent.sampling_generators)
        self.execution_config = execution_config or ExecutionConfig(
            device="cpu", environment_workers=worker_count
        )
        if self.execution_config.environment_workers != agent.collection_size:
            raise ValueError(
                "REINFORCE requires one environment per complete trajectory."
            )
        if worker_count != self.execution_config.environment_workers:
            raise ValueError(
                "REINFORCE requires one policy-sampling stream per environment."
            )
        reset_generators = _generator_tuple(
            environment_reset_generator, worker_count, "reset"
        )
        track_generators = _generator_tuple(
            track_selection_generator, worker_count, "track-selection"
        )
        self.agent = agent
        self.tracks = tuple(tracks)
        self.environment_config = environment_config
        self.normalizer = normalizer
        self.history = EducationalTrainingHistory()
        self._episode_batch: list[OnPolicyRollout] = []
        self.environments = PersistentRacingVectorEnv(
            self.tracks,
            environment_config,
            self.execution_config,
            reset_generators,
            track_generators,
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
        Collect concurrent trajectories and report them after each possible update.
        """
        if episode_count <= 0:
            raise ValueError("Episode count must be positive.")

        next_episode_index = len(self.history.episodes)
        remaining_episodes = episode_count
        progress_bar = tqdm(
            total=episode_count,
            desc="Training episodes",
            unit="episode",
        )
        try:
            while remaining_episodes:
                needed_for_update = self.agent.collection_size - len(
                    self._episode_batch
                )
                wave_size = min(remaining_episodes, needed_for_update)
                active = np.zeros(
                    self.execution_config.environment_workers, dtype=np.bool_
                )
                active[:wave_size] = True
                observations, reset_infos = self.environments.reset(active)
                episode_indices = np.arange(
                    next_episode_index,
                    next_episode_index + wave_size,
                    dtype=np.int64,
                )
                transitions: list[list[TrainingTransition]] = [
                    [] for _ in range(wave_size)
                ]
                returns = np.zeros(wave_size, dtype=np.float64)
                maximum_progress = np.zeros(wave_size, dtype=np.float64)
                speed_totals = np.zeros(wave_size, dtype=np.float64)
                throttle_totals = np.zeros(wave_size, dtype=np.float64)
                circuit_identities = [
                    str(vector_info(reset_infos, "circuit_identity", index))
                    for index in range(wave_size)
                ]
                completed_records: list[EducationalEpisodeRecord] = []

                while np.any(active):
                    normalized = self.normalizer.update_and_normalize_batch(
                        observations, active
                    )
                    active_indices = np.flatnonzero(active)
                    decisions = self.agent.collect_actions(
                        normalized[active_indices],
                        environment_indices=[int(index) for index in active_indices],
                    )
                    actions = np.zeros(
                        (self.execution_config.environment_workers, 2),
                        dtype=np.float32,
                    )
                    actions[active_indices] = decisions.env_actions
                    next_observations, rewards, terminated, truncated, infos = (
                        self.environments.step(actions, active)
                    )

                    for decision_index, environment_index_value in enumerate(
                        active_indices
                    ):
                        environment_index = int(environment_index_value)
                        info = vector_worker_info(infos, environment_index)
                        next_normalized = self.normalizer.normalize(
                            next_observations[environment_index]
                        )
                        episode_rows = transitions[environment_index]
                        episode_rows.append(
                            TrainingTransition(
                                normalized_observation=normalized[environment_index],
                                raw_action=decisions.raw_actions[decision_index],
                                env_action=decisions.env_actions[decision_index],
                                reward=float(rewards[environment_index]),
                                behaviour_log_probability=float(
                                    decisions.behaviour_log_probabilities[
                                        decision_index
                                    ]
                                ),
                                current_value=None,
                                next_value=None,
                                terminated=bool(terminated[environment_index]),
                                truncated=bool(truncated[environment_index]),
                                next_normalized_observation=next_normalized,
                                episode_identity=int(
                                    episode_indices[environment_index]
                                ),
                                episode_step_index=len(episode_rows),
                                circuit_identity=circuit_identities[environment_index],
                                environment_index=environment_index,
                            )
                        )
                        self.history.training_interactions += 1
                        returns[environment_index] += rewards[environment_index]
                        speed_totals[environment_index] += observations[
                            environment_index, 2
                        ]
                        throttle_totals[environment_index] += abs(
                            float(decisions.env_actions[decision_index, 0])
                        )
                        progress = float(info["episode_progress"]) / float(
                            info["track_length"]
                        )
                        maximum_progress[environment_index] = max(
                            maximum_progress[environment_index], progress
                        )
                        if (
                            terminated[environment_index]
                            or truncated[environment_index]
                        ):
                            active[environment_index] = False
                            completed_records.append(
                                EducationalEpisodeRecord(
                                    episode_index=int(
                                        episode_indices[environment_index]
                                    ),
                                    circuit_identity=circuit_identities[
                                        environment_index
                                    ],
                                    interactions=len(episode_rows),
                                    undiscounted_return=float(
                                        returns[environment_index]
                                    ),
                                    outcome=racing_outcome(
                                        bool(terminated[environment_index]),
                                        bool(truncated[environment_index]),
                                        info,
                                    ),
                                    final_progress=progress,
                                    maximum_progress=float(
                                        maximum_progress[environment_index]
                                    ),
                                    mean_speed=float(
                                        speed_totals[environment_index]
                                        / len(episode_rows)
                                    ),
                                    mean_throttle_magnitude=float(
                                        throttle_totals[environment_index]
                                        / len(episode_rows)
                                    ),
                                )
                            )
                    observations = next_observations

                completed_records.sort(key=lambda record: record.episode_index)
                self.history.episodes.extend(completed_records)
                self._episode_batch.extend(
                    OnPolicyRollout(tuple(transitions[index]))
                    for index in range(wave_size)
                )
                next_episode_index += wave_size
                remaining_episodes -= wave_size
                progress_bar.update(wave_size)

                if len(self._episode_batch) == self.agent.collection_size:
                    output = self.agent.update(
                        AgentUpdateInput(
                            mode=CollectionMode.COMPLETE_EPISODES,
                            episodes=tuple(self._episode_batch),
                        )
                    )
                    self.history.updates.append(
                        EducationalUpdateRecord(
                            update_index=len(self.history.updates),
                            final_episode_index=max(
                                episode.transitions[-1].episode_identity
                                for episode in self._episode_batch
                            ),
                            training_interactions=self.history.training_interactions,
                            transition_count=sum(
                                len(episode.transitions)
                                for episode in self._episode_batch
                            ),
                            diagnostics=output.diagnostics,
                        )
                    )
                    self._episode_batch = []

                if on_episode_end is not None:
                    for episode_record in completed_records:
                        on_episode_end(episode_record, self.history)
        finally:
            progress_bar.close()
        return self.history

    def close(self) -> None:
        """
        Close the persistent environment processes owned by this engine.
        """
        self.environments.close()


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
