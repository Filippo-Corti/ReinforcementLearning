"""Readable complete-episode training loop for REINFORCE."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agents import AgentUpdateInput, CollectionMode, ReinforceAgent
from configs import EnvironmentConfig
from envs.racing import RacingEnv
from envs.tracks import TrackWithGeometry

from ..buffers import OnPolicyRollout, TrainingTransition
from ..normalization import RunningObservationNormalizer
from .records import (
    EducationalEpisodeRecord,
    EducationalTrainingHistory,
    EducationalUpdateRecord,
    racing_outcome,
)


class ReinforceTrainingEngine:
    """
    Train REINFORCE with an explicit complete-trajectory agent-environment loop.

    Each episode is kept separate because the return-to-go in `docs/LEARNING.md`
    is computed only from its own rewards. Once the documented number of complete
    episodes has been collected, the engine gives that batch to the agent for one
    optimizer update. An incomplete final batch remains recorded but is not used.

    Fields:
        * agent: REINFORCE policy and optimizer.
        * tracks: Circuits available for selection at episode boundaries.
        * environment_config: Racing dynamics, reward, observation, and time limit.
        * normalizer: Running observation statistics updated only during training.
        * environment_reset_generator: Independent stream for reset seeds.
        * track_selection_generator: Independent stream for circuit selection.
        * history: Episode and optimizer-update records collected so far.
    """

    def __init__(
        self,
        agent: ReinforceAgent,
        tracks: Sequence[TrackWithGeometry],
        environment_config: EnvironmentConfig,
        normalizer: RunningObservationNormalizer,
        environment_reset_generator: np.random.Generator,
        track_selection_generator: np.random.Generator,
    ) -> None:
        """
        Construct an educational REINFORCE engine over one or more circuits.
        """
        if not tracks:
            raise ValueError("REINFORCE training requires at least one circuit.")
        self.agent = agent
        self.tracks = tuple(tracks)
        self.environment_config = environment_config
        self.normalizer = normalizer
        self.environment_reset_generator = environment_reset_generator
        self.track_selection_generator = track_selection_generator
        self.history = EducationalTrainingHistory()
        self._episode_batch: list[OnPolicyRollout] = []

    def train(self, episode_count: int) -> EducationalTrainingHistory:
        """
        Collect the requested complete episodes and update after each full batch.
        """
        if episode_count <= 0:
            raise ValueError("Episode count must be positive.")

        first_episode_index = len(self.history.episodes)
        for episode_index in range(
            first_episode_index, first_episode_index + episode_count
        ):
            # Circuit selection is independent from policy sampling and reset noise.
            track_index = int(
                self.track_selection_generator.integers(0, len(self.tracks))
            )
            track = self.tracks[track_index]
            circuit_identity = str(track.track.generation.seed)
            environment = RacingEnv(track, config=self.environment_config)
            reset_seed = int(
                self.environment_reset_generator.integers(0, 2**32, dtype=np.uint32)
            )

            episode_transitions: list[TrainingTransition] = []
            episode_return = 0.0
            maximum_progress = 0.0
            observation, _ = environment.reset(seed=reset_seed)

            try:
                while True:
                    # The current training observation enters the running statistics.
                    normalized_observation = self.normalizer.update_and_normalize(
                        observation
                    )
                    decision = self.agent.collect_action(normalized_observation)
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(decision.env_action)

                    # The next observation is normalized with frozen statistics. It
                    # will update the statistics only if it becomes a current input.
                    next_normalized_observation = self.normalizer.normalize(
                        next_observation
                    )
                    episode_transitions.append(
                        TrainingTransition(
                            normalized_observation=normalized_observation,
                            raw_action=decision.raw_action,
                            env_action=decision.env_action,
                            reward=float(reward),
                            behaviour_log_probability=(
                                decision.behaviour_log_probability
                            ),
                            current_value=None,
                            next_value=None,
                            terminated=terminated,
                            truncated=truncated,
                            next_normalized_observation=(next_normalized_observation),
                            episode_identity=episode_index,
                            episode_step_index=len(episode_transitions),
                            circuit_identity=circuit_identity,
                        )
                    )
                    self.history.training_interactions += 1
                    episode_return += float(reward)
                    progress = (
                        float(info["episode_progress"]) / track.track.track_length
                    )
                    maximum_progress = max(maximum_progress, progress)

                    if terminated or truncated:
                        self.history.episodes.append(
                            EducationalEpisodeRecord(
                                episode_index=episode_index,
                                circuit_identity=circuit_identity,
                                interactions=len(episode_transitions),
                                undiscounted_return=episode_return,
                                outcome=racing_outcome(terminated, truncated, info),
                                final_progress=progress,
                                maximum_progress=maximum_progress,
                            )
                        )
                        break
                    observation = next_observation
            finally:
                environment.close()

            # REINFORCE never mixes rewards between trajectories. The agent computes
            # one trajectory-sum loss per episode, then averages those losses.
            self._episode_batch.append(OnPolicyRollout(tuple(episode_transitions)))
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
                        final_episode_index=episode_index,
                        training_interactions=self.history.training_interactions,
                        transition_count=sum(
                            len(episode.transitions) for episode in self._episode_batch
                        ),
                        diagnostics=output.diagnostics,
                    )
                )
                self._episode_batch = []

        return self.history
