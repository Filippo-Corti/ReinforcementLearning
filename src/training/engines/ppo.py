"""Readable fixed-rollout training loop for clipped PPO."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from agents import AgentUpdateInput, CollectionMode, PPOAgent
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


class PPOTrainingEngine:
    """
    Train clipped PPO with a visible fixed-rollout agent-environment loop.

    As specified in `docs/LEARNING.md`, collection fixes the behaviour-policy
    log probability, GAE advantage, and critic target for every rollout row.
    The agent then reuses that fixed data for its configured seeded minibatch
    epochs. A rollout may cross episode boundaries, and a final shorter rollout
    is also used.

    Fields:
        * agent: PPO actor, critic, optimizers, and minibatch-order generator.
        * tracks: Circuits available for selection at episode boundaries.
        * environment_config: Racing dynamics, reward, observation, and time limit.
        * normalizer: Running observation statistics updated only during training.
        * environment_reset_generator: Independent stream for reset seeds.
        * track_selection_generator: Independent stream for circuit selection.
        * history: Episode and optimizer-update records collected so far.
    """

    def __init__(
        self,
        agent: PPOAgent,
        tracks: Sequence[TrackWithGeometry],
        environment_config: EnvironmentConfig,
        normalizer: RunningObservationNormalizer,
        environment_reset_generator: np.random.Generator,
        track_selection_generator: np.random.Generator,
    ) -> None:
        """
        Construct an educational PPO engine over one or more circuits.
        """
        if not tracks:
            raise ValueError("PPO training requires at least one circuit.")
        self.agent = agent
        self.tracks = tuple(tracks)
        self.environment_config = environment_config
        self.normalizer = normalizer
        self.environment_reset_generator = environment_reset_generator
        self.track_selection_generator = track_selection_generator
        self.history = EducationalTrainingHistory()

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
        Collect episodes and optionally report each completed episode to a caller.
        """
        if episode_count <= 0:
            raise ValueError("Episode count must be positive.")

        rollout_transitions: list[TrainingTransition] = []
        first_episode_index = len(self.history.episodes)
        final_episode_index = first_episode_index

        for episode_index in tqdm(
            range(first_episode_index, first_episode_index + episode_count),
            desc="Training episodes",
            unit="episode",
        ):
            final_episode_index = episode_index
            track_index = int(
                self.track_selection_generator.integers(0, len(self.tracks))
            )
            track = self.tracks[track_index]
            circuit_identity = str(track.track.generation.seed)
            environment = RacingEnv(track, config=self.environment_config)
            reset_seed = int(
                self.environment_reset_generator.integers(0, 2**32, dtype=np.uint32)
            )

            episode_interactions = 0
            episode_return = 0.0
            maximum_progress = 0.0
            observation, _ = environment.reset(seed=reset_seed)

            try:
                while True:
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
                    next_normalized_observation = self.normalizer.normalize(
                        next_observation
                    )

                    # A true termination has no future value. At a time-limit
                    # truncation the critic value is retained, while GAE recursion
                    # still stops at that episode boundary inside the agent.
                    next_value = (
                        0.0
                        if terminated
                        else self.agent.bootstrap_value(next_normalized_observation)
                    )
                    rollout_transitions.append(
                        TrainingTransition(
                            normalized_observation=normalized_observation,
                            raw_action=decision.raw_action,
                            env_action=decision.env_action,
                            reward=float(reward),
                            behaviour_log_probability=(
                                decision.behaviour_log_probability
                            ),
                            current_value=decision.current_value,
                            next_value=next_value,
                            terminated=terminated,
                            truncated=truncated,
                            next_normalized_observation=(next_normalized_observation),
                            episode_identity=episode_index,
                            episode_step_index=episode_interactions,
                            circuit_identity=circuit_identity,
                        )
                    )
                    episode_interactions += 1
                    self.history.training_interactions += 1
                    episode_return += float(reward)
                    progress = (
                        float(info["episode_progress"]) / track.track.track_length
                    )
                    maximum_progress = max(maximum_progress, progress)

                    if len(rollout_transitions) == self.agent.collection_size:
                        rollout = OnPolicyRollout(tuple(rollout_transitions))
                        output = self.agent.update(
                            AgentUpdateInput(
                                mode=CollectionMode.FIXED_ROLLOUT,
                                rollout=rollout,
                            )
                        )
                        self.history.updates.append(
                            EducationalUpdateRecord(
                                update_index=len(self.history.updates),
                                final_episode_index=episode_index,
                                training_interactions=(
                                    self.history.training_interactions
                                ),
                                transition_count=len(rollout.transitions),
                                diagnostics=output.diagnostics,
                            )
                        )
                        rollout_transitions = []

                    if terminated or truncated:
                        episode_record = EducationalEpisodeRecord(
                            episode_index=episode_index,
                            circuit_identity=circuit_identity,
                            interactions=episode_interactions,
                            undiscounted_return=episode_return,
                            outcome=racing_outcome(terminated, truncated, info),
                            final_progress=progress,
                            maximum_progress=maximum_progress,
                        )
                        self.history.episodes.append(episode_record)
                        if on_episode_end is not None:
                            on_episode_end(episode_record, self.history)
                        break
                    observation = next_observation
            finally:
                environment.close()

        # PPO keeps the same fixed targets and behaviour log probabilities even
        # when the interaction boundary produces a shorter final rollout.
        if rollout_transitions:
            rollout = OnPolicyRollout(tuple(rollout_transitions))
            output = self.agent.update(
                AgentUpdateInput(
                    mode=CollectionMode.FIXED_ROLLOUT,
                    rollout=rollout,
                )
            )
            self.history.updates.append(
                EducationalUpdateRecord(
                    update_index=len(self.history.updates),
                    final_episode_index=final_episode_index,
                    training_interactions=self.history.training_interactions,
                    transition_count=len(rollout.transitions),
                    diagnostics=output.diagnostics,
                )
            )

        return self.history
