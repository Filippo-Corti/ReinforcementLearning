"""Semantic on-policy records, collection buffers, and fixed learning targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from recording.records import TrainingTransition
from utils.vectors import optional_tensor, to_vector


@dataclass(frozen=True, slots=True)
class OnPolicyTensors:
    """
    Bundle one rollout's detached tensor views for actor and critic updates.

    Fields:
        * observations: Float32 normalized observations with shape `(rows, features)`.
        * pre_squash_actions: Gaussian samples before `tanh`, shape `(rows, actions)`.
        * actions: Float32 bounded actions with shape `(rows, actions)`.
        * rewards: Float32 environment rewards with shape `(rows,)`.
        * behaviour_log_probabilities: Detached collection values when every row has one.
        * current_values: Detached critic values when every row has one.
        * next_values: Detached bootstrap values when every row has one.
        * terminated: Boolean true-MDP-ending mask.
        * truncated: Boolean time-limit mask.
        * next_observations: Float32 next normalized observations.
        * episode_identities: Integer episode identities.
        * circuit_identities: Stable circuit identities kept outside tensor arithmetic.
    """

    observations: Tensor
    pre_squash_actions: Tensor
    actions: Tensor
    rewards: Tensor
    behaviour_log_probabilities: Tensor | None
    current_values: Tensor | None
    next_values: Tensor | None
    terminated: Tensor
    truncated: Tensor
    next_observations: Tensor
    episode_identities: Tensor
    episode_step_indices: Tensor
    circuit_identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OnPolicyRollout:
    """
    Keep an ordered fixed collection of semantic on-policy transitions.

    Fields:
        * transitions: Ordered rows, which may span complete episode boundaries.
    """

    transitions: tuple[TrainingTransition, ...]

    def __post_init__(self) -> None:
        """
        Reject empty rollouts, whose targets would have no meaningful shape.
        """
        if not self.transitions:
            raise ValueError("A rollout must contain at least one transition.")

    def tensors(self, device: torch.device | str | None = None) -> OnPolicyTensors:
        """
        Return detached framework-ready tensors without changing stored rows.
        """
        rows = self.transitions
        return OnPolicyTensors(
            observations=torch.as_tensor(
                np.stack(
                    [
                        to_vector(
                            row.normalized_observation, name="normalized_observation"
                        )
                        for row in rows
                    ]
                ),
                device=device,
            ),
            pre_squash_actions=torch.as_tensor(
                np.stack(
                    [
                        to_vector(row.pre_squash_action, name="pre_squash_action")
                        for row in rows
                    ]
                ),
                device=device,
            ),
            actions=torch.as_tensor(
                np.stack([to_vector(row.action, name="action") for row in rows]),
                device=device,
            ),
            rewards=torch.tensor(
                [row.reward for row in rows], dtype=torch.float32, device=device
            ),
            behaviour_log_probabilities=optional_tensor(
                [row.behaviour_log_probability for row in rows], device=device
            ),
            current_values=optional_tensor(
                [row.current_value for row in rows], device=device
            ),
            next_values=optional_tensor(
                [row.next_value for row in rows], device=device
            ),
            terminated=torch.tensor(
                [row.terminated for row in rows], dtype=torch.bool, device=device
            ),
            truncated=torch.tensor(
                [row.truncated for row in rows], dtype=torch.bool, device=device
            ),
            next_observations=torch.as_tensor(
                np.stack(
                    [
                        to_vector(
                            row.next_normalized_observation,
                            name="next_normalized_observation",
                        )
                        for row in rows
                    ]
                ),
                device=device,
            ),
            episode_identities=torch.tensor(
                [row.episode_identity for row in rows], dtype=torch.int64, device=device
            ),
            episode_step_indices=torch.tensor(
                [row.episode_step_index for row in rows],
                dtype=torch.int64,
                device=device,
            ),
            circuit_identities=tuple(row.circuit_identity for row in rows),
        )


class ReinforceEpisodeBuffer:
    """
    Collect only complete, separate trajectories for Monte Carlo REINFORCE.

    Fields:
        * completed_episodes: Ordered finalized complete trajectories awaiting an update.
        * _active_episode: The currently collecting episode, never part of an update.
    """

    def __init__(self) -> None:
        """
        Initialize an empty complete-episode collection.
        """
        self.completed_episodes: list[OnPolicyRollout] = []
        self._active_episode: list[TrainingTransition] = []

    def append(self, transition: TrainingTransition) -> None:
        """
        Append one ordered row to the active episode without finalizing it.
        """
        self._validate_next_transition(self._active_episode, transition)
        self._active_episode.append(transition)

    def finalize_episode(self) -> OnPolicyRollout:
        """
        Move an environment-ended active episode into the completed collection.
        """
        if not self._active_episode:
            raise ValueError("No active episode is available to finalize.")
        if not self._active_episode[-1].ends_episode:
            raise ValueError(
                "REINFORCE only finalizes terminated or truncated episodes."
            )
        episode = OnPolicyRollout(tuple(self._active_episode))
        self.completed_episodes.append(episode)
        self._active_episode = []
        return episode

    def take_completed_batch(
        self, episode_count: int
    ) -> tuple[OnPolicyRollout, ...] | None:
        """
        Remove and return exactly the requested number of complete episodes.
        """
        if episode_count <= 0:
            raise ValueError("Episode batch size must be positive.")
        if len(self.completed_episodes) < episode_count:
            return None
        batch = tuple(self.completed_episodes[:episode_count])
        del self.completed_episodes[:episode_count]
        return batch

    def restore(
        self,
        completed_episodes: Sequence[OnPolicyRollout],
        active_episode: Sequence[TrainingTransition],
    ) -> None:
        """
        Restore checkpointed complete and active episode rows.
        """
        for episode in completed_episodes:
            if not episode.transitions[-1].ends_episode:
                raise ValueError("A restored completed episode must have ended.")
        validated_active: list[TrainingTransition] = []
        for transition in active_episode:
            self._validate_next_transition(validated_active, transition)
            validated_active.append(transition)
        if validated_active and validated_active[-1].ends_episode:
            raise ValueError("An active REINFORCE episode cannot already be ended.")
        self.completed_episodes = list(completed_episodes)
        self._active_episode = validated_active

    @property
    def active_episode(self) -> tuple[TrainingTransition, ...]:
        """
        Return the incomplete active trajectory without making it update-eligible.
        """
        return tuple(self._active_episode)

    @staticmethod
    def _validate_next_transition(
        rows: Sequence[TrainingTransition], transition: TrainingTransition
    ) -> None:
        if not rows:
            if transition.episode_step_index != 0:
                raise ValueError(
                    "A new complete episode must start at step index zero."
                )
            return
        previous = rows[-1]
        if previous.ends_episode:
            raise ValueError("Finalize an ended episode before appending another row.")
        if transition.episode_identity != previous.episode_identity:
            raise ValueError("REINFORCE episodes cannot mix episode identities.")
        if transition.circuit_identity != previous.circuit_identity:
            raise ValueError("REINFORCE episodes cannot mix circuit identities.")
        if transition.episode_step_index != previous.episode_step_index + 1:
            raise ValueError("Episode step indices must be consecutive.")


class FixedRolloutBuffer:
    """
    Collect a bounded ordered rollout that may span completed episode boundaries.

    Fields:
        * capacity: Maximum number of transitions in one rollout.
        * _transitions: Rows awaiting finalization into a target-computation batch.
        * _previous_transition: Last finalized row used to verify rollout continuity.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize an empty fixed-length rollout buffer.
        """
        if capacity <= 0:
            raise ValueError("Rollout capacity must be positive.")
        self.capacity = capacity
        self._transitions: list[TrainingTransition] = []
        self._previous_transition: TrainingTransition | None = None

    def append(self, transition: TrainingTransition) -> None:
        """
        Append one row, preserving episode boundaries and rejecting overflow.
        """
        if len(self._transitions) >= self.capacity:
            raise ValueError("Finalize the full rollout before appending another row.")
        preceding_rows: Sequence[TrainingTransition] = self._transitions
        if not preceding_rows and self._previous_transition is not None:
            preceding_rows = (self._previous_transition,)
        self._validate_next_transition(preceding_rows, transition)
        self._transitions.append(transition)

    def finalize(self) -> OnPolicyRollout:
        """
        Return and clear the current partial or full rollout exactly once.
        """
        rollout = OnPolicyRollout(tuple(self._transitions))
        self._previous_transition = self._transitions[-1]
        self._transitions = []
        return rollout

    @property
    def transitions(self) -> tuple[TrainingTransition, ...]:
        """
        Return the stored rows without allowing duplicate finalization by mutation.
        """
        return tuple(self._transitions)

    @property
    def previous_transition(self) -> TrainingTransition | None:
        """
        Return the last finalized row retained for boundary validation.
        """
        return self._previous_transition

    def restore(
        self,
        transitions: Sequence[TrainingTransition],
        previous_transition: TrainingTransition | None,
    ) -> None:
        """
        Restore checkpointed active rows and rollout-boundary context.
        """
        if len(transitions) > self.capacity:
            raise ValueError("Checkpointed rollout exceeds this buffer's capacity.")
        validated: list[TrainingTransition] = []
        if previous_transition is not None:
            validated.append(previous_transition)
        for transition in transitions:
            self._validate_next_transition(validated, transition)
            validated.append(transition)
        self._transitions = list(transitions)
        self._previous_transition = previous_transition

    @staticmethod
    def _validate_next_transition(
        rows: Sequence[TrainingTransition], transition: TrainingTransition
    ) -> None:
        if not rows:
            return
        previous = rows[-1]
        if transition.episode_identity == previous.episode_identity:
            if previous.ends_episode:
                raise ValueError("An ended episode cannot receive another transition.")
            if transition.circuit_identity != previous.circuit_identity:
                raise ValueError("One episode cannot change circuit identity.")
            if transition.episode_step_index != previous.episode_step_index + 1:
                raise ValueError("Episode step indices must be consecutive.")
            return
        if not previous.ends_episode:
            raise ValueError("A different episode must follow an environment boundary.")
        if transition.episode_step_index != 0:
            raise ValueError("A new episode must start at step index zero.")


@dataclass(frozen=True, slots=True)
class GAETargets:
    """
    Store fixed TD errors, raw GAE advantages, and critic targets for one rollout.

    Fields:
        * temporal_difference_errors: Detached one-step TD errors.
        * raw_advantages: Detached GAE values before actor-only standardization.
        * value_targets: Detached fixed critic targets equal to advantage plus old value.
    """

    temporal_difference_errors: Tensor
    raw_advantages: Tensor
    value_targets: Tensor


def monte_carlo_return_to_go(
    episode: OnPolicyRollout,
    discount: float,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Compute detached return-to-go values for one complete REINFORCE episode.
    """
    if not episode.transitions[-1].ends_episode:
        raise ValueError("Monte Carlo return-to-go requires a complete episode.")
    returns = torch.empty(len(episode.transitions), dtype=torch.float32, device=device)
    running_return = 0.0
    for index in range(len(episode.transitions) - 1, -1, -1):
        running_return = episode.transitions[index].reward + discount * running_return
        returns[index] = running_return
    return returns.detach()


def compute_gae_targets(
    rollout: OnPolicyRollout,
    discount: float,
    gae_lambda: float,
    *,
    device: torch.device | str | None = None,
) -> GAETargets:
    """
    Compute detached TD, GAE, and critic targets with the approved boundaries.
    """
    tensors = rollout.tensors(device=device)
    if tensors.current_values is None or tensors.next_values is None:
        raise ValueError(
            "GAE requires a current and bootstrap value on every rollout row."
        )

    values = tensors.current_values.detach()
    next_values = tensors.next_values.detach()
    bootstrap_values = torch.where(
        tensors.terminated, torch.zeros_like(next_values), next_values
    )
    temporal_difference_errors = (
        tensors.rewards + discount * bootstrap_values - values
    ).detach()
    raw_advantages = torch.empty_like(temporal_difference_errors)
    next_advantage = torch.zeros(
        (), dtype=torch.float32, device=temporal_difference_errors.device
    )

    for index in range(len(rollout.transitions) - 1, -1, -1):
        transition = rollout.transitions[index]
        recursion_ends = (
            index == len(rollout.transitions) - 1
            or transition.ends_episode
            or rollout.transitions[index + 1].episode_identity
            != transition.episode_identity
        )
        if recursion_ends:
            next_advantage = temporal_difference_errors[index]
        else:
            next_advantage = (
                temporal_difference_errors[index]
                + discount * gae_lambda * next_advantage
            )
        raw_advantages[index] = next_advantage

    value_targets = (raw_advantages + values).detach()
    return GAETargets(
        temporal_difference_errors=temporal_difference_errors,
        raw_advantages=raw_advantages.detach(),
        value_targets=value_targets,
    )
