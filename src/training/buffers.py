"""
Semantic on-policy records, collection buffers, and fixed learning targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from utils.vectors import optional_scalar, optional_tensor, to_vector

VectorInput = NDArray[np.float32] | Tensor


@dataclass(frozen=True, slots=True)
class TrainingTransition:
    """
    Store the detached transition data consumed by on-policy updates.

    The training buffer retains normalized network inputs, raw policy actions,
    behaviour probabilities, and critic estimates required to reproduce an
    update. It is checkpoint state, not an experiment log record.

    Fields:
        * normalized_observation: Exact network input before the action.
        * raw_action: Action before policy-specific post-processing.
        * env_action: Bounded action sent to the environment.
        * reward: Environment reward returned after the action.
        * behaviour_log_probability: Collection-policy log probability.
        * current_value: Critic value at the current observation, if used.
        * next_value: Bootstrap value, zero at true termination.
        * terminated: Whether a genuine MDP ending occurred.
        * truncated: Whether the external time limit occurred.
        * next_normalized_observation: Frozen-statistics next-state input.
        * episode_identity: Stable identity of the environment episode.
        * episode_step_index: Zero-based position within that episode.
        * circuit_identity: Stable identity of the episode's circuit.
        * environment_index: Stable vector-worker column that produced the row.
    """

    normalized_observation: VectorInput
    raw_action: VectorInput
    env_action: VectorInput
    reward: float
    behaviour_log_probability: float | Tensor | None
    current_value: float | Tensor | None
    next_value: float | Tensor | None
    terminated: bool
    truncated: bool
    next_normalized_observation: VectorInput
    episode_identity: int
    episode_step_index: int
    circuit_identity: str
    environment_index: int = 0

    def __post_init__(self) -> None:
        """
        Detach scalar tensors and preserve read-only float32 vector copies.
        """
        observation = to_vector(
            self.normalized_observation,
            name="normalized_observation",
            readonly=True,
        )
        raw_action = to_vector(self.raw_action, name="raw_action", readonly=True)
        env_action = to_vector(self.env_action, name="env_action", readonly=True)
        next_observation = to_vector(
            self.next_normalized_observation,
            name="next_normalized_observation",
            readonly=True,
        )
        if observation.shape != next_observation.shape:
            raise ValueError(
                "Current and next normalized observations must have one shape."
            )
        if raw_action.shape != env_action.shape:
            raise ValueError("Raw and environment actions must have one shape.")
        if self.terminated and self.truncated:
            raise ValueError("A transition cannot be both terminated and truncated.")
        if self.episode_step_index < 0:
            raise ValueError("Episode step indices cannot be negative.")
        if self.environment_index < 0:
            raise ValueError("Environment indices cannot be negative.")
        next_value = optional_scalar(self.next_value)
        if self.terminated and next_value not in (None, 0.0):
            raise ValueError("A true termination must store a zero bootstrap value.")
        object.__setattr__(self, "normalized_observation", observation)
        object.__setattr__(self, "raw_action", raw_action)
        object.__setattr__(self, "env_action", env_action)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(
            self,
            "behaviour_log_probability",
            optional_scalar(self.behaviour_log_probability),
        )
        object.__setattr__(self, "current_value", optional_scalar(self.current_value))
        object.__setattr__(self, "next_value", next_value)
        object.__setattr__(self, "next_normalized_observation", next_observation)

    @property
    def ends_episode(self) -> bool:
        """
        Return whether this transition reaches an environment boundary.
        """
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class OnPolicyTensors:
    """
    Bundle one rollout's detached tensor views for actor and critic updates.

    Fields:
        * observations: Float32 normalized observations with shape `(rows, features)`.
        * raw_actions: Actions before policy-specific post-processing.
        * env_actions: Float32 bounded environment actions.
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
    raw_actions: Tensor
    env_actions: Tensor
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
            raw_actions=torch.as_tensor(
                np.stack(
                    [to_vector(row.raw_action, name="raw_action") for row in rows]
                ),
                device=device,
            ),
            env_actions=torch.as_tensor(
                np.stack(
                    [to_vector(row.env_action, name="env_action") for row in rows]
                ),
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


@dataclass(frozen=True, slots=True)
class VectorOnPolicyTensors:
    """
    Bundle a time-by-environment rollout and its valid-transition mask.

    Fields:
        * observations: Normalized inputs with shape `(time, environments, features)`.
        * raw_actions: Pre-squash actions with shape `(time, environments, actions)`.
        * env_actions: Bounded actions with shape `(time, environments, actions)`.
        * rewards: Rewards with shape `(time, environments)`.
        * behaviour_log_probabilities: Collection log probabilities when present.
        * current_values: Detached current critic estimates when present.
        * next_values: Detached bootstrap estimates when present.
        * terminated: True-MDP-ending mask.
        * truncated: Time-limit-ending mask.
        * next_observations: Frozen-statistics next inputs.
        * episode_identities: Episode identity per valid row.
        * episode_step_indices: Within-episode step index per valid row.
        * valid: Whether a time/environment cell contains a collected transition.
        * circuit_identities: Circuit identity or `None` per cell.
    """

    observations: Tensor
    raw_actions: Tensor
    env_actions: Tensor
    rewards: Tensor
    behaviour_log_probabilities: Tensor | None
    current_values: Tensor | None
    next_values: Tensor | None
    terminated: Tensor
    truncated: Tensor
    next_observations: Tensor
    episode_identities: Tensor
    episode_step_indices: Tensor
    valid: Tensor
    circuit_identities: tuple[tuple[str | None, ...], ...]

    def flatten_valid(self, values: Tensor) -> Tensor:
        """
        Return valid cells in deterministic time-major, environment-minor order.
        """
        return values[self.valid]


@dataclass(frozen=True, slots=True)
class VectorOnPolicyRollout:
    """
    Keep fixed-rollout transitions in explicit time-by-environment columns.

    A `None` cell means that worker was intentionally parked for that collection
    tick. Valid cells retain their environment column, so target recursion never
    crosses from one process into another.

    Fields:
        * transition_steps: Ordered `(time, environments)` semantic rows.
    """

    transition_steps: tuple[tuple[TrainingTransition | None, ...], ...]

    def __post_init__(self) -> None:
        """
        Require a non-empty rectangular collection with at least one valid row.
        """
        if not self.transition_steps:
            raise ValueError("A vector rollout must contain at least one time step.")
        environment_count = len(self.transition_steps[0])
        if environment_count == 0:
            raise ValueError("A vector rollout must contain an environment column.")
        if any(len(step) != environment_count for step in self.transition_steps):
            raise ValueError("Vector rollout steps must have one rectangular shape.")
        if not self.transitions:
            raise ValueError("A vector rollout must contain a valid transition.")
        for environment_index in range(environment_count):
            previous: TrainingTransition | None = None
            for step in self.transition_steps:
                transition = step[environment_index]
                if transition is None:
                    continue
                if transition.environment_index != environment_index:
                    raise ValueError(
                        "A transition environment index must match its vector column."
                    )
                if previous is not None:
                    FixedRolloutBuffer._validate_next_transition(
                        (previous,), transition
                    )
                previous = transition

    @property
    def environment_count(self) -> int:
        """
        Return the number of persistent worker columns.
        """
        return len(self.transition_steps[0])

    @property
    def transitions(self) -> tuple[TrainingTransition, ...]:
        """
        Return valid transitions in time-major, environment-minor order.
        """
        return tuple(
            transition
            for step in self.transition_steps
            for transition in step
            if transition is not None
        )

    def tensors(
        self, device: torch.device | str | None = None
    ) -> VectorOnPolicyTensors:
        """
        Return padded framework tensors while retaining an explicit valid mask.
        """
        first = self.transitions[0]
        time_count = len(self.transition_steps)
        environment_count = self.environment_count
        observation_dimensions = first.normalized_observation.shape[0]
        action_dimensions = first.raw_action.shape[0]
        valid = torch.zeros(
            (time_count, environment_count), dtype=torch.bool, device=device
        )
        observations = torch.zeros(
            (time_count, environment_count, observation_dimensions),
            dtype=torch.float32,
            device=device,
        )
        raw_actions = torch.zeros(
            (time_count, environment_count, action_dimensions),
            dtype=torch.float32,
            device=device,
        )
        env_actions = torch.zeros_like(raw_actions)
        rewards = torch.zeros(
            (time_count, environment_count), dtype=torch.float32, device=device
        )
        terminated = torch.zeros_like(valid)
        truncated = torch.zeros_like(valid)
        next_observations = torch.zeros_like(observations)
        episode_identities = torch.full(
            (time_count, environment_count), -1, dtype=torch.int64, device=device
        )
        episode_step_indices = torch.full_like(episode_identities, -1)
        circuit_identities: list[tuple[str | None, ...]] = []

        for time_index, step in enumerate(self.transition_steps):
            circuits: list[str | None] = []
            for environment_index, transition in enumerate(step):
                if transition is None:
                    circuits.append(None)
                    continue
                valid[time_index, environment_index] = True
                observations[time_index, environment_index] = torch.tensor(
                    transition.normalized_observation, device=device
                )
                raw_actions[time_index, environment_index] = torch.tensor(
                    transition.raw_action, device=device
                )
                env_actions[time_index, environment_index] = torch.tensor(
                    transition.env_action, device=device
                )
                rewards[time_index, environment_index] = transition.reward
                terminated[time_index, environment_index] = transition.terminated
                truncated[time_index, environment_index] = transition.truncated
                next_observations[time_index, environment_index] = torch.tensor(
                    transition.next_normalized_observation, device=device
                )
                episode_identities[time_index, environment_index] = (
                    transition.episode_identity
                )
                episode_step_indices[time_index, environment_index] = (
                    transition.episode_step_index
                )
                circuits.append(transition.circuit_identity)
            circuit_identities.append(tuple(circuits))

        return VectorOnPolicyTensors(
            observations=observations,
            raw_actions=raw_actions,
            env_actions=env_actions,
            rewards=rewards,
            behaviour_log_probabilities=self._optional_scalar_tensor(
                "behaviour_log_probability", valid, device
            ),
            current_values=self._optional_scalar_tensor("current_value", valid, device),
            next_values=self._optional_scalar_tensor("next_value", valid, device),
            terminated=terminated,
            truncated=truncated,
            next_observations=next_observations,
            episode_identities=episode_identities,
            episode_step_indices=episode_step_indices,
            valid=valid,
            circuit_identities=tuple(circuit_identities),
        )

    def _optional_scalar_tensor(
        self,
        field_name: str,
        valid: Tensor,
        device: torch.device | str | None,
    ) -> Tensor | None:
        values = [getattr(transition, field_name) for transition in self.transitions]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Vector rollout field {field_name} is incomplete.")
        tensor = torch.zeros(valid.shape, dtype=torch.float32, device=device)
        tensor[valid] = torch.tensor(values, dtype=torch.float32, device=device)
        return tensor


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


class VectorRolloutBuffer:
    """
    Collect a pooled transition count in rectangular environment-time steps.

    Fields:
        * capacity: Maximum number of valid transitions in one pooled rollout.
        * environment_count: Number of persistent worker columns.
        * _steps: Time-ordered rows with `None` for intentionally parked workers.
        * _previous: Last valid transition by environment column.
    """

    def __init__(self, capacity: int, environment_count: int) -> None:
        """
        Initialize an empty pooled vector rollout.
        """
        if capacity <= 0:
            raise ValueError("Rollout capacity must be positive.")
        if environment_count <= 0:
            raise ValueError("Vector rollouts require a positive environment count.")
        self.capacity = capacity
        self.environment_count = environment_count
        self._steps: list[tuple[TrainingTransition | None, ...]] = []
        self._previous: list[TrainingTransition | None] = [None] * environment_count
        self._transition_count = 0

    def append_step(
        self,
        transitions: Sequence[TrainingTransition | None],
    ) -> None:
        """
        Append one synchronous worker step without exceeding pooled capacity.
        """
        step = tuple(transitions)
        if len(step) != self.environment_count:
            raise ValueError("A vector step must contain one cell per environment.")
        valid_count = sum(transition is not None for transition in step)
        if valid_count == 0:
            raise ValueError("A vector rollout step must contain a valid transition.")
        if self._transition_count + valid_count > self.capacity:
            raise ValueError("Finalize the full vector rollout before appending rows.")
        for environment_index, transition in enumerate(step):
            if transition is None:
                continue
            if transition.environment_index != environment_index:
                raise ValueError(
                    "A transition environment index must match its vector column."
                )
            previous = self._previous[environment_index]
            if previous is not None:
                FixedRolloutBuffer._validate_next_transition((previous,), transition)
            self._previous[environment_index] = transition
        self._steps.append(step)
        self._transition_count += valid_count

    def finalize(self) -> VectorOnPolicyRollout:
        """
        Return and clear the current partial or full rectangular rollout.
        """
        rollout = VectorOnPolicyRollout(tuple(self._steps))
        self._steps = []
        self._transition_count = 0
        return rollout

    @property
    def transition_count(self) -> int:
        """
        Return the number of valid pooled transitions currently stored.
        """
        return self._transition_count

    @property
    def remaining_capacity(self) -> int:
        """
        Return how many further valid transitions fit before finalization.
        """
        return self.capacity - self._transition_count

    @property
    def transition_steps(
        self,
    ) -> tuple[tuple[TrainingTransition | None, ...], ...]:
        """
        Return stored time/environment cells without exposing mutable lists.
        """
        return tuple(self._steps)

    @property
    def previous_transitions(self) -> tuple[TrainingTransition | None, ...]:
        """
        Return the last valid row retained for each worker column.
        """
        return tuple(self._previous)

    def restore(
        self,
        transition_steps: Sequence[Sequence[TrainingTransition | None]],
        previous_transitions: Sequence[TrainingTransition | None],
    ) -> None:
        """
        Restore checkpointed cells and column-specific continuity context.
        """
        if len(previous_transitions) != self.environment_count:
            raise ValueError("Vector checkpoint column count does not match.")
        self._steps = []
        self._previous = [None] * self.environment_count
        self._transition_count = 0
        for step in transition_steps:
            self.append_step(step)
        restored_previous = tuple(previous_transitions)
        for environment_index, previous in enumerate(restored_previous):
            current = self._previous[environment_index]
            if current is not None and previous != current:
                raise ValueError("Vector checkpoint continuity state is inconsistent.")
        self._previous = list(restored_previous)


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


def compute_vector_gae_targets(
    rollout: VectorOnPolicyRollout,
    discount: float,
    gae_lambda: float,
    *,
    device: torch.device | str | None = None,
) -> GAETargets:
    """
    Compute GAE backward within each environment column independently.
    """
    tensors = rollout.tensors(device=device)
    if tensors.current_values is None or tensors.next_values is None:
        raise ValueError(
            "GAE requires a current and bootstrap value on every valid vector row."
        )
    values = tensors.current_values.detach()
    next_values = tensors.next_values.detach()
    bootstrap_values = torch.where(
        tensors.terminated, torch.zeros_like(next_values), next_values
    )
    temporal_difference_errors = torch.where(
        tensors.valid,
        tensors.rewards + discount * bootstrap_values - values,
        torch.zeros_like(tensors.rewards),
    ).detach()
    raw_advantages = torch.zeros_like(temporal_difference_errors)

    for environment_index in range(rollout.environment_count):
        valid_times = torch.nonzero(
            tensors.valid[:, environment_index], as_tuple=False
        ).flatten()
        next_advantage = torch.zeros(
            (), dtype=torch.float32, device=temporal_difference_errors.device
        )
        next_transition: TrainingTransition | None = None
        for valid_position in range(valid_times.numel() - 1, -1, -1):
            time_index = int(valid_times[valid_position].item())
            transition = rollout.transition_steps[time_index][environment_index]
            if transition is None:
                raise RuntimeError("A valid vector mask cannot refer to an empty cell.")
            recursion_ends = (
                next_transition is None
                or transition.ends_episode
                or next_transition.episode_identity != transition.episode_identity
            )
            if recursion_ends:
                next_advantage = temporal_difference_errors[
                    time_index, environment_index
                ]
            else:
                next_advantage = (
                    temporal_difference_errors[time_index, environment_index]
                    + discount * gae_lambda * next_advantage
                )
            raw_advantages[time_index, environment_index] = next_advantage
            next_transition = transition

    value_targets = torch.where(
        tensors.valid,
        raw_advantages + values,
        torch.zeros_like(values),
    ).detach()
    return GAETargets(
        temporal_difference_errors=temporal_difference_errors,
        raw_advantages=raw_advantages.detach(),
        value_targets=value_targets,
    )
