"""The transitions an on-policy update learns from, and the buffer they land in.

A `TrainingTransition` is one step of one car: what the policy saw, what it
did, and what came back. A `Trajectory` is however many of those a single
worker has produced so far, in order, and it is the same object whether that
worker is still driving or has just crashed -- an episode is finished by the
environment, not by changing container.

The multi-worker counterpart lives in `multienvs`, because a rollout laid out
across parallel workers is only meaningful there.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from utils.vectors import optional_scalar, to_vector

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
        * environment_index: Stable vector-worker column that produced the transition.
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

    def validate_follows(self, previous: TrainingTransition | None) -> None:
        """
        Reject this transition if it cannot legally follow `previous`.

        A collection is a continuous stretch of one worker's experience, so the
        only two legal continuations are the next step of the same episode or
        the first step of a new one. Anything else means a step was lost or two
        workers were mixed, and the resulting returns would be silently wrong
        rather than obviously broken, so it is checked as transitions arrive.
        """
        if previous is None:
            return
        if self.episode_identity == previous.episode_identity:
            if previous.ends_episode:
                raise ValueError("An ended episode cannot receive another transition.")
            if self.circuit_identity != previous.circuit_identity:
                raise ValueError("One episode cannot change circuit identity.")
            if self.episode_step_index != previous.episode_step_index + 1:
                raise ValueError("Episode step indices must be consecutive.")
            return
        if not previous.ends_episode:
            raise ValueError("A different episode must follow an environment boundary.")
        if self.episode_step_index != 0:
            raise ValueError("A new episode must start at step index zero.")


class Trajectory:
    """
    Accumulate one worker's transitions, in the order it drove them.

    This is the only container for a run of transitions, in flight or finished.
    A worker appends to it while it races and hands the same object over once
    the episode ends, so nothing has to be copied into a second "completed"
    type at the boundary. Whether the run is complete is a property of its last
    transition, which is exactly where the environment recorded it.

    Fields:
        * transitions: What has been collected so far, first step to last.
    """

    def __init__(self, transitions: Iterable[TrainingTransition] = ()) -> None:
        """
        Open a trajectory, optionally restoring transitions already collected.
        """
        self._transitions: list[TrainingTransition] = []
        for transition in transitions:
            self.append(transition)

    def append(self, transition: TrainingTransition) -> None:
        """
        Extend the trajectory, refusing a transition that cannot follow the last.
        """
        transition.validate_follows(self.last)
        self._transitions.append(transition)

    @property
    def transitions(self) -> tuple[TrainingTransition, ...]:
        """
        Return what has been collected, without exposing the mutable list.
        """
        return tuple(self._transitions)

    @property
    def last(self) -> TrainingTransition | None:
        """
        Return the most recent transition, or `None` before the first one.
        """
        return self._transitions[-1] if self._transitions else None

    @property
    def is_complete(self) -> bool:
        """
        Return whether the run reached an environment boundary.

        Only a complete run has a Monte Carlo return: until the car crashes or
        crosses the line, the return from every state in it is still unknown.
        """
        return bool(self._transitions) and self._transitions[-1].ends_episode

    def clear(self) -> None:
        """
        Drop every transition, leaving the trajectory ready to collect again.
        """
        self._transitions = []

    def __len__(self) -> int:
        """
        Return how many transitions have been collected.
        """
        return len(self._transitions)

    def __iter__(self) -> Iterator[TrainingTransition]:
        """
        Iterate the collected transitions in order.
        """
        return iter(self._transitions)

    def __getitem__(self, index: int) -> TrainingTransition:
        """
        Return one collected transition by position.
        """
        return self._transitions[index]
