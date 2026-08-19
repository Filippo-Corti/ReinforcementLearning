"""What every on-policy agent owes the engine, and the little it can share.

An agent owns models, optimizers, the random streams it samples from, and its
serializable state. The engine owns environment interaction, normalization,
episode lifecycle and accounting. That boundary is the reason this class
exists: it is the whole list of things a training loop is allowed to ask of an
algorithm, so the loop can be written once for three of them.

The concrete methods here are deliberately unexciting. They are the parts that
are about *being an agent in this project* rather than about the algorithm --
resolving which sampling stream serves which worker, turning stored transitions
into the two tensors every actor needs. Anything that differs between REINFORCE,
A2C and PPO is abstract, because those differences are the algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from utils.vectors import stack_vectors

from ..types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectedActionBatch,
    CollectionMode,
)

# Imported for annotations only. Nothing here touches a training object at
# runtime -- an agent is handed records and reads their attributes -- so this
# stays inside TYPE_CHECKING and `agents` keeps importing nothing from
# `training` when the interpreter actually loads it. That one-directional
# runtime graph is what stops the two packages deadlocking on each other.
if TYPE_CHECKING:
    from training.buffers import TrainingTransition


class OnPolicyAgent(ABC):
    """
    Define the learning boundary for bounded continuous on-policy agents.

    Subclasses declare `collection_mode` as a class attribute, because it is a
    property of the algorithm rather than of an instance: an engine reads it to
    decide whether to collect whole episodes or a fixed rollout, and it must be
    answerable before any learning has happened.

    Fields:
        * collection_size: Episodes or transitions gathered before one update.
        * device: Where owned models and the tensors built here live.
        * dtype: Floating precision the actor and critic are evaluated in.
        * sampling_generators: One isolated policy-sampling stream per worker.
    """

    collection_mode: CollectionMode
    collection_size: int
    device: torch.device
    dtype: torch.dtype
    sampling_generators: tuple[torch.Generator, ...]

    @abstractmethod
    def collect_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> CollectedAction:
        """
        Sample one stochastic bounded action for training collection.
        """

    @abstractmethod
    def collect_actions(
        self,
        normalized_observations: NDArray[np.float32],
        environment_indices: Sequence[int] | None = None,
    ) -> CollectedActionBatch:
        """
        Sample one stochastic action per vector-environment observation.
        """

    @abstractmethod
    def deterministic_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Return one bounded action without consuming a training random stream.
        """

    @abstractmethod
    def bootstrap_value(
        self, normalized_observation: NDArray[np.float32]
    ) -> float | None:
        """
        Return a detached critic value for one bootstrap state when applicable.
        """

    @abstractmethod
    def bootstrap_values(
        self, normalized_observations: NDArray[np.float32]
    ) -> NDArray[np.float32] | None:
        """
        Return detached critic values for a batch of bootstrap states.
        """

    @abstractmethod
    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Optimize owned state from detached collected records.

        Implementations accept only the input subclass their collection mode
        produces, and reject the other one rather than trying to interpret it.
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """
        Return all owned model, optimizer and mutable generator state.
        """

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore all owned model, optimizer and mutable generator state.
        """

    @property
    @abstractmethod
    def actor_parameter_count(self) -> int:
        """
        Return the number of trainable actor parameters.
        """

    @property
    @abstractmethod
    def critic_parameter_count(self) -> int | None:
        """
        Return trainable critic parameters, or `None` for actor-only agents.
        """

    def _policy_inputs(
        self, transitions: Sequence[TrainingTransition]
    ) -> tuple[Tensor, Tensor]:
        """
        Build the observation and pre-squash action tensors an actor scores.

        These two are all any of the three actors ever needs from a stored
        transition, so only these two are built. The other recorded fields
        belong to the critic, to the target math, or to the run log, and each
        of those converts what it needs where it needs it.
        """
        observations = stack_vectors(
            (transition.normalized_observation for transition in transitions),
            name="normalized_observation",
            device=self.device,
        ).to(dtype=self.dtype)
        raw_actions = stack_vectors(
            (transition.raw_action for transition in transitions),
            name="raw_action",
            device=self.device,
        ).to(dtype=self.dtype)
        return observations, raw_actions

    def _resolve_stream_indices(
        self,
        row_count: int,
        environment_indices: Sequence[int] | None,
    ) -> tuple[int, ...]:
        """
        Resolve and validate which sampling-generator stream serves each action row.

        Defaults to one stream per row, in order, when no explicit environment
        indices are given. Always checks that there is exactly one index per row
        and that every index names an existing sampling stream.
        """
        indices = (
            tuple(range(row_count))
            if environment_indices is None
            else tuple(environment_indices)
        )
        if len(indices) != row_count:
            raise ValueError("One environment index is required per action row.")
        if any(
            index < 0 or index >= len(self.sampling_generators) for index in indices
        ):
            raise ValueError("Policy-sampling environment index is out of range.")
        return indices

    @staticmethod
    def _sampling_generators(
        generators: torch.Generator | Sequence[torch.Generator],
    ) -> tuple[torch.Generator, ...]:
        """
        Accept one sampling stream, or one per worker, as a fixed tuple.
        """
        if isinstance(generators, torch.Generator):
            return (generators,)
        values = tuple(generators)
        if not values:
            raise ValueError("At least one policy-sampling generator is required.")
        return values
