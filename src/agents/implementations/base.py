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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from configs import ActorConfig, CriticConfig
from utils.vectors import stack_vectors, to_tensor

from ..diagnostics import gradient_dispersion
from ..models import ActorNetwork, CriticNetwork
from ..types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectedActionBatch,
    CollectionMode,
)

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
    actor: ActorNetwork
    actor_optimizer: torch.optim.Optimizer
    actor_learning_rate: float
    actor_config: ActorConfig
    sampling_generators: tuple[torch.Generator, ...]
    gradient_dispersion_subbatch: int | None

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        *,
        collection_size: int,
        adam_betas: tuple[float, float],
        optimizer_epsilon: float,
        actor_initialization_generator: torch.Generator,
        sampling_generator: torch.Generator | Sequence[torch.Generator],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        gradient_dispersion_subbatch: int | None = 256,
    ) -> None:
        """
        Build the policy, its optimizer, and the streams it samples from.

        The optimizer settings arrive as values rather than as the algorithm's
        config object, so that this stays independent of which algorithm is
        being constructed: every one of them keeps these knobs, but each keeps
        them in a config of its own type alongside fields nothing here needs.

        `collection_size` is passed in for the same reason. It is the same idea
        everywhere -- how much to gather before learning -- but each algorithm
        measures it in its own unit, episodes or transitions, and only the
        subclass knows which.
        """
        if actor_config.learning_rate is None:
            raise ValueError("ActorConfig requires an explicit learning rate.")
        self.actor_config = actor_config
        self.collection_size = collection_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.gradient_dispersion_subbatch = gradient_dispersion_subbatch
        self._adam_betas = adam_betas
        self._optimizer_epsilon = optimizer_epsilon
        self.actor_learning_rate = float(actor_config.learning_rate)
        self.actor = ActorNetwork(
            observation_dimensions,
            actor_config,
            actor_initialization_generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.actor_optimizer = self._adam(
            self.actor.parameters(), self.actor_learning_rate
        )
        self.sampling_generators = self._sampling_generators(sampling_generator)

    def _adam(
        self, parameters: Iterable[torch.nn.Parameter], learning_rate: float
    ) -> torch.optim.Adam:
        """
        Build one Adam optimizer over a single network's parameters.

        Actor and critic get separate optimizers built the same way. Separate
        is the point: one optimizer over both would let the value loss carry
        moment estimates into the policy's parameters.
        """
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=self._adam_betas,
            eps=self._optimizer_epsilon,
        )

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

    def collect_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> CollectedAction:
        """
        Sample one stochastic bounded action for training collection.
        """
        observation = to_tensor(
            normalized_observation, dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        sample = self.actor.sample(observation, self.sampling_generators[0])
        values = self._critic_values(observation)
        return CollectedAction(
            raw_action=sample.raw_action[0].cpu().numpy(),
            env_action=sample.env_action[0].cpu().numpy(),
            behaviour_log_probability=float(sample.log_probability[0].item()),
            current_value=None if values is None else float(values[0]),
        )

    def collect_actions(
        self,
        normalized_observations: NDArray[np.float32],
        environment_indices: Sequence[int] | None = None,
    ) -> CollectedActionBatch:
        """
        Sample one stochastic action per vector-environment observation.
        """
        observations = to_tensor(
            normalized_observations, dtype=self.dtype, device=self.device
        )
        indices = self._resolve_stream_indices(
            observations.shape[0], environment_indices
        )
        sample = self.actor.sample_with_generators(
            observations,
            tuple(self.sampling_generators[index] for index in indices),
        )
        return CollectedActionBatch(
            raw_actions=sample.raw_action.cpu().numpy(),
            env_actions=sample.env_action.cpu().numpy(),
            behaviour_log_probabilities=sample.log_probability.cpu().numpy(),
            current_values=self._critic_values(observations),
        )

    def deterministic_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Return the actor mean action without advancing a training random stream.
        """
        observation = to_tensor(
            normalized_observation, dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor.deterministic_action(observation)[0]
        return action.cpu().numpy().astype(np.float32, copy=False)

    def bootstrap_value(
        self, normalized_observation: NDArray[np.float32]
    ) -> float | None:
        """
        Return a detached critic value for one bootstrap state, if there is a critic.
        """
        observation = to_tensor(
            normalized_observation, dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        values = self._critic_values(observation)
        return None if values is None else float(values[0])

    def bootstrap_values(
        self, normalized_observations: NDArray[np.float32]
    ) -> NDArray[np.float32] | None:
        """
        Return detached critic values for a batch of bootstrap states, if there is a critic.
        """
        return self._critic_values(
            to_tensor(normalized_observations, dtype=self.dtype, device=self.device)
        )

    @property
    def actor_parameter_count(self) -> int:
        """
        Return the number of trainable actor parameters.
        """
        return self.actor.parameter_count

    @property
    def critic_parameter_count(self) -> int | None:
        """
        Return trainable critic parameters, or `None` for an actor-only agent.
        """
        return None

    def _critic_values(self, observations: Tensor) -> NDArray[np.float32] | None:
        """
        Score already-converted observations with the critic, if there is one.

        Taking a tensor rather than an array is what lets one conversion serve
        both the actor and the critic on the collection path, which is the
        hottest thing an agent does.

        Whether an agent has a critic is the only way the three of them differ
        at inference time. An actor-only agent answers with this, reporting no
        value at all; `ActorCriticAgent` overrides it with the critic's answer,
        and nothing else about collecting an action changes between them.
        """
        del observations
        return None

    def _gradient_dispersion(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        weights: Tensor,
    ) -> dict[str, float | int | None]:
        """
        Measure the spread of the weight-scaled estimator over equal samples.

        Every algorithm here forms the same actor objective -- log probability
        scaled by some per-transition weight -- and differs only in what that
        weight is: a standardized return for REINFORCE, a standardized
        advantage for the actor-critic pair. Supplying the weight is therefore
        the whole of what a subclass has to contribute.
        """
        detached_weights = weights.detach()

        def subbatch_loss(selected: Tensor) -> Tensor:
            log_probabilities = self.actor.log_probability(
                observations[selected], raw_actions[selected]
            )
            return -(log_probabilities * detached_weights[selected]).mean()

        return gradient_dispersion(
            tuple(self.actor.parameters()),
            subbatch_loss,
            observations.shape[0],
            self.gradient_dispersion_subbatch,
        )

    def _extract_policy_inputs(
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


class ActorCriticAgent(OnPolicyAgent):
    """
    Add a trained value estimate to the shared on-policy contract.

    A2C and PPO differ from REINFORCE in the same way and from each other in a
    different one. Both replace the real return with a bootstrapped estimate,
    which needs a second network, a second optimizer, and a second loss --
    identical in both, and gathered here. What separates them is only how many
    times they reuse a rollout, and that stays in their own files.

    Fields:
        * critic: Trainable state-value estimator.
        * critic_optimizer: Optimizer over critic parameters only.
        * critic_learning_rate: Step size recorded alongside every update.
    """

    critic: CriticNetwork
    critic_optimizer: torch.optim.Optimizer
    critic_learning_rate: float
    critic_config: CriticConfig

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
        *,
        collection_size: int,
        adam_betas: tuple[float, float],
        optimizer_epsilon: float,
        actor_initialization_generator: torch.Generator,
        critic_initialization_generator: torch.Generator,
        sampling_generator: torch.Generator | Sequence[torch.Generator],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        gradient_dispersion_subbatch: int | None = 256,
    ) -> None:
        """
        Build the actor as usual, then the critic and its own optimizer.

        The two networks are initialized from separate generators so that
        neither one's random start depends on the other's shape, which keeps a
        run reproducible when only one of them is reconfigured.
        """
        super().__init__(
            observation_dimensions,
            actor_config,
            collection_size=collection_size,
            adam_betas=adam_betas,
            optimizer_epsilon=optimizer_epsilon,
            actor_initialization_generator=actor_initialization_generator,
            sampling_generator=sampling_generator,
            device=device,
            dtype=dtype,
            gradient_dispersion_subbatch=gradient_dispersion_subbatch,
        )
        if critic_config.learning_rate is None:
            raise ValueError("CriticConfig requires an explicit learning rate.")
        self.critic_config = critic_config
        self.critic_learning_rate = float(critic_config.learning_rate)
        self.critic = CriticNetwork(
            observation_dimensions,
            critic_config,
            critic_initialization_generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.critic_optimizer = self._adam(
            self.critic.parameters(), self.critic_learning_rate
        )

    @property
    def critic_parameter_count(self) -> int:
        """
        Return the number of trainable critic parameters.
        """
        return self.critic.parameter_count

    def _critic_values(self, observations: Tensor) -> NDArray[np.float32]:
        """
        Score already-converted observations with the trained critic.
        """
        with torch.inference_mode():
            values = self.critic(observations)
        return values.cpu().numpy().astype(np.float32, copy=False)

    def _critic_loss(
        self, observations: Tensor, value_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the critic's mean-squared error loss and its predictions.

        Critic's loss is the MSE between its predictions and the given
        value_targets. The targets are computed by combining the standard
        TD(lambda) returns into the GAE advantage estimate.
        """
        predictions = self.critic(observations)
        return 0.5 * (predictions - value_targets.detach()).square().mean(), predictions
