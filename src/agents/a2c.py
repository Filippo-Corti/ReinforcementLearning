"""Synchronous advantage actor-critic learning with detached GAE targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from configs import A2CConfig, ActorConfig, CriticConfig
from models import ActorNetwork, CriticNetwork, agent_parameter_counts
from training.buffers import (
    GAETargets,
    OnPolicyRollout,
    VectorOnPolicyRollout,
    compute_gae_targets,
    compute_vector_gae_targets,
)

from .diagnostics import (
    explained_variance,
    gradient_dispersion,
    parameter_norm,
    parameter_update_norm,
    standardize,
)
from .types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectedActionBatch,
    CollectionMode,
)


class A2CAgent:
    """
    Optimize a bounded Gaussian actor and a fixed-capacity value critic.

    One fixed rollout supplies detached GAE advantages for the actor and
    detached value targets for the critic. Separate Adam optimizers preserve
    the required actor/critic gradient boundary.

    Fields:
        * collection_mode: Fixed-rollout collection required by bootstrapped targets.
        * collection_size: Maximum transitions collected before one update.
        * actor: Trainable bounded Gaussian policy.
        * critic: Trainable state-value estimator.
        * actor_optimizer: Adam optimizer over actor parameters only.
        * critic_optimizer: Adam optimizer over critic parameters only.
        * sampling_generator: Isolated generator used only for policy sampling.
    """

    STATE_VERSION = 3
    collection_mode = CollectionMode.FIXED_ROLLOUT

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
        config: A2CConfig,
        critic_learning_rate: float,
        actor_initialization_generator: torch.Generator,
        critic_initialization_generator: torch.Generator,
        sampling_generator: torch.Generator | Sequence[torch.Generator],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        gradient_dispersion_subbatch: int | None = 256,
    ) -> None:
        """
        Construct the actor, critic, and their separate documented optimizers.
        """
        self.gradient_dispersion_subbatch = gradient_dispersion_subbatch
        self.config = config
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.collection_size = config.transitions_per_rollout
        self.device = torch.device(device)
        self.dtype = dtype
        if actor_config.learning_rate is None:
            raise ValueError("ActorConfig requires an explicit learning rate.")
        self.actor_learning_rate = float(actor_config.learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.actor = ActorNetwork(
            observation_dimensions,
            actor_config,
            actor_initialization_generator,
            device=self.device,
            dtype=dtype,
        )
        self.critic = CriticNetwork(
            observation_dimensions,
            critic_config,
            critic_initialization_generator,
            device=self.device,
            dtype=dtype,
        )
        optimizer_arguments = {
            "betas": (config.beta_1, config.beta_2),
            "eps": config.optimizer_epsilon,
        }
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate, **optimizer_arguments
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            **optimizer_arguments,
        )
        self.sampling_generators = self._sampling_generators(sampling_generator)
        self.sampling_generator = self.sampling_generators[0]

    def collect_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> CollectedAction:
        """
        Sample one detached action and retain the detached current critic value.
        """
        observation = self._observation_tensor(normalized_observation).unsqueeze(0)
        sample = self.actor.sample(observation, self.sampling_generator)
        with torch.inference_mode():
            current_value = self.critic(observation)[0]
        return CollectedAction(
            raw_action=sample.raw_action[0].cpu().numpy(),
            env_action=sample.env_action[0].cpu().numpy(),
            behaviour_log_probability=float(sample.log_probability[0].item()),
            current_value=float(current_value.item()),
        )

    def deterministic_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Return the actor mean action without advancing the sampling stream.
        """
        observation = self._observation_tensor(normalized_observation).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor.deterministic_action(observation)[0]
        return action.cpu().numpy().astype(np.float32, copy=False)

    def collect_actions(
        self,
        normalized_observations: NDArray[np.float32],
        environment_indices: Sequence[int] | None = None,
    ) -> CollectedActionBatch:
        """
        Sample one action and critic value per independent environment row.
        """
        observations = self._observation_tensor(normalized_observations)
        indices = self._environment_indices(observations.shape[0], environment_indices)
        sample = self.actor.sample_with_generators(
            observations,
            tuple(self.sampling_generators[index] for index in indices),
        )
        with torch.inference_mode():
            current_values = self.critic(observations)
        return CollectedActionBatch(
            raw_actions=sample.raw_action.cpu().numpy(),
            env_actions=sample.env_action.cpu().numpy(),
            behaviour_log_probabilities=sample.log_probability.cpu().numpy(),
            current_values=current_values.cpu().numpy(),
        )

    def bootstrap_value(self, normalized_observation: NDArray[np.float32]) -> float:
        """
        Return a detached critic estimate for a non-terminal bootstrap state.
        """
        observation = self._observation_tensor(normalized_observation).unsqueeze(0)
        with torch.inference_mode():
            value = self.critic(observation)[0]
        return float(value.item())

    def bootstrap_values(
        self, normalized_observations: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Return detached critic estimates for batched non-terminal next states.
        """
        observations = self._observation_tensor(normalized_observations)
        with torch.inference_mode():
            values = self.critic(observations)
        return values.cpu().numpy().astype(np.float32, copy=False)

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Apply one actor update and one critic update from a fixed rollout.
        """
        if update_input.mode is not CollectionMode.FIXED_ROLLOUT:
            raise ValueError("A2C requires fixed-rollout update input.")
        rollout = update_input.rollout
        if rollout is None:
            raise ValueError("A2C fixed-rollout input lacks a rollout.")
        if len(rollout.transitions) > self.collection_size:
            raise ValueError("A2C rollout exceeds its configured collection size.")

        observations, raw_actions, behaviour_log_probabilities, targets = (
            self._rollout_training_tensors(rollout)
        )
        advantages = self._standardize_advantages(targets.raw_advantages)
        dispersion = self._gradient_dispersion(observations, raw_actions, advantages)
        actor_loss = self._actor_loss_tensors(observations, raw_actions, advantages)
        critic_loss, predictions = self._critic_loss(
            observations, targets.value_targets
        )
        actor_weight_norm = parameter_norm(self.actor.parameters())
        critic_weight_norm = parameter_norm(self.critic.parameters())
        actor_parameters_before = tuple(
            parameter.detach().clone() for parameter in self.actor.parameters()
        )
        critic_parameters_before = tuple(
            parameter.detach().clone() for parameter in self.critic.parameters()
        )

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.gradient_norm_limit
            ).item()
        )
        self.actor_optimizer.step()
        self.actor.project_parameters()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.gradient_norm_limit
            ).item()
        )
        self.critic_optimizer.step()

        return AgentUpdateOutput(
            diagnostics={
                "actor_loss": float(actor_loss.detach().item()),
                "critic_loss": float(critic_loss.detach().item()),
                "actor_gradient_norm": actor_gradient_norm,
                "critic_gradient_norm": critic_gradient_norm,
                "actor_weight_norm": actor_weight_norm,
                "critic_weight_norm": critic_weight_norm,
                "actor_update_norm": parameter_update_norm(
                    self.actor.parameters(), actor_parameters_before
                ),
                "critic_update_norm": parameter_update_norm(
                    self.critic.parameters(), critic_parameters_before
                ),
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate,
                "entropy_proxy": float(-behaviour_log_probabilities.mean().item()),
                "explained_variance": explained_variance(
                    targets.value_targets, predictions
                ),
                "advantage_mean": float(targets.raw_advantages.mean().item()),
                "advantage_standard_deviation": float(
                    targets.raw_advantages.std(unbiased=False).item()
                ),
                "standardized_advantage_mean": float(advantages.mean().item()),
                "standardized_advantage_standard_deviation": float(
                    advantages.std(unbiased=False).item()
                ),
                "value_target_mean": float(targets.value_targets.mean().item()),
                "value_target_standard_deviation": float(
                    targets.value_targets.std(unbiased=False).item()
                ),
                "value_prediction_mean": float(predictions.detach().mean().item()),
                "value_prediction_standard_deviation": float(
                    predictions.detach().std(unbiased=False).item()
                ),
                "temporal_difference_error_mean": float(
                    targets.temporal_difference_errors.mean().item()
                ),
                "transition_count": len(rollout.transitions),
                "log_standard_deviation_0": float(
                    self.actor.policy.log_standard_deviation[0].detach().item()
                ),
                "log_standard_deviation_1": float(
                    self.actor.policy.log_standard_deviation[1].detach().item()
                ),
                **dispersion,
            }
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Return models, optimizers, and sampling state for exact resume.
        """
        return {
            "state_version": self.STATE_VERSION,
            "actor_config": asdict(self.actor_config),
            "critic_config": asdict(self.critic_config),
            "a2c_config": asdict(self.config),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "sampling_generators": [
                generator.get_state() for generator in self.sampling_generators
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore owned learning state after checking immutable configuration facts.
        """
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("checkpoint has an incompatible A2C state version.")
        if state.get("actor_config") != asdict(self.actor_config):
            raise ValueError("checkpoint actor configuration does not match A2C.")
        if state.get("critic_config") != asdict(self.critic_config):
            raise ValueError("checkpoint critic configuration does not match A2C.")
        if state.get("a2c_config") != asdict(self.config):
            raise ValueError("checkpoint A2C configuration does not match.")
        if state.get("actor_learning_rate") != self.actor_learning_rate:
            raise ValueError("checkpoint actor learning rate does not match A2C.")
        if state.get("critic_learning_rate") != self.critic_learning_rate:
            raise ValueError("checkpoint critic learning rate does not match A2C.")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        generator_states = state["sampling_generators"]
        if len(generator_states) != len(self.sampling_generators):
            raise ValueError("checkpoint sampling-worker count does not match.")
        for generator, generator_state in zip(
            self.sampling_generators, generator_states, strict=True
        ):
            generator.set_state(generator_state)

    @property
    def actor_parameter_count(self) -> int:
        """
        Return the number of trainable actor parameters, including dispersion.
        """
        return agent_parameter_counts(self.actor, self.critic).actor

    @property
    def critic_parameter_count(self) -> int:
        """
        Return the number of trainable fixed-architecture critic parameters.
        """
        count = agent_parameter_counts(self.actor, self.critic).critic
        if count is None:
            raise RuntimeError("A2C must own a critic.")
        return count

    def _gradient_dispersion(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        advantages: Tensor,
    ) -> dict[str, float | int | None]:
        """
        Measure the spread of the advantage-weighted estimator over equal samples.
        """
        weights = advantages.detach()

        def subbatch_loss(selected: Tensor) -> Tensor:
            log_probabilities = self.actor.log_probability(
                observations[selected], raw_actions[selected]
            )
            return -(log_probabilities * weights[selected]).mean()

        return gradient_dispersion(
            tuple(self.actor.parameters()),
            subbatch_loss,
            observations.shape[0],
            self.gradient_dispersion_subbatch,
        )

    def _actor_loss(self, rollout: OnPolicyRollout, advantages: Tensor) -> Tensor:
        tensors = rollout.tensors(device=self.device)
        return self._actor_loss_tensors(
            tensors.observations.to(dtype=self.dtype),
            tensors.raw_actions.to(dtype=self.dtype),
            advantages,
        )

    def _actor_loss_tensors(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        log_probabilities = self.actor.log_probability(observations, raw_actions)
        return -(log_probabilities * advantages.detach()).mean()

    def _critic_loss(
        self, observations: Tensor, value_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        predictions = self.critic(observations)
        return 0.5 * (predictions - value_targets.detach()).square().mean(), predictions

    def _standardize_advantages(self, advantages: Tensor) -> Tensor:
        return standardize(advantages, self.config.optimizer_epsilon)

    def _entropy_proxy(self, rollout: OnPolicyRollout) -> float:
        collection_values = tuple(
            row.behaviour_log_probability for row in rollout.transitions
        )
        if all(value is not None for value in collection_values):
            return float(-np.mean(np.asarray(collection_values, dtype=np.float64)))
        tensors = rollout.tensors(device=self.device)
        return float(
            -self.actor.log_probability(
                tensors.observations.to(dtype=self.dtype),
                tensors.raw_actions.to(dtype=self.dtype),
            )
            .detach()
            .mean()
            .item()
        )

    def _observation_tensor(self, observation: NDArray[np.float32]) -> Tensor:
        return torch.as_tensor(observation, dtype=self.dtype, device=self.device)

    def _rollout_training_tensors(
        self,
        rollout: OnPolicyRollout | VectorOnPolicyRollout,
    ) -> tuple[Tensor, Tensor, Tensor, GAETargets]:
        if isinstance(rollout, VectorOnPolicyRollout):
            tensors = rollout.tensors(device=self.device)
            vector_targets = compute_vector_gae_targets(
                rollout,
                self.config.discount,
                self.config.gae_lambda,
                device=self.device,
            )
            targets = GAETargets(
                temporal_difference_errors=tensors.flatten_valid(
                    vector_targets.temporal_difference_errors
                ),
                raw_advantages=tensors.flatten_valid(vector_targets.raw_advantages),
                value_targets=tensors.flatten_valid(vector_targets.value_targets),
            )
            observations = tensors.flatten_valid(tensors.observations).to(
                dtype=self.dtype
            )
            raw_actions = tensors.flatten_valid(tensors.raw_actions).to(
                dtype=self.dtype
            )
            probabilities = tensors.behaviour_log_probabilities
            if probabilities is None:
                flattened_probabilities = self.actor.log_probability(
                    observations, raw_actions
                ).detach()
            else:
                flattened_probabilities = tensors.flatten_valid(probabilities)
            return observations, raw_actions, flattened_probabilities, targets
        tensors = rollout.tensors(device=self.device)
        targets = compute_gae_targets(
            rollout,
            self.config.discount,
            self.config.gae_lambda,
            device=self.device,
        )
        observations = tensors.observations.to(dtype=self.dtype)
        raw_actions = tensors.raw_actions.to(dtype=self.dtype)
        probabilities = tensors.behaviour_log_probabilities
        if probabilities is None:
            probabilities = self.actor.log_probability(
                observations, raw_actions
            ).detach()
        return observations, raw_actions, probabilities, targets

    @staticmethod
    def _sampling_generators(
        generators: torch.Generator | Sequence[torch.Generator],
    ) -> tuple[torch.Generator, ...]:
        if isinstance(generators, torch.Generator):
            return (generators,)
        values = tuple(generators)
        if not values:
            raise ValueError("At least one policy-sampling generator is required.")
        return values

    def _environment_indices(
        self,
        row_count: int,
        environment_indices: Sequence[int] | None,
    ) -> tuple[int, ...]:
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
