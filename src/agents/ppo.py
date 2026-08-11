"""Clipped proximal policy optimization with fixed GAE targets."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from configs import ActorConfig, CriticConfig, PPOConfig
from models import ActorNetwork, CriticNetwork, agent_parameter_counts
from training.buffers import compute_gae_targets

from .diagnostics import (
    explained_variance,
    parameter_norm,
    parameter_update_norm,
    standardize,
)
from .types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectionMode,
)


class PPOAgent:
    """
    Optimize a bounded Gaussian actor with clipped multi-epoch sample reuse.

    A collected rollout fixes its behaviour log probabilities, standardized GAE
    advantages, and critic targets before every seeded minibatch epoch begins.
    The actor and critic retain separate optimizers and gradient clipping.

    Fields:
        * collection_mode: Fixed-rollout collection required by bootstrapped targets.
        * collection_size: Maximum transitions collected before one PPO update.
        * actor: Trainable bounded Gaussian policy.
        * critic: Trainable state-value estimator.
        * actor_optimizer: Adam optimizer over actor parameters only.
        * critic_optimizer: Adam optimizer over critic parameters only.
        * sampling_generator: Isolated generator used only for policy sampling.
        * optimization_generator: Isolated generator used only for minibatch order.
    """

    STATE_VERSION = 1
    collection_mode = CollectionMode.FIXED_ROLLOUT

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
        config: PPOConfig,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_initialization_generator: torch.Generator,
        critic_initialization_generator: torch.Generator,
        sampling_generator: torch.Generator,
        optimization_generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct the PPO models, optimizers, and isolated random generators.
        """
        self._validate_config(config)
        self.config = config
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.collection_size = config.transitions_per_rollout
        self.device = torch.device(device)
        self.dtype = dtype
        self.actor_learning_rate = float(actor_learning_rate)
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
        self.sampling_generator = sampling_generator
        self.optimization_generator = optimization_generator
        self._last_minibatch_indices: tuple[tuple[tuple[int, ...], ...], ...] = ()

    def collect_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> CollectedAction:
        """
        Sample one action and retain fixed behaviour-policy quantities.
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
        Return the actor mean action without advancing a training stream.
        """
        observation = self._observation_tensor(normalized_observation).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor.deterministic_action(observation)[0]
        return action.cpu().numpy().astype(np.float32, copy=False)

    def bootstrap_value(self, normalized_observation: NDArray[np.float32]) -> float:
        """
        Return a detached critic estimate for a non-terminal bootstrap state.
        """
        observation = self._observation_tensor(normalized_observation).unsqueeze(0)
        with torch.inference_mode():
            value = self.critic(observation)[0]
        return float(value.item())

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Optimize one fixed rollout through every seeded PPO minibatch epoch.
        """
        if update_input.mode is not CollectionMode.FIXED_ROLLOUT:
            raise ValueError("PPO requires fixed-rollout update input.")
        rollout = update_input.rollout
        if rollout is None:
            raise ValueError("PPO fixed-rollout input lacks a rollout.")
        if len(rollout.transitions) > self.collection_size:
            raise ValueError("PPO rollout exceeds its configured collection size.")

        tensors = rollout.tensors(device=self.device)
        if tensors.behaviour_log_probabilities is None:
            raise ValueError("PPO requires a behaviour log probability on every row.")
        targets = compute_gae_targets(
            rollout,
            self.config.discount,
            self.config.gae_lambda,
            device=self.device,
        )
        observations = tensors.observations.to(dtype=self.dtype)
        raw_actions = tensors.raw_actions.to(dtype=self.dtype)
        old_log_probabilities = tensors.behaviour_log_probabilities.detach().clone()
        advantages = self._standardize_advantages(targets.raw_advantages)
        value_targets = targets.value_targets.detach().clone()
        actor_weight_norm = parameter_norm(self.actor.parameters())
        critic_weight_norm = parameter_norm(self.critic.parameters())
        actor_parameters_before = tuple(
            parameter.detach().clone() for parameter in self.actor.parameters()
        )
        critic_parameters_before = tuple(
            parameter.detach().clone() for parameter in self.critic.parameters()
        )
        metrics: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "actor_gradient_norm": [],
            "critic_gradient_norm": [],
            "approximate_kl": [],
            "clip_fraction": [],
            "ratio_mean": [],
            "ratio_second_moment": [],
            "ratio_minimum": [],
            "ratio_maximum": [],
        }
        minibatch_sizes: list[int] = []
        minibatch_orders: list[tuple[tuple[int, ...], ...]] = []

        for _ in range(self.config.optimization_epochs):
            permutation = torch.randperm(
                len(rollout.transitions),
                generator=self.optimization_generator,
                device=self.device,
            )
            epoch_batches: list[tuple[int, ...]] = []
            for start in range(0, len(rollout.transitions), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                epoch_batches.append(tuple(int(index) for index in indices.tolist()))
                minibatch_metrics = self._update_minibatch(
                    observations[indices],
                    raw_actions[indices],
                    old_log_probabilities[indices],
                    advantages[indices],
                    value_targets[indices],
                )
                for name, value in minibatch_metrics.items():
                    metrics[name].append(value)
                minibatch_sizes.append(len(indices))
            minibatch_orders.append(tuple(epoch_batches))

        self._last_minibatch_indices = tuple(minibatch_orders)
        with torch.inference_mode():
            final_predictions = self.critic(observations)
        weighted_mean = lambda name: float(
            np.average(metrics[name], weights=minibatch_sizes)
        )
        ratio_mean = weighted_mean("ratio_mean")
        ratio_standard_deviation = float(
            np.sqrt(max(weighted_mean("ratio_second_moment") - ratio_mean**2, 0.0))
        )

        return AgentUpdateOutput(
            diagnostics={
                "actor_loss": weighted_mean("actor_loss"),
                "critic_loss": weighted_mean("critic_loss"),
                "actor_gradient_norm": weighted_mean("actor_gradient_norm"),
                "critic_gradient_norm": weighted_mean("critic_gradient_norm"),
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
                "entropy_proxy": float(-old_log_probabilities.mean().item()),
                "explained_variance": explained_variance(
                    value_targets, final_predictions
                ),
                "approximate_kl": weighted_mean("approximate_kl"),
                "clip_fraction": weighted_mean("clip_fraction"),
                "ratio_mean": ratio_mean,
                "ratio_standard_deviation": ratio_standard_deviation,
                "ratio_minimum": float(np.min(metrics["ratio_minimum"])),
                "ratio_maximum": float(np.max(metrics["ratio_maximum"])),
                "advantage_mean": float(targets.raw_advantages.mean().item()),
                "advantage_standard_deviation": float(
                    targets.raw_advantages.std(unbiased=False).item()
                ),
                "standardized_advantage_mean": float(advantages.mean().item()),
                "standardized_advantage_standard_deviation": float(
                    advantages.std(unbiased=False).item()
                ),
                "value_target_mean": float(value_targets.mean().item()),
                "value_target_standard_deviation": float(
                    value_targets.std(unbiased=False).item()
                ),
                "value_prediction_mean": float(final_predictions.mean().item()),
                "value_prediction_standard_deviation": float(
                    final_predictions.std(unbiased=False).item()
                ),
                "temporal_difference_error_mean": float(
                    targets.temporal_difference_errors.mean().item()
                ),
                "transition_count": len(rollout.transitions),
                "optimization_epochs": self.config.optimization_epochs,
                "minibatch_count": len(metrics["actor_loss"]),
                "log_standard_deviation_0": float(
                    self.actor.policy.log_standard_deviation[0].detach().item()
                ),
                "log_standard_deviation_1": float(
                    self.actor.policy.log_standard_deviation[1].detach().item()
                ),
            }
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Return models, optimizers, and generator states for exact resume.
        """
        return {
            "state_version": self.STATE_VERSION,
            "actor_config": asdict(self.actor_config),
            "critic_config": asdict(self.critic_config),
            "ppo_config": asdict(self.config),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "sampling_generator": self.sampling_generator.get_state(),
            "optimization_generator": self.optimization_generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore owned learning state after checking immutable configuration facts.
        """
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("checkpoint has an incompatible PPO state version.")
        if state.get("actor_config") != asdict(self.actor_config):
            raise ValueError("checkpoint actor configuration does not match PPO.")
        if state.get("critic_config") != asdict(self.critic_config):
            raise ValueError("checkpoint critic configuration does not match PPO.")
        if state.get("ppo_config") != asdict(self.config):
            raise ValueError("checkpoint PPO configuration does not match.")
        if state.get("actor_learning_rate") != self.actor_learning_rate:
            raise ValueError("checkpoint actor learning rate does not match PPO.")
        if state.get("critic_learning_rate") != self.critic_learning_rate:
            raise ValueError("checkpoint critic learning rate does not match PPO.")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.sampling_generator.set_state(state["sampling_generator"])
        self.optimization_generator.set_state(state["optimization_generator"])

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
            raise RuntimeError("PPO must own a critic.")
        return count

    @property
    def last_minibatch_indices(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """
        Return the immutable row batches used by the most recent PPO update.
        """
        return self._last_minibatch_indices

    def _update_minibatch(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
        value_targets: Tensor,
    ) -> dict[str, float]:
        actor_loss, ratios, log_ratios = self._actor_loss(
            observations,
            raw_actions,
            old_log_probabilities,
            advantages,
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

        critic_loss, _ = self._critic_loss(observations, value_targets)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.gradient_norm_limit
            ).item()
        )
        self.critic_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "actor_gradient_norm": actor_gradient_norm,
            "critic_gradient_norm": critic_gradient_norm,
            "approximate_kl": float(
                ((ratios - 1.0) - log_ratios).detach().mean().item()
            ),
            "clip_fraction": float(
                (ratios.sub(1.0).abs() > self.config.clip_epsilon)
                .detach()
                .float()
                .mean()
                .item()
            ),
            "ratio_mean": float(ratios.detach().mean().item()),
            "ratio_second_moment": float(ratios.detach().square().mean().item()),
            "ratio_minimum": float(ratios.detach().min().item()),
            "ratio_maximum": float(ratios.detach().max().item()),
        }

    def _actor_loss(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        current_log_probabilities = self.actor.log_probability(
            observations, raw_actions
        )
        log_ratios = current_log_probabilities - old_log_probabilities.detach()
        ratios = log_ratios.exp()
        unclipped_surrogate = ratios * advantages.detach()
        clipped_surrogate = (
            ratios.clamp(1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
            * advantages.detach()
        )
        return (
            -torch.minimum(unclipped_surrogate, clipped_surrogate).mean(),
            ratios,
            log_ratios,
        )

    def _critic_loss(
        self, observations: Tensor, value_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        predictions = self.critic(observations)
        return 0.5 * (predictions - value_targets.detach()).square().mean(), predictions

    def _standardize_advantages(self, advantages: Tensor) -> Tensor:
        return standardize(advantages, self.config.optimizer_epsilon)

    def _observation_tensor(self, observation: NDArray[np.float32]) -> Tensor:
        return torch.as_tensor(observation, dtype=self.dtype, device=self.device)

    @staticmethod
    def _validate_config(config: PPOConfig) -> None:
        if config.value_clipping_enabled:
            raise ValueError("PPO value clipping is disabled by the learning contract.")
        if config.kl_early_stop_enabled:
            raise ValueError(
                "PPO KL early stopping is disabled by the learning contract."
            )
        if config.entropy_bonus_enabled:
            raise ValueError(
                "PPO entropy bonuses are disabled by the learning contract."
            )
        if config.weight_decay_enabled:
            raise ValueError("PPO weight decay is disabled by the learning contract.")
        if config.learning_rate_scheduler_enabled:
            raise ValueError(
                "PPO learning-rate scheduling is disabled by the learning contract."
            )
