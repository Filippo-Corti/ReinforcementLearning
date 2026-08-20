"""Clipped proximal policy optimization with fixed GAE targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from configs import ActorConfig, CriticConfig, PPOConfig
from utils.vectors import optional_tensor

from ..diagnostics import (
    explained_variance,
    parameter_norm,
    parameter_update_norm,
    standardize,
)
from ..targets import compute_vector_gae_targets
from ..types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectionMode,
    FixedRolloutInput,
)
from .base import ActorCriticAgent


@dataclass(frozen=True, slots=True)
class ClippedActorLoss:
    """
    Hold the clipped objective together with the importance ratios that produced it.

    The importance ratios come back with the loss because they are computed on the way to
    it and are needed afterwards: six of the recorded diagnostics are functions
    of them, and one of those, the approximate KL, is what decides whether the
    remaining epochs run at all. Recomputing them outside would mean a second
    forward pass through the actor on every minibatch.

    Fields:
        * loss: Negated clipped surrogate, the quantity actually minimized.
        * importance_ratios: Probability of each action now over its probability when chosen.
        * log_ratios: The same comparison in log space, kept for a stabler KL.
    """

    loss: Tensor
    importance_ratios: Tensor
    log_ratios: Tensor


class PPOAgent(ActorCriticAgent):
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
        * sampling_generators: One isolated policy-sampling stream per worker.
        * optimization_generator: Isolated generator used only for minibatch order.
    """

    STATE_VERSION = 4
    collection_mode = CollectionMode.FIXED_ROLLOUT

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
        config: PPOConfig,
        actor_initialization_generator: torch.Generator,
        critic_initialization_generator: torch.Generator,
        sampling_generator: torch.Generator | Sequence[torch.Generator],
        optimization_generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        gradient_dispersion_subbatch: int | None = 256,
    ) -> None:
        """
        Construct the PPO models, optimizers, and isolated random generators.
        """
        self._validate_config(config)
        self.config = config
        super().__init__(
            observation_dimensions,
            actor_config,
            critic_config,
            # An actor-critic bootstraps rather than waiting for an episode to
            # end, so it counts its collection in transitions.
            collection_size=config.transitions_per_rollout,
            adam_betas=(config.beta_1, config.beta_2),
            optimizer_epsilon=config.optimizer_epsilon,
            actor_initialization_generator=actor_initialization_generator,
            critic_initialization_generator=critic_initialization_generator,
            sampling_generator=sampling_generator,
            device=device,
            dtype=dtype,
            gradient_dispersion_subbatch=gradient_dispersion_subbatch,
        )
        self.optimization_generator = optimization_generator
        self._last_minibatch_indices: tuple[tuple[tuple[int, ...], ...], ...] = ()

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Optimize one fixed rollout through every seeded PPO minibatch epoch.
        """
        if not isinstance(update_input, FixedRolloutInput):
            raise TypeError("PPO requires fixed-rollout update input.")
        rollout = update_input.rollout
        if rollout.transition_count > self.collection_size:
            raise ValueError("PPO rollout exceeds its configured collection size.")

        # 1. Compute the GAE advantages A and the critic targets, then
        # standardize the advantages to reduce variance in the actor step.
        transitions = rollout.transitions
        observations, raw_actions = self._extract_policy_inputs(transitions)
        targets = compute_vector_gae_targets(
            rollout, self.config.discount, self.config.gae_lambda, device=self.device
        )

        # 1.5 Unlike A2C, in PPO we immediately compute the log probabilities
        # of the collected actions under the policy that produced them.
        stored_log_probabilities = optional_tensor(
            [transition.behaviour_log_probability for transition in transitions],
            device=self.device,
        )
        if stored_log_probabilities is None:
            raise ValueError("PPO requires collection log probabilities.")
        old_log_probabilities = stored_log_probabilities.detach().clone()
        advantages = self._standardize_advantages(targets.raw_advantages)
        value_targets = targets.value_targets.detach().clone()

        # [Compute diagnostics for recording training progress.]
        dispersion = self._gradient_dispersion(observations, raw_actions, advantages)
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

        # 2. Reuse the rollout for several epochs of shuffled minibatches:
        completed_epochs = 0
        for _ in range(self.config.optimization_epochs):

            # Extract minibatches by shuffling the rollout indices and slicing them into
            # contiguous chunks. This way, each epoch has a different random order.
            permutation = torch.randperm(
                len(rollout.transitions),
                generator=self.optimization_generator,
                device=self.device,
            )
            epoch_batches: list[tuple[int, ...]] = []
            epoch_kl: list[float] = []

            # For each of the minibatches:
            for start in range(0, len(rollout.transitions), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                epoch_batches.append(tuple(int(index) for index in indices.tolist()))

                # 3-4. Update actor and critic, just like A2C but with PPO's clipped surrogate loss.
                minibatch_metrics = self._update_minibatch(
                    observations[indices],
                    raw_actions[indices],
                    old_log_probabilities[indices],
                    advantages[indices],
                    value_targets[indices],
                )

                # [Record metrics for this minibatch.]
                for name, value in minibatch_metrics.items():
                    metrics[name].append(value)
                epoch_kl.append(minibatch_metrics["approximate_kl"])
                minibatch_sizes.append(len(indices))
            minibatch_orders.append(tuple(epoch_batches))
            completed_epochs += 1

            # 5. Stop early if the policy has already moved too far from the
            # one that collected this rollout.
            if (
                self.config.kl_early_stop_enabled
                and float(np.mean(epoch_kl)) > self.config.target_kl
            ):
                break

        # [Compute final diagnostics, after all epochs are done.]
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
                "completed_epochs": float(completed_epochs),
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
                **dispersion,
            }
        )

    def _standardize_advantages(self, advantages: Tensor) -> Tensor:
        """
        Rescale advantages to zero mean and unit variance for the actor only.

        The critic keeps the unstandardized targets: the shift and scale are a
        step-size convenience for the policy gradient, not a change to what the
        value function is supposed to predict.
        """
        return standardize(advantages, self.config.optimizer_epsilon)

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
            "sampling_generators": [
                generator.get_state() for generator in self.sampling_generators
            ],
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
        generator_states = state["sampling_generators"]
        if len(generator_states) != len(self.sampling_generators):
            raise ValueError("checkpoint sampling-worker count does not match.")
        for generator, generator_state in zip(
            self.sampling_generators, generator_states, strict=True
        ):
            generator.set_state(generator_state)
        self.optimization_generator.set_state(state["optimization_generator"])

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
        actor = self._actor_loss(
            observations,
            raw_actions,
            old_log_probabilities,
            advantages,
        )
        actor_loss, ratios, log_ratios = (
            actor.loss,
            actor.importance_ratios,
            actor.log_ratios,
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
    ) -> ClippedActorLoss:
        """
        Bound how far one step may move the policy from the one that collected.

        Clipping is the whole of PPO: an action whose probability has already
        grown past the trust region stops contributing gradient, so reusing a
        rollout for several epochs cannot run away from the behaviour policy.

        The extra parameters (ratio and log_ratio) are returned to compute
        the KL indicator, used to detect early stopping.
        """
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
        return ClippedActorLoss(
            loss=-torch.minimum(unclipped_surrogate, clipped_surrogate).mean(),
            importance_ratios=ratios,
            log_ratios=log_ratios,
        )

    @staticmethod
    def _validate_config(config: PPOConfig) -> None:
        if config.value_clipping_enabled:
            raise ValueError("PPO value clipping is disabled by the learning contract.")
        if config.kl_early_stop_enabled and config.target_kl <= 0:
            raise ValueError("PPO KL early stopping requires a positive target.")
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
