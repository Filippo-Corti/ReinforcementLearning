"""Synchronous advantage actor-critic learning with detached GAE targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from configs import A2CConfig, ActorConfig, CriticConfig
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

if TYPE_CHECKING:
    from training.buffers import TrainingTransition


class A2CAgent(ActorCriticAgent):
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
        * sampling_generators: One isolated policy-sampling stream per worker.
    """

    STATE_VERSION = 4
    collection_mode = CollectionMode.FIXED_ROLLOUT

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
        config: A2CConfig,
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

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Apply one actor update and one critic update from a fixed rollout.
        """
        if not isinstance(update_input, FixedRolloutInput):
            raise TypeError("A2C requires fixed-rollout update input.")
        rollout = update_input.rollout
        if rollout.transition_count > self.collection_size:
            raise ValueError("A2C rollout exceeds its configured collection size.")

        # 1. Compute the GAE advantages A and the critic targets, then
        # standardize the advantages to reduce variance in the actor step.
        transitions = rollout.transitions
        observations, raw_actions = self._extract_policy_inputs(transitions)
        targets = compute_vector_gae_targets(
            rollout, self.config.discount, self.config.gae_lambda, device=self.device
        )
        advantages = self._standardize_advantages(targets.raw_advantages)

        # 2. Compute both losses:
        # * The actor's loss is log probs * A_standardized.
        # * The critic's loss is MSE between predictions and targets.
        actor_loss = self._actor_loss(observations, raw_actions, advantages)
        critic_loss, predictions = self._critic_loss(
            observations, targets.value_targets
        )

        # [Compute diagnostics for recording training progress.]
        behaviour_log_probabilities = self._behaviour_log_probabilities(
            transitions, observations, raw_actions
        )
        dispersion = self._gradient_dispersion(observations, raw_actions, advantages)
        actor_weight_norm = parameter_norm(self.actor.parameters())
        critic_weight_norm = parameter_norm(self.critic.parameters())
        actor_parameters_before = tuple(
            parameter.detach().clone() for parameter in self.actor.parameters()
        )
        critic_parameters_before = tuple(
            parameter.detach().clone() for parameter in self.critic.parameters()
        )

        # 3. Perform the actor step, clipping gradients to avoid exploding
        # updates.
        # Both optimizers are cleared each time so that neither loss
        # can leave a gradient behind in the other network.
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

        # 4. Perform the critic step, separately, for the same reason.
        # Again, gradients are clipped to avoid exploding updates.
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

    def _actor_loss(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """
        Score the actions taken by how much better than average they turned out.

        A2C reads every transition once, under the policy that chose it, so
        there is no correction to make and nothing to report beyond the loss
        itself. PPO returns more from the same-named method because its
        objective genuinely produces more: an importance ratio that only exists
        once a rollout is reused.
        """
        log_probabilities = self.actor.log_probability(observations, raw_actions)
        return -(log_probabilities * advantages.detach()).mean()

    def _behaviour_log_probabilities(
        self,
        transitions: Sequence[TrainingTransition],
        observations: Tensor,
        raw_actions: Tensor,
    ) -> Tensor:
        """
        Return the log probabilities the collecting policy assigned each action.

        A2C looks at every transition once, so when the collector stored none
        the current actor is still the one that chose them and recomputing is
        exact rather than an approximation.
        """
        stored = optional_tensor(
            [transition.behaviour_log_probability for transition in transitions],
            device=self.device,
        )
        if stored is not None:
            return stored
        return self.actor.log_probability(observations, raw_actions).detach()
