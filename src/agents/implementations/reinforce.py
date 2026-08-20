"""Actor-only Monte Carlo policy-gradient learning for bounded actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from configs import ActorConfig, ReinforceConfig

from ..diagnostics import (
    parameter_norm,
    parameter_update_norm,
    standardize,
)
from ..targets import monte_carlo_return_to_go
from ..types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectionMode,
    CompleteEpisodesInput,
)
from .base import OnPolicyAgent

if TYPE_CHECKING:
    from training.buffers import Trajectory


class ReinforceAgent(OnPolicyAgent):
    """
    Optimize a bounded Gaussian actor with complete-episode Monte Carlo returns.

    The agent owns no critic. It recomputes policy log probabilities from the
    detached collection records, standardizes returns across every trajectory in
    one batch, and applies the documented actor-only REINFORCE loss.

    Fields:
        * collection_mode: Complete-episode collection required by Monte Carlo targets.
        * collection_size: Number of complete trajectories per optimizer update.
        * actor: Trainable bounded Gaussian actor.
        * optimizer: Adam optimizer with explicit MLP-weight regularization.
        * sampling_generators: One isolated policy-sampling stream per worker.
    """

    STATE_VERSION = 5
    collection_mode = CollectionMode.COMPLETE_EPISODES

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        config: ReinforceConfig,
        initialization_generator: torch.Generator,
        sampling_generator: torch.Generator | Sequence[torch.Generator],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        gradient_dispersion_subbatch: int | None = 256,
    ) -> None:
        """
        Construct the actor and its documented Adam optimizer.
        """
        self.config = config
        super().__init__(
            observation_dimensions,
            actor_config,
            # REINFORCE counts its collection in finished episodes, because a
            # Monte Carlo return does not exist until an episode ends.
            collection_size=config.completed_episodes_per_update,
            adam_betas=(config.beta_1, config.beta_2),
            optimizer_epsilon=config.optimizer_epsilon,
            actor_initialization_generator=initialization_generator,
            sampling_generator=sampling_generator,
            device=device,
            dtype=dtype,
            gradient_dispersion_subbatch=gradient_dispersion_subbatch,
        )

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Apply one exact complete-episode REINFORCE optimizer update.
        """
        if not isinstance(update_input, CompleteEpisodesInput):
            raise TypeError("REINFORCE requires complete-episode update input.")
        if len(update_input.episodes) != self.collection_size:
            raise ValueError(
                "REINFORCE updates require exactly "
                f"{self.collection_size} complete trajectories."
            )

        # 1. Compute the returns-to-go G for each trajectory in the batch, then
        # apply standardization to reduce variance.
        returns = tuple(
            monte_carlo_return_to_go(
                episode.transitions, self.config.discount, device=self.device
            )
            for episode in update_input.episodes
        )
        standardized_returns = self._standardize_returns(returns)

        # 2. Compute the REINFORCE loss for each trajectory, as log probs * G_standardize, then
        # average across the batch.
        trajectory_losses = tuple(
            self._trajectory_loss(episode, targets)
            for episode, targets in zip(
                update_input.episodes, standardized_returns, strict=True
            )
        )
        actor_loss = torch.stack(trajectory_losses).mean()

        # [Compute diagnostics for recording training progress.]
        dispersion = self._batch_dispersion(update_input.episodes, standardized_returns)
        entropy_proxy = self._entropy_proxy(update_input.episodes)
        actor_weight_norm = parameter_norm(self.actor.parameters())
        parameters_before = tuple(
            parameter.detach().clone() for parameter in self.actor.parameters()
        )

        # 3. Perform the optimizer step, clipping gradients to avoid exploding updates.
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.gradient_norm_limit
            ).item()
        )
        self.actor_optimizer.step()
        self.actor.project_parameters()

        flattened_returns = torch.cat(returns)
        return AgentUpdateOutput(
            diagnostics={
                "actor_loss": float(actor_loss.detach().item()),
                "actor_gradient_norm": actor_gradient_norm,
                "actor_weight_norm": actor_weight_norm,
                "actor_update_norm": parameter_update_norm(
                    self.actor.parameters(), parameters_before
                ),
                "actor_learning_rate": self.actor_learning_rate,
                "entropy_proxy": entropy_proxy,
                "return_mean": float(flattened_returns.mean().item()),
                "return_standard_deviation": float(
                    flattened_returns.std(unbiased=False).item()
                ),
                "transition_count": int(flattened_returns.numel()),
                "episode_count": len(update_input.episodes),
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
        Return actor, optimizer, and policy-sampling state for exact resume.
        """
        return {
            "state_version": self.STATE_VERSION,
            "actor_config": asdict(self.actor_config),
            "reinforce_config": asdict(self.config),
            "actor_learning_rate": self.actor_learning_rate,
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "sampling_generators": [
                generator.get_state() for generator in self.sampling_generators
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore owned learning state after checking immutable configuration facts.
        """
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("checkpoint has an incompatible REINFORCE state version.")
        if state.get("actor_config") != asdict(self.actor_config):
            raise ValueError("checkpoint actor configuration does not match REINFORCE.")
        if state.get("reinforce_config") != asdict(self.config):
            raise ValueError("checkpoint REINFORCE configuration does not match.")
        if state.get("actor_learning_rate") != self.actor_learning_rate:
            raise ValueError("checkpoint actor learning rate does not match REINFORCE.")
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        generator_states = state["sampling_generators"]
        if len(generator_states) != len(self.sampling_generators):
            raise ValueError("checkpoint sampling-worker count does not match.")
        for generator, generator_state in zip(
            self.sampling_generators, generator_states, strict=True
        ):
            generator.set_state(generator_state)

    def _batch_dispersion(
        self,
        episodes: tuple[Trajectory, ...],
        standardized_returns: tuple[Tensor, ...],
    ) -> dict[str, float | int | None]:
        """
        Flatten the batch, then measure estimator spread the shared way.

        This exists only to record a diagnostic, and all it contributes is the
        weight: the standardized return each transition carries. The measuring
        itself is the same for every algorithm and belongs to the base class.

        Flattening across trajectories is what makes that possible, and it is
        legitimate here because REINFORCE's loss sums per-transition terms and
        divides by a trajectory count, so a sample of transitions estimates the
        same direction up to that constant, which the scale-free summaries
        remove.
        """
        transitions = tuple(
            transition for episode in episodes for transition in episode
        )
        observations, raw_actions = self._extract_policy_inputs(transitions)
        return self._gradient_dispersion(
            observations, raw_actions, torch.cat(standardized_returns)
        )

    def _trajectory_loss(
        self, episode: Trajectory, standardized_returns: Tensor
    ) -> Tensor:
        log_probabilities = self._log_probabilities(episode)
        return -(log_probabilities * standardized_returns.detach()).sum()

    def _log_probabilities(self, episode: Trajectory) -> Tensor:
        observations, raw_actions = self._extract_policy_inputs(episode.transitions)
        return self.actor.log_probability(observations, raw_actions)

    def _standardize_returns(self, returns: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        all_returns = torch.cat(returns)
        standardized = standardize(all_returns, self.config.optimizer_epsilon)
        lengths = tuple(return_.numel() for return_ in returns)
        return tuple(torch.split(standardized, list(lengths)))

    def _entropy_proxy(self, episodes: tuple[Trajectory, ...]) -> float:
        """
        Return the negative mean collection log probability as a dispersion proxy.
        """
        collection_values = tuple(
            row.behaviour_log_probability
            for episode in episodes
            for row in episode.transitions
        )
        if all(value is not None for value in collection_values):
            return float(-np.mean(np.asarray(collection_values, dtype=np.float64)))
        return float(
            -torch.cat(tuple(self._log_probabilities(episode) for episode in episodes))
            .detach()
            .mean()
            .item()
        )
