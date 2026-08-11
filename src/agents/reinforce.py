"""Actor-only Monte Carlo policy-gradient learning for bounded actions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from configs import ActorConfig, ReinforceConfig
from models import ActorNetwork, agent_parameter_counts
from training.buffers import OnPolicyRollout, monte_carlo_return_to_go

from .diagnostics import parameter_norm, parameter_update_norm, standardize
from .types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectionMode,
)


class ReinforceAgent:
    """
    Optimize a bounded Gaussian actor with complete-episode Monte Carlo returns.

    The agent owns no critic. It recomputes policy log probabilities from the
    detached collection records, standardizes returns across every trajectory in
    one batch, and applies the documented actor-only REINFORCE loss.

    Fields:
        * collection_mode: Complete-episode collection required by Monte Carlo targets.
        * collection_size: Number of complete trajectories per optimizer update.
        * actor: Trainable bounded Gaussian actor.
        * optimizer: Adam optimizer over actor parameters only.
        * sampling_generator: Isolated generator used only for policy sampling.
    """

    STATE_VERSION = 1
    collection_mode = CollectionMode.COMPLETE_EPISODES

    def __init__(
        self,
        observation_dimensions: int,
        actor_config: ActorConfig,
        config: ReinforceConfig,
        actor_learning_rate: float,
        initialization_generator: torch.Generator,
        sampling_generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct the actor and its documented Adam optimizer.
        """
        self.config = config
        self.actor_config = actor_config
        self.collection_size = config.completed_episodes_per_update
        self.device = torch.device(device)
        self.dtype = dtype
        self.actor_learning_rate = float(actor_learning_rate)
        self.actor = ActorNetwork(
            observation_dimensions,
            actor_config,
            initialization_generator,
            device=self.device,
            dtype=dtype,
        )
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
            betas=(config.beta_1, config.beta_2),
            eps=config.optimizer_epsilon,
        )
        self.sampling_generator = sampling_generator

    def collect_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> CollectedAction:
        """
        Sample one detached bounded action using the isolated policy stream.
        """
        observation = self._observation_tensor(normalized_observation).unsqueeze(0)
        sample = self.actor.sample(observation, self.sampling_generator)
        return CollectedAction(
            raw_action=sample.raw_action[0].cpu().numpy(),
            env_action=sample.env_action[0].cpu().numpy(),
            behaviour_log_probability=float(sample.log_probability[0].item()),
            current_value=None,
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

    def bootstrap_value(self, normalized_observation: NDArray[np.float32]) -> None:
        """
        Report that actor-only REINFORCE has no bootstrap value.
        """
        del normalized_observation

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        """
        Apply one exact complete-episode REINFORCE optimizer update.
        """
        if update_input.mode is not CollectionMode.COMPLETE_EPISODES:
            raise ValueError("REINFORCE requires complete-episode update input.")
        if len(update_input.episodes) != self.collection_size:
            raise ValueError(
                "REINFORCE updates require exactly "
                f"{self.collection_size} complete trajectories."
            )

        returns = tuple(
            monte_carlo_return_to_go(episode, self.config.discount, device=self.device)
            for episode in update_input.episodes
        )
        standardized_returns = self._standardize_returns(returns)
        trajectory_losses = tuple(
            self._trajectory_loss(episode, targets)
            for episode, targets in zip(
                update_input.episodes, standardized_returns, strict=True
            )
        )
        actor_loss = torch.stack(trajectory_losses).mean()
        entropy_proxy = self._entropy_proxy(update_input.episodes)
        actor_weight_norm = parameter_norm(self.actor.parameters())
        parameters_before = tuple(
            parameter.detach().clone() for parameter in self.actor.parameters()
        )

        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.gradient_norm_limit
            ).item()
        )
        self.optimizer.step()

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
            "optimizer": self.optimizer.state_dict(),
            "sampling_generator": self.sampling_generator.get_state(),
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
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampling_generator.set_state(state["sampling_generator"])

    @property
    def actor_parameter_count(self) -> int:
        """
        Return the number of trainable actor parameters, including dispersion.
        """
        return agent_parameter_counts(self.actor).actor

    @property
    def critic_parameter_count(self) -> None:
        """
        Report that actor-only REINFORCE has no critic parameters.
        """

    def _trajectory_loss(
        self, episode: OnPolicyRollout, standardized_returns: Tensor
    ) -> Tensor:
        log_probabilities = self._log_probabilities(episode)
        return -(log_probabilities * standardized_returns.detach()).sum()

    def _log_probabilities(self, episode: OnPolicyRollout) -> Tensor:
        tensors = episode.tensors(device=self.device)
        observations = tensors.observations.to(dtype=self.dtype)
        raw_actions = tensors.raw_actions.to(dtype=self.dtype)
        return self.actor.log_probability(observations, raw_actions)

    def _standardize_returns(self, returns: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        all_returns = torch.cat(returns)
        standardized = standardize(all_returns, self.config.optimizer_epsilon)
        lengths = tuple(return_.numel() for return_ in returns)
        return tuple(torch.split(standardized, list(lengths)))

    def _entropy_proxy(self, episodes: tuple[OnPolicyRollout, ...]) -> float:
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

    def _observation_tensor(self, observation: NDArray[np.float32]) -> Tensor:
        return torch.as_tensor(observation, dtype=self.dtype, device=self.device)
