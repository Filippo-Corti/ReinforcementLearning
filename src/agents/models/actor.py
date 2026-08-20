"""Trainable actor wrapper for bounded Gaussian policies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from configs.training import ActorConfig

from .policies import GaussianPolicy, Policy, PolicySample


class ActorNetwork(nn.Module, Policy):
    """
    Own the trainable Gaussian policy used as an agent's actor.

    This explicit actor role keeps algorithm code and parameter accounting from
    confusing a neural actor with scripted or random policies.

    Fields:
        * policy: Gaussian policy parametrized by this actor's neural network.
    """

    def __init__(
        self,
        observation_dimensions: int,
        config: ActorConfig,
        initialization_generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct the contained Gaussian policy from the actor configuration.
        """
        super().__init__()
        self.policy = GaussianPolicy(
            observation_dimensions,
            config,
            initialization_generator,
            device=device,
            dtype=dtype,
        )

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return the contained policy's deterministic action for evaluation.
        """
        return self.policy.action(observation)

    def sample(
        self, observations: Tensor, sampling_generator: torch.Generator
    ) -> PolicySample[Tensor, Tensor]:
        """
        Draw a detached training action from the contained Gaussian policy.
        """
        return self.policy.sample(observations, sampling_generator)

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """
        Return the bounded Gaussian mean action.
        """
        return self.policy.deterministic_action(observations)

    def sample_with_generators(
        self,
        observations: Tensor,
        sampling_generators: Sequence[torch.Generator],
    ) -> PolicySample[Tensor, Tensor]:
        """
        Draw one detached row per independent environment sampling stream.
        """
        return self.policy.sample_with_generators(observations, sampling_generators)

    def log_probability(self, observations: Tensor, raw_actions: Tensor) -> Tensor:
        """
        Recompute action log probabilities through the contained policy.
        """
        return self.policy.log_probability(observations, raw_actions)

    def project_parameters(self) -> None:
        """
        Restore the contained policy's bounds after an optimizer step.
        """
        self.policy.project_parameters()

    @property
    def parameter_count(self) -> int:
        """
        Return how many parameters this network trains.

        Every parameter here is trained: nothing is frozen and nothing is
        registered as a buffer, so there is no distinction to draw between
        trainable and total.
        """
        return sum(parameter.numel() for parameter in self.parameters())
