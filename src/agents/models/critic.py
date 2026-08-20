"""V-function critic network shared by actor-critic algorithms."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from configs.training import CriticConfig

from .mlp import make_mlp


class CriticNetwork(nn.Module):
    """
    Approximate the state-value function V with an MLP.

    A2C and PPO use this critic to estimate V(s) for bootstrapping,
    advantage estimation, and the value-regression objective. It returns one
    scalar estimate for each normalized observation.

    Fields:
        * network: MLP with one scalar output unit.
    """

    def __init__(
        self,
        observation_dimensions: int,
        config: CriticConfig,
        initialization_generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the fixed-capacity critic MLP from a local generator.
        """
        super().__init__()
        self.network = make_mlp(
            input_dimensions=observation_dimensions,
            output_dimensions=1,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            hidden_initialization_gain=config.hidden_initialization_gain,
            output_initialization_gain=config.output_initialization_gain,
            initialization_generator=initialization_generator,
            device=device,
            dtype=dtype,
        )

    def forward(self, observations: Tensor) -> Tensor:
        """
        Return one value per observation without a trailing singleton dimension.
        """
        return self.network(observations).squeeze(dim=-1)

    @property
    def parameter_count(self) -> int:
        """
        Return how many parameters this network trains.

        Every parameter here is trained: nothing is frozen and nothing is
        registered as a buffer, so there is no distinction to draw between
        trainable and total.
        """
        return sum(parameter.numel() for parameter in self.parameters())
