"""Bounded Gaussian policy for continuous control."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional

from configs.training import ActorConfig

from .mlp import make_mlp

_LOG_TWO_PI = log(2.0 * torch.pi)


@dataclass(frozen=True, slots=True)
class PolicySample:
    """
    Store one detached stochastic action and its behaviour-policy quantities.

    Fields:
        * latent: Pre-squash Gaussian sample $U$.
        * action: Bounded environment action $\tanh(U)$.
        * log_probability: Corrected log probability of the complete action vector.
    """

    latent: Tensor
    action: Tensor
    log_probability: Tensor


class GaussianPolicy(nn.Module):
    """
    Map normalized observations to a diagonal Gaussian policy over bounded actions.

    Fields:
        * mean_network: MLP producing the unbounded Gaussian mean.
        * log_standard_deviation: Learned state-independent Gaussian dispersion.
        * log_standard_deviation_bounds: Bounds applied to dispersion during use.
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
        Initialize the mean MLP and learned diagonal dispersion.
        """
        super().__init__()
        self.mean_network = make_mlp(
            input_dimensions=observation_dimensions,
            output_dimensions=config.action_dimensions,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            hidden_initialization_gain=config.hidden_initialization_gain,
            output_initialization_gain=config.output_initialization_gain,
            initialization_generator=initialization_generator,
            device=device,
            dtype=dtype,
        )
        self.log_standard_deviation = nn.Parameter(
            torch.full(
                (config.action_dimensions,),
                config.initial_log_standard_deviation,
                device=device,
                dtype=dtype,
            )
        )
        self.log_standard_deviation_bounds = config.log_standard_deviation_bounds

    def sample(
        self, observations: Tensor, sampling_generator: torch.Generator
    ) -> PolicySample:
        """
        Draw and detach a bounded action using the supplied local generator.
        """
        with torch.no_grad():
            mean = self.mean(observations)
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=sampling_generator,
            )
            latent = mean + self.standard_deviation() * noise
            action = torch.tanh(latent)
            log_probability = self._log_probability_from_latent(observations, latent)
        return PolicySample(
            latent=latent.detach(),
            action=action.detach(),
            log_probability=log_probability.detach(),
        )

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """
        Return the bounded mean action without reading any sampling generator.
        """
        return torch.tanh(self.mean(observations))

    def log_probability(self, observations: Tensor, latent: Tensor) -> Tensor:
        """
        Recompute the corrected scalar log probability of stored latent actions.
        """
        return self._log_probability_from_latent(observations, latent)

    def mean(self, observations: Tensor) -> Tensor:
        """
        Return the Gaussian mean for one observation or an arbitrary observation batch.
        """
        return self.mean_network(observations)

    def standard_deviation(self) -> Tensor:
        """
        Return the bounded learned standard deviations used by the policy.
        """
        minimum, maximum = self.log_standard_deviation_bounds
        return torch.exp(self.log_standard_deviation.clamp(minimum, maximum))

    def _log_probability_from_latent(
        self, observations: Tensor, latent: Tensor
    ) -> Tensor:
        mean = self.mean(observations)
        log_standard_deviation = self.log_standard_deviation.clamp(
            *self.log_standard_deviation_bounds
        )
        standardized = (latent - mean) / torch.exp(log_standard_deviation)
        gaussian_log_density = -0.5 * (
            standardized.square() + 2.0 * log_standard_deviation + _LOG_TWO_PI
        )
        log_jacobian = 2.0 * (log(2.0) - latent - functional.softplus(-2.0 * latent))
        return (gaussian_log_density - log_jacobian).sum(dim=-1)
