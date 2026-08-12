"""Learned, scripted, and random policies for bounded racing actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import log
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional

from configs.training import ActorConfig

from .mlp import make_mlp

_LOG_TWO_PI = log(2.0 * torch.pi)


class Policy(ABC):
    """
    Transform one environment observation into one bounded action.

    A policy may be learned, scripted, or random. This small abstract interface
    lets evaluation code drive the environment without knowing how the action
    was selected.
    """

    @abstractmethod
    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return one normalized throttle/brake and steering action.
        """


@dataclass(frozen=True, slots=True)
class ScriptedFrenetPolicy(Policy):
    """
    Drive from Frenet observations with a deterministic hand-written policy.

    Steering combines lateral and heading-error feedback with curvature
    feed-forward. The target speed decreases with absolute curvature; its error
    determines throttle or braking. Every control is clipped to `[-1, 1]`.

    Its target lateral acceleration sits below the car's friction budget rather
    than at it. Preview curvature is averaged over the lookahead and therefore
    understates a corner on entry, and braking spends grip that then cannot be
    used to turn, so a controller aiming at the limit arrives at corners already
    over it.

    Fields:
        * lateral_gain: Steering correction for lateral distance.
        * heading_gain: Steering correction for heading error.
        * curvature_gain: Steering feed-forward from preview curvature.
        * lateral_acceleration_limit: Curvature-dependent speed-target constant.
        * maximum_target_speed: Upper bound for target speed.
        * speed_error_scale: Speed error that saturates throttle or braking.
    """

    lateral_gain: float = 0.15
    heading_gain: float = 0.8
    curvature_gain: float = 10.0
    lateral_acceleration_limit: float = 12.0
    maximum_target_speed: float = 50.0
    speed_error_scale: float = 6.0

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply the documented Frenet feedback and target-speed equations.
        """
        return self.sample(observation).env_action

    def sample(
        self, observation: NDArray[np.float32]
    ) -> PolicySample[NDArray[np.float32], None]:
        """
        Return the unbounded controls and the clipped environment action.
        """
        lateral_distance, heading_error, speed, _, curvature = map(float, observation)
        raw_steering = (
            -self.lateral_gain * lateral_distance
            - self.heading_gain * heading_error
            + self.curvature_gain * curvature
        )
        target_speed = min(
            self.maximum_target_speed,
            np.sqrt(self.lateral_acceleration_limit / max(abs(curvature), 1e-4)),
        )
        raw_throttle = (target_speed - speed) / self.speed_error_scale
        raw_action = np.asarray((raw_throttle, raw_steering), dtype=np.float32)
        env_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32, copy=False)
        return PolicySample(
            raw_action=raw_action,
            env_action=env_action,
            log_probability=None,
        )


@dataclass(slots=True)
class RandomPolicy(Policy):
    """
    Sample uniformly random bounded actions as an evaluation baseline.

    Fields:
        * generator: Isolated NumPy generator used only for baseline actions.
    """

    generator: np.random.Generator

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return one independent uniform action without global randomness.
        """
        return self.sample(observation).env_action

    def sample(
        self, observation: NDArray[np.float32]
    ) -> PolicySample[NDArray[np.float32], float]:
        """
        Return one uniform action and its density on the bounded action square.
        """
        del observation
        env_action = self.generator.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return PolicySample(
            raw_action=env_action.copy(),
            env_action=env_action,
            log_probability=-2.0 * log(2.0),
        )


@dataclass(frozen=True, slots=True)
class PolicySample[ActionValue, ProbabilityValue]:
    """
    Store a sampled bounded action and its behaviour-policy probability.

    Fields:
        * raw_action: Action before policy-specific clipping or transformation.
        * env_action: Bounded action after post-processing, sent to the environment.
        * log_probability: Log probability of the complete action vector, when defined.
    """

    raw_action: ActionValue
    env_action: ActionValue
    log_probability: ProbabilityValue


class GaussianPolicy(nn.Module, Policy):
    """
    Map observations to a diagonal Gaussian and bounded continuous actions.

    The policy feeds normalized observations into an MLP that outputs the mean
    action vector.
    Then:
    * If acting deterministically, the mean is bounded via `tanh`.
    * If acting stochastically, Gaussian noise uses the learned standard
      deviation, then the sample is bounded via `tanh`.

    Fields:
        * mean: MLP representing the observation-dependent Gaussian mean.
        * log_standard_deviation: Learned state-independent log scale.
        * log_standard_deviation_bounds: [Min, Max] bounds applied before exponentiation.
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
        self._mean = make_mlp(
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
        with torch.no_grad():
            output_layer = cast(nn.Linear, self._mean[-1])
            output_layer.bias.copy_(
                torch.as_tensor(
                    config.initial_action_bias,
                    device=output_layer.bias.device,
                    dtype=output_layer.bias.dtype,
                )
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

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return the deterministic bounded action for one NumPy observation.
        """
        parameter = next(self.parameters())
        tensor = torch.as_tensor(
            observation,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.inference_mode():
            action = self.deterministic_action(tensor)
        return action.cpu().numpy().astype(np.float32, copy=False)

    def sample(
        self, observations: Tensor, sampling_generator: torch.Generator
    ) -> PolicySample[Tensor, Tensor]:
        """
        Sample u = mean(x) + standard_deviation * noise and return tanh(u).

        Sampling is detached because collection data defines a fixed behaviour
        policy. Optimization later recomputes log probabilities with gradients.
        """
        with torch.no_grad():
            mean = self.mean(observations)
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=sampling_generator,
            )
            raw_action = mean + self.standard_deviation * noise
            env_action = torch.tanh(raw_action)
            log_probability = self._log_probability_from_pre_squash_action(
                observations, raw_action
            )
        return PolicySample(
            raw_action=raw_action.detach(),
            env_action=env_action.detach(),
            log_probability=log_probability.detach(),
        )

    def sample_with_generators(
        self,
        observations: Tensor,
        sampling_generators: Sequence[torch.Generator],
    ) -> PolicySample[Tensor, Tensor]:
        """
        Sample a batched mean with one independent generator per environment row.
        """
        if observations.ndim != 2:
            raise ValueError("Batched policy sampling requires a rank-two input.")
        if observations.shape[0] != len(sampling_generators):
            raise ValueError("One sampling generator is required per observation row.")
        with torch.no_grad():
            mean = self.mean(observations)
            noise = torch.stack(
                [
                    torch.randn(
                        mean.shape[1:],
                        dtype=mean.dtype,
                        device=mean.device,
                        generator=generator,
                    )
                    for generator in sampling_generators
                ]
            )
            raw_action = mean + self.standard_deviation * noise
            env_action = torch.tanh(raw_action)
            log_probability = self._log_probability_from_pre_squash_action(
                observations, raw_action
            )
        return PolicySample(
            raw_action=raw_action.detach(),
            env_action=env_action.detach(),
            log_probability=log_probability.detach(),
        )

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """
        Return tanh(mean(x)) without consuming a sampling generator.
        """
        return torch.tanh(self.mean(observations))

    def log_probability(self, observations: Tensor, raw_actions: Tensor) -> Tensor:
        """
        Recompute the corrected log probability of stored raw Gaussian actions.
        """
        return self._log_probability_from_pre_squash_action(observations, raw_actions)

    @property
    def mean(self) -> nn.Module:
        """
        Return the MLP mean(x) that maps observations to Gaussian means.

        The mean itself cannot be a stored tensor because it depends on the
        observation. Calling this module as `policy.mean(observations)` computes
        the appropriate mean vector for each supplied observation.
        """
        return self._mean

    @property
    def standard_deviation(self) -> Tensor:
        """
        Return exp(clamp(log_standard_deviation, minimum, maximum)).

        Clamping the learned log scale prevents vanishing exploration and
        numerically extreme Gaussian samples while preserving gradients inside
        the approved interval.
        """
        minimum, maximum = self.log_standard_deviation_bounds
        return torch.exp(self.log_standard_deviation.clamp(minimum, maximum))

    def _log_probability_from_pre_squash_action(
        self, observations: Tensor, raw_actions: Tensor
    ) -> Tensor:
        """
        Evaluate the bounded vector action density from its raw Gaussian value.

        For every component, this subtracts the `tanh` change-of-variables term
        log(1 - tanh(u_i)^2) from the diagonal Gaussian log density of u_i.
        The stable equivalent below avoids evaluating log(1 - a^2) near
        the action bounds. Components are summed because one environment action
        is a vector-valued joint event.
        """
        mean = self.mean(observations)
        log_standard_deviation = self.log_standard_deviation.clamp(
            *self.log_standard_deviation_bounds
        )
        standardized = (raw_actions - mean) / torch.exp(log_standard_deviation)
        gaussian_log_density = -0.5 * (
            standardized.square() + 2.0 * log_standard_deviation + _LOG_TWO_PI
        )
        log_jacobian = 2.0 * (
            log(2.0) - raw_actions - functional.softplus(-2.0 * raw_actions)
        )
        return (gaussian_log_density - log_jacobian).sum(dim=-1)
