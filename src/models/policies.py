"""Learned, scripted, and random policies for bounded racing actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import log

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
    curvature_gain: float = 50.0
    lateral_acceleration_limit: float = 20.0
    maximum_target_speed: float = 50.0
    speed_error_scale: float = 10.0

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply the documented Frenet feedback and target-speed equations.
        """
        lateral_distance, heading_error, speed, curvature = map(float, observation)
        steering = np.clip(
            -self.lateral_gain * lateral_distance
            - self.heading_gain * heading_error
            + self.curvature_gain * curvature,
            -1.0,
            1.0,
        )
        target_speed = min(
            self.maximum_target_speed,
            np.sqrt(self.lateral_acceleration_limit / max(abs(curvature), 1e-4)),
        )
        throttle = np.clip((target_speed - speed) / self.speed_error_scale, -1.0, 1.0)
        return np.asarray((throttle, steering), dtype=np.float32)


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
        del observation
        return self.generator.uniform(-1.0, 1.0, size=2).astype(np.float32)


@dataclass(frozen=True, slots=True)
class PolicySample:
    r"""
    Store a sampled bounded action and its behaviour-policy probability.

    Fields:
        * pre_squash_action: Gaussian sample $u$ before the `tanh` transform.
        * action: Bounded environment action $a = \tanh(u)$.
        * log_probability: Corrected log probability of the complete action vector.
    """

    pre_squash_action: Tensor
    action: Tensor
    log_probability: Tensor


class GaussianPolicy(nn.Module, Policy):
    r"""
    Map observations to a diagonal Gaussian and bounded continuous actions.

    For normalized observation $x$, the mean network produces
    $\mu_\theta(x)$. A learned state-independent vector $\ell$ supplies
    $\sigma = \exp(\operatorname{clamp}(\ell, \ell_{min}, \ell_{max}))$.
    Training samples an unbounded pre-squash action
    $u = \mu_\theta(x) + \sigma \odot \epsilon$, with
    $\epsilon \sim \mathcal{N}(0, I)$, then sends $a = \tanh(u)$ to the
    environment. `action` and `deterministic_action` instead return
    $\tanh(\mu_\theta(x))$ without consuming randomness.

    Fields:
        * mean: MLP representing the observation-dependent Gaussian mean.
        * log_standard_deviation: Learned state-independent log scale.
        * log_standard_deviation_bounds: Bounds applied before exponentiation.
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
    ) -> PolicySample:
        r"""
        Sample $u = \mu_\theta(x) + \sigma \odot \epsilon$ and return `tanh(u)`.

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
            pre_squash_action = mean + self.standard_deviation * noise
            action = torch.tanh(pre_squash_action)
            log_probability = self._log_probability_from_pre_squash_action(
                observations, pre_squash_action
            )
        return PolicySample(
            pre_squash_action=pre_squash_action.detach(),
            action=action.detach(),
            log_probability=log_probability.detach(),
        )

    def deterministic_action(self, observations: Tensor) -> Tensor:
        r"""
        Return $\tanh(\mu_\theta(x))$ without consuming a sampling generator.
        """
        return torch.tanh(self.mean(observations))

    def log_probability(
        self, observations: Tensor, pre_squash_actions: Tensor
    ) -> Tensor:
        """
        Recompute the corrected log probability of stored pre-squash actions.
        """
        return self._log_probability_from_pre_squash_action(
            observations, pre_squash_actions
        )

    @property
    def mean(self) -> nn.Module:
        r"""
        Return the MLP $\mu_\theta$ that maps observations to Gaussian means.

        The mean itself cannot be a stored tensor because it depends on the
        observation. Calling this module as `policy.mean(observations)` computes
        the appropriate mean vector for each supplied observation.
        """
        return self._mean

    @property
    def standard_deviation(self) -> Tensor:
        r"""
        Return $\exp(\operatorname{clamp}(\ell, \ell_{min}, \ell_{max}))$.

        Clamping the learned log scale prevents vanishing exploration and
        numerically extreme Gaussian samples while preserving gradients inside
        the approved interval.
        """
        minimum, maximum = self.log_standard_deviation_bounds
        return torch.exp(self.log_standard_deviation.clamp(minimum, maximum))

    def _log_probability_from_pre_squash_action(
        self, observations: Tensor, pre_squash_actions: Tensor
    ) -> Tensor:
        r"""
        Evaluate the bounded vector action density from its pre-squash value.

        For every component, this subtracts the `tanh` change-of-variables term
        $\log(1 - \tanh^2(u_i))$ from the diagonal Gaussian log density of
        $u_i$. The stable equivalent below avoids evaluating `log(1-a^2)` near
        the action bounds. Components are summed because one environment action
        is a vector-valued joint event.
        """
        mean = self.mean(observations)
        log_standard_deviation = self.log_standard_deviation.clamp(
            *self.log_standard_deviation_bounds
        )
        standardized = (pre_squash_actions - mean) / torch.exp(log_standard_deviation)
        gaussian_log_density = -0.5 * (
            standardized.square() + 2.0 * log_standard_deviation + _LOG_TWO_PI
        )
        log_jacobian = 2.0 * (
            log(2.0)
            - pre_squash_actions
            - functional.softplus(-2.0 * pre_squash_actions)
        )
        return (gaussian_log_density - log_jacobian).sum(dim=-1)


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
    ) -> PolicySample:
        """
        Draw a detached training action from the contained Gaussian policy.
        """
        return self.policy.sample(observations, sampling_generator)

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """
        Return the bounded Gaussian mean action.
        """
        return self.policy.deterministic_action(observations)

    def log_probability(
        self, observations: Tensor, pre_squash_actions: Tensor
    ) -> Tensor:
        """
        Recompute action log probabilities through the contained policy.
        """
        return self.policy.log_probability(observations, pre_squash_actions)
