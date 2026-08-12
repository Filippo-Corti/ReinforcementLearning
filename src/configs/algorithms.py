"""Immutable configurations for the three learning algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from .serialization import SerializableConfig


@dataclass(frozen=True, slots=True)
class ReinforceConfig(SerializableConfig):
    """
    Configure complete-episode REINFORCE collection and optimization.

    Fields:
        * discount: Reward discount used in return-to-go.
        * completed_episodes_per_update: Complete trajectories per update.
        * actor_learning_rate_candidates: Rates tried during calibration.
        * beta_1: Adam first-moment coefficient.
        * beta_2: Adam second-moment coefficient.
        * optimizer_epsilon: Adam numerical constant.
        * gradient_norm_limit: Global actor-gradient norm limit.
        * entropy_bonus_enabled: Whether entropy changes the actor loss.
        * learning_rate_scheduler_enabled: Whether a scheduler changes the rate.
    """

    discount: float = 0.9995
    completed_episodes_per_update: int = 8
    actor_learning_rate_candidates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    beta_1: float = 0.9
    beta_2: float = 0.999
    optimizer_epsilon: float = 1e-8
    gradient_norm_limit: float = 0.5
    entropy_bonus_enabled: bool = False
    learning_rate_scheduler_enabled: bool = False


@dataclass(frozen=True, slots=True)
class A2CConfig(SerializableConfig):
    """
    Configure synchronous advantage actor-critic with GAE.

    Fields:
        * discount: Reward discount used in temporal-difference targets.
        * gae_lambda: GAE trace parameter.
        * transitions_per_rollout: Transitions collected before each update.
        * learning_rate_candidates: Candidate `(actor, critic)` rate pairs.
        * beta_1: Adam first-moment coefficient.
        * beta_2: Adam second-moment coefficient.
        * optimizer_epsilon: Adam numerical constant.
        * gradient_norm_limit: Separate global norm limit for actor and critic.
        * entropy_bonus_enabled: Whether entropy changes the actor loss.
        * weight_decay_enabled: Whether Adam weight decay is used.
        * learning_rate_scheduler_enabled: Whether a scheduler changes the rates.
    """

    discount: float = 0.9995
    gae_lambda: float = 0.95
    transitions_per_rollout: int = 2_048
    learning_rate_candidates: tuple[tuple[float, float], ...] = (
        (1e-4, 3e-4),
        (3e-4, 1e-3),
    )
    beta_1: float = 0.9
    beta_2: float = 0.999
    optimizer_epsilon: float = 1e-8
    gradient_norm_limit: float = 0.5
    entropy_bonus_enabled: bool = False
    weight_decay_enabled: bool = False
    learning_rate_scheduler_enabled: bool = False


@dataclass(frozen=True, slots=True)
class PPOConfig(SerializableConfig):
    """
    Configure clipped proximal policy optimization with GAE.

    Clipping bounds how far one minibatch step may move the policy, but nothing
    bounds the drift accumulated across many reuses of the same rollout. Four
    epochs with an approximate-KL stop keeps that drift small enough that the
    policy does not periodically collapse and relearn, and costs far less
    optimizer time than ten unconditional epochs.

    Fields:
        * discount: Reward discount used in temporal-difference targets.
        * gae_lambda: GAE trace parameter.
        * transitions_per_rollout: Transitions collected before optimization.
        * optimization_epochs: Reuses of each rollout.
        * minibatch_size: Rows optimized together in one gradient step.
        * clip_epsilon: Importance-ratio clipping half-width.
        * value_clipping_enabled: Whether the critic loss is PPO-value clipped.
        * kl_early_stop_enabled: Whether approximate KL may end an update early.
        * target_kl: Mean per-epoch approximate KL above which an update stops.
        * learning_rate_candidates: Candidate `(actor, critic)` rate pairs.
        * beta_1: Adam first-moment coefficient.
        * beta_2: Adam second-moment coefficient.
        * optimizer_epsilon: Adam numerical constant.
        * gradient_norm_limit: Separate global norm limit for actor and critic.
        * entropy_bonus_enabled: Whether entropy changes the actor loss.
        * weight_decay_enabled: Whether Adam weight decay is used.
        * learning_rate_scheduler_enabled: Whether a scheduler changes the rates.
    """

    discount: float = 0.9995
    gae_lambda: float = 0.95
    transitions_per_rollout: int = 2_048
    optimization_epochs: int = 4
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    value_clipping_enabled: bool = False
    kl_early_stop_enabled: bool = True
    target_kl: float = 0.02
    learning_rate_candidates: tuple[tuple[float, float], ...] = (
        (1e-4, 3e-4),
        (3e-4, 1e-3),
    )
    beta_1: float = 0.9
    beta_2: float = 0.999
    optimizer_epsilon: float = 1e-8
    gradient_norm_limit: float = 0.5
    entropy_bonus_enabled: bool = False
    weight_decay_enabled: bool = False
    learning_rate_scheduler_enabled: bool = False
