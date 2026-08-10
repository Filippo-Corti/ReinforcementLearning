"""Immutable configurations for the project's learning system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .serialization import SerializableConfig


@dataclass(frozen=True, slots=True)
class ActorConfig(SerializableConfig):
    """
    Configuration of the bounded-Gaussian policy network.

    Fields:
        * name: Human-readable actor-size identity.
        * hidden_sizes: Width of each hidden layer.
        * activation: Hidden-layer activation.
        * action_dimensions: Number of bounded control outputs.
        * hidden_initialization_gain: Orthogonal gain for hidden layers.
        * output_initialization_gain: Orthogonal gain for the mean output layer.
        * initial_log_standard_deviation: Shared initial log standard deviation.
        * log_standard_deviation_bounds: Bounds applied when using learned dispersion.
    """

    name: Literal["small", "medium", "large"]
    hidden_sizes: tuple[int, int]
    activation: Literal["tanh"] = "tanh"
    action_dimensions: int = 2
    hidden_initialization_gain: float = 2**0.5
    output_initialization_gain: float = 0.01
    initial_log_standard_deviation: float = -0.5
    log_standard_deviation_bounds: tuple[float, float] = (-5.0, 2.0)


SMALL_ACTOR_CONFIG = ActorConfig(name="small", hidden_sizes=(32, 32))
MEDIUM_ACTOR_CONFIG = ActorConfig(name="medium", hidden_sizes=(64, 64))
LARGE_ACTOR_CONFIG = ActorConfig(name="large", hidden_sizes=(256, 256))


@dataclass(frozen=True, slots=True)
class CriticConfig(SerializableConfig):
    """
    Configuration of the fixed-capacity value network.

    Fields:
        * hidden_sizes: Width of each hidden layer.
        * activation: Hidden-layer activation.
        * hidden_initialization_gain: Orthogonal gain for hidden layers.
        * output_initialization_gain: Orthogonal gain for the scalar output layer.
    """

    hidden_sizes: tuple[int, int] = (64, 64)
    activation: Literal["tanh"] = "tanh"
    hidden_initialization_gain: float = 2**0.5
    output_initialization_gain: float = 1.0


FIXED_CRITIC_CONFIG = CriticConfig()


@dataclass(frozen=True, slots=True)
class OptimizerConfig(SerializableConfig):
    """
    Shared Adam and gradient-norm clipping configuration.

    Fields:
        * beta_1: Adam first-moment coefficient.
        * beta_2: Adam second-moment coefficient.
        * epsilon: Adam numerical constant.
        * gradient_norm_limit: Per-network global gradient-norm limit.
        * entropy_bonus_enabled: Whether an entropy bonus changes the actor loss.
        * weight_decay_enabled: Whether Adam weight decay is used.
        * learning_rate_scheduler_enabled: Whether a scheduler changes learning rates.
    """

    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    gradient_norm_limit: float = 0.5
    entropy_bonus_enabled: bool = False
    weight_decay_enabled: bool = False
    learning_rate_scheduler_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ObservationNormalizationConfig(SerializableConfig):
    """
    Configuration of running-sum observation normalization.

    Fields:
        * accumulator_dtype: Precision used for counts and running sums.
        * variance_epsilon: Denominator safeguard.
        * normalized_value_limit: Absolute normalized-input safeguard.
        * update_during_evaluation: Whether evaluation may update statistics.
    """

    accumulator_dtype: Literal["float64"] = "float64"
    variance_epsilon: float = 1e-8
    normalized_value_limit: float = 10.0
    update_during_evaluation: bool = False


@dataclass(frozen=True, slots=True)
class EpisodeCollectionConfig(SerializableConfig):
    """
    Collection configuration for complete Monte Carlo trajectories.

    Fields:
        * completed_episodes_per_update: Complete trajectories in one REINFORCE update.
    """

    completed_episodes_per_update: int = 8


@dataclass(frozen=True, slots=True)
class RolloutCollectionConfig(SerializableConfig):
    """
    Fixed-length transition collection for critic-based algorithms.

    Fields:
        * transitions_per_rollout: Maximum stored transitions in one rollout.
    """

    transitions_per_rollout: int = 2_048


@dataclass(frozen=True, slots=True)
class LearningRatePair(SerializableConfig):
    """
    One candidate pair of actor and critic learning rates.

    Fields:
        * actor: Adam learning rate for the actor.
        * critic: Adam learning rate for the critic.
    """

    actor: float
    critic: float


@dataclass(frozen=True, slots=True)
class ReinforceConfig(SerializableConfig):
    """
    Learning configuration for the project-owned REINFORCE agent.

    Fields:
        * discount: Reward discount used in return-to-go.
        * collection: Complete-episode collection configuration.
        * actor_learning_rate_candidates: Rates resolved only by pre-experiment calibration.
    """

    discount: float = 0.9995
    collection: EpisodeCollectionConfig = field(default_factory=EpisodeCollectionConfig)
    actor_learning_rate_candidates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)


@dataclass(frozen=True, slots=True)
class A2CConfig(SerializableConfig):
    """
    Learning configuration for synchronous actor-critic with GAE.

    Fields:
        * discount: Reward discount used in temporal-difference targets.
        * gae_lambda: GAE trace parameter.
        * collection: Fixed-length rollout collection configuration.
        * learning_rate_candidates: Pairs resolved only by pre-experiment calibration.
    """

    discount: float = 0.9995
    gae_lambda: float = 0.95
    collection: RolloutCollectionConfig = field(default_factory=RolloutCollectionConfig)
    learning_rate_candidates: tuple[LearningRatePair, ...] = (
        LearningRatePair(actor=1e-4, critic=3e-4),
        LearningRatePair(actor=3e-4, critic=1e-3),
    )


@dataclass(frozen=True, slots=True)
class PPOConfig(SerializableConfig):
    """
    Learning configuration for clipped proximal policy optimization.

    Fields:
        * discount: Reward discount used in temporal-difference targets.
        * gae_lambda: GAE trace parameter.
        * collection: Fixed-length rollout collection configuration.
        * optimization_epochs: Reuses of each rollout during optimization.
        * minibatch_size: Rows optimized together during one update.
        * clip_epsilon: Importance-ratio clipping half-width.
        * value_clipping_enabled: Whether the critic loss is PPO-value clipped.
        * kl_early_stop_enabled: Whether approximate KL can stop an update early.
        * learning_rate_candidates: Pairs resolved only by pre-experiment calibration.
    """

    discount: float = 0.9995
    gae_lambda: float = 0.95
    collection: RolloutCollectionConfig = field(default_factory=RolloutCollectionConfig)
    optimization_epochs: int = 10
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    value_clipping_enabled: bool = False
    kl_early_stop_enabled: bool = False
    learning_rate_candidates: tuple[LearningRatePair, ...] = (
        LearningRatePair(actor=1e-4, critic=3e-4),
        LearningRatePair(actor=3e-4, critic=1e-3),
    )


@dataclass(frozen=True, slots=True)
class TrainingScheduleConfig(SerializableConfig):
    """
    Training budget and checkpoint cadence shared by reported runs.

    Fields:
        * training_interaction_budget: Environment steps available to learning.
        * checkpoint_interval: Training interactions between checkpoints.
    """

    training_interaction_budget: int = 2_000_000
    checkpoint_interval: int = 250_000


@dataclass(frozen=True, slots=True)
class EvaluationConfig(SerializableConfig):
    """
    Deterministic evaluation policy and cadence.

    Fields:
        * evaluation_interval: Training interactions between evaluations.
        * deterministic_actions: Whether evaluation removes Gaussian exploration.
        * evaluation_updates_normalizer: Whether evaluation changes observation statistics.
    """

    evaluation_interval: int = 50_000
    deterministic_actions: bool = True
    evaluation_updates_normalizer: bool = False


@dataclass(frozen=True, slots=True)
class LoggingConfig(SerializableConfig):
    """
    Required raw-record categories and deterministic trajectory cadence.

    Fields:
        * record_episode_metrics: Whether every episode emits a record.
        * record_update_metrics: Whether every optimizer update emits a record.
        * record_evaluation_metrics: Whether every evaluation emits a record.
        * record_hardware_context: Whether machine and dependency context is retained.
        * trajectory_interval: Training interactions between retained trajectories.
    """

    record_episode_metrics: bool = True
    record_update_metrics: bool = True
    record_evaluation_metrics: bool = True
    record_hardware_context: bool = True
    trajectory_interval: int = 250_000


@dataclass(frozen=True, slots=True)
class ExecutionConfig(SerializableConfig):
    """
    Device and threading policy for reproducible execution.

    Fields:
        * device: Requested torch device for neural-network work.
        * dtype: Tensor precision used for training and evaluation.
        * environment_workers: Concurrent environment workers.
        * intraop_threads: PyTorch intra-operation CPU threads.
        * interop_threads: PyTorch inter-operation CPU threads.
        * deterministic_algorithms: Whether PyTorch deterministic operations are required.
        * deterministic_warn_only: Whether nondeterministic operations warn instead of error.
        * cudnn_benchmark: Whether cuDNN may benchmark and select algorithms dynamically.
    """

    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32"] = "float32"
    environment_workers: int = 1
    intraop_threads: int = 1
    interop_threads: int = 1
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False
    cudnn_benchmark: bool = False


@dataclass(frozen=True, slots=True)
class TrainingConfig(SerializableConfig):
    """
    Complete reusable learning-system configuration without selected learning rates.

    Fields:
        * actor: Policy architecture for the run.
        * critic: Fixed value-network architecture where an agent uses a critic.
        * optimizer: Shared optimizer safeguards.
        * normalization: Observation-normalization contract.
        * reinforce: REINFORCE collection and finite learning-rate candidates.
        * a2c: A2C collection and finite learning-rate candidates.
        * ppo: PPO collection, clipping and finite learning-rate candidates.
        * schedule: Training budget and checkpoint cadence.
        * evaluation: Deterministic evaluation policy and cadence.
        * logging: Required records and trajectory cadence.
        * execution: Device and deterministic threading policy.
    """

    actor: ActorConfig
    critic: CriticConfig = field(default_factory=lambda: FIXED_CRITIC_CONFIG)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    normalization: ObservationNormalizationConfig = field(
        default_factory=ObservationNormalizationConfig
    )
    reinforce: ReinforceConfig = field(default_factory=ReinforceConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    schedule: TrainingScheduleConfig = field(default_factory=TrainingScheduleConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
