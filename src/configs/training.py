"""Immutable configuration of models and the shared training lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import psutil

from .algorithms import A2CConfig, PPOConfig, ReinforceConfig
from .serialization import SerializableConfig


def physical_cpu_count() -> int:
    """
    Return the available physical CPU count, falling back to one worker.
    """
    return psutil.cpu_count(logical=False) or 1


@dataclass(frozen=True, slots=True)
class TrainingConfig(SerializableConfig):
    """
    Collect every model, algorithm, lifecycle, and execution choice for a run.

    Fields:
        * actor: Policy architecture used by the run.
        * critic: Fixed V-function architecture used by actor-critic agents.
        * normalization: Observation-normalization settings.
        * reinforce: REINFORCE-specific settings.
        * a2c: A2C-specific settings.
        * ppo: PPO-specific settings.
        * training_interaction_budget: Environment steps available to learning.
        * checkpoint_interval: Training interactions between saved checkpoints.
        * evaluation: Evaluation-policy and cadence settings.
        * logging: Run statistics and trajectory recording settings.
        * execution: Neural-network device and deterministic execution settings.
    """

    actor: ActorConfig
    critic: CriticConfig = field(default_factory=lambda: FIXED_CRITIC_CONFIG)
    normalization: ObservationNormalizationConfig = field(
        default_factory=lambda: ObservationNormalizationConfig()
    )
    reinforce: ReinforceConfig = field(default_factory=lambda: ReinforceConfig())
    a2c: A2CConfig = field(default_factory=lambda: A2CConfig())
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig())
    training_interaction_budget: int = 2_000_000
    checkpoint_interval: int = 250_000
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig())
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
    execution: ExecutionConfig = field(default_factory=lambda: ExecutionConfig())


@dataclass(frozen=True, slots=True)
class ActorConfig(SerializableConfig):
    """
    Configure the trainable network that parametrizes a bounded Gaussian policy.

    The hidden-layer gain controls the initial scale of features propagated by
    every hidden affine map. The smaller output gain keeps the initial Gaussian
    means near zero, so the initial bounded actions are not saturated. The
    initial log standard deviation sets the policy's initial exploration scale:
    each pre-squash action component initially has standard deviation
    `exp(initial_log_standard_deviation)`. During training, its learned log
    standard deviation is clamped to the configured bounds before exponentiation,
    preventing numerical collapse or excessive sampling variance.

    The upper bound matters more than it looks. Because actions are squashed by
    `tanh`, a large scale makes almost every sample saturate at a control limit,
    so the policy degenerates into random bang-bang control while its mean stops
    being identifiable. The bound keeps samples inside the responsive part of the
    squashing function.

    It is `-0.3`, a scale of about `0.74`, rather than `0`. At a scale of one PPO
    pressed against the bound for the last two thirds of a run and its finished
    policy still alternated between control limits, because the `tanh` change of
    variables rewards a wider policy whenever the actions with positive advantage
    are saturated ones, and PPO compounds that pressure over the many gradient
    steps it takes on each rollout. The bound applies to every algorithm, though
    only PPO has reached it: REINFORCE and A2C settle near `0.58` on their own.
    This lower ceiling is under evaluation, and the recorded learning-rate
    calibration was run at the previous one.

    The initial action bias exists for the same reason, one level down. A mean of
    zero is neutral in the *action*, but not in the physical quantity the action
    controls: braking is more than twice as strong as acceleration, so symmetric
    exploration noise around zero throttle decelerates the car by about
    `2.2 m/s^2` on average. An untrained policy therefore brakes to a standstill
    within seconds and every episode ends as a stall, which is precisely the
    degenerate behaviour the reward orderings are designed to avoid. The default
    throttle bias is the value at which the initial policy's expected
    acceleration is zero, so the untrained car coasts instead of braking.

    Fields:
        * name: Human-readable actor-size identity.
        * hidden_sizes: Width of each hidden layer.
        * learning_rate: Explicit per-run actor optimizer learning rate.
        * activation: Hidden-layer activation.
        * action_dimensions: Number of bounded control outputs.
        * hidden_initialization_gain: Orthogonal gain for hidden layers.
        * output_initialization_gain: Orthogonal gain for the mean output layer.
        * initial_log_standard_deviation: Initial log exploration scale.
        * log_standard_deviation_bounds: Bounds on the learned log scale.
        * initial_action_bias: Output-layer bias of the mean network per action.
    """

    name: Literal["small", "medium", "large"]
    hidden_sizes: tuple[int, int]
    learning_rate: float | None = None
    activation: Literal["tanh"] = "tanh"
    action_dimensions: int = 2
    hidden_initialization_gain: float = 2**0.5
    output_initialization_gain: float = 0.01
    initial_log_standard_deviation: float = -0.5
    log_standard_deviation_bounds: tuple[float, float] = (-5.0, -0.3)
    initial_action_bias: tuple[float, float] = (0.2, 0.0)


SMALL_ACTOR_CONFIG = ActorConfig(name="small", hidden_sizes=(32, 32))
MEDIUM_ACTOR_CONFIG = ActorConfig(name="medium", hidden_sizes=(64, 64))
LARGE_ACTOR_CONFIG = ActorConfig(name="large", hidden_sizes=(256, 256))


@dataclass(frozen=True, slots=True)
class CriticConfig(SerializableConfig):
    """
    Configure the fixed V-function critic used by actor-critic algorithms.

    Fields:
        * hidden_sizes: Width of each hidden layer.
        * activation: Hidden-layer activation.
        * hidden_initialization_gain: Orthogonal gain for hidden layers.
        * output_initialization_gain: Orthogonal gain for the scalar output.
    """

    hidden_sizes: tuple[int, int] = (64, 64)
    activation: Literal["tanh"] = "tanh"
    hidden_initialization_gain: float = 2**0.5
    output_initialization_gain: float = 1.0


FIXED_CRITIC_CONFIG = CriticConfig()


@dataclass(frozen=True, slots=True)
class ObservationNormalizationConfig(SerializableConfig):
    """
    Configure componentwise normalization of observations used as network inputs.

    Different observation components have different physical scales. Centering
    each component and dividing by its running standard deviation prevents one
    large-magnitude component from dominating early network updates.

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
class EvaluationConfig(SerializableConfig):
    """
    Configure deterministic evaluation episodes run during learning.

    Fields:
        * evaluation_interval: Training interactions between policy evaluations.
        * deterministic_actions: Whether evaluation removes Gaussian exploration.
        * evaluation_updates_normalizer: Whether evaluation changes input statistics.
    """

    evaluation_interval: int = 50_000
    deterministic_actions: bool = True
    evaluation_updates_normalizer: bool = False


@dataclass(frozen=True, slots=True)
class LoggingConfig(SerializableConfig):
    """
    Configure which training, optimization, and evaluation facts are recorded.

    Fields:
        * record_episode_metrics: Whether every training episode is recorded.
        * record_update_metrics: Whether every optimizer update is recorded.
        * record_evaluation_metrics: Whether every evaluation is recorded.
        * record_hardware_context: Whether machine and dependency context is retained.
        * trajectory_interval: Training interactions between saved trajectories.
        * near_saturated_steering_threshold: Explicit absolute steering threshold
          used by reported episode summaries.
    """

    record_episode_metrics: bool = True
    record_update_metrics: bool = True
    record_evaluation_metrics: bool = True
    record_hardware_context: bool = True
    trajectory_interval: int = 250_000
    near_saturated_steering_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class ExecutionConfig(SerializableConfig):
    """
    Configure device, precision, workers, and threading for a training run.

    Neural work runs on the CPU. The networks here are two hidden layers wide and
    are stepped in batches of one row per environment worker, which is far too
    small to amortize host-to-device transfer: measured over 20,000 interactions
    with eight workers, CUDA reached 1,144 interactions per second against the
    CPU's 3,694 for A2C, and was slower for all three algorithms. GPU execution is
    therefore not offered rather than merely not selected.

    Fields:
        * device: Torch device used for neural-network work.
        * dtype: Tensor precision used for training and evaluation.
        * environment_workers: Concurrent training-environment workers.
        * intraop_threads: PyTorch intra-operation CPU threads.
        * interop_threads: PyTorch inter-operation CPU threads.
        * deterministic_algorithms: Whether deterministic operations are required.
        * deterministic_warn_only: Whether nondeterministic operations only warn.
    """

    device: Literal["cpu"] = "cpu"
    dtype: Literal["float32"] = "float32"
    environment_workers: int = field(default_factory=physical_cpu_count)
    intraop_threads: int = 1
    interop_threads: int = 1
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False
