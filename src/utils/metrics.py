"""JSON-compatible semantic records for experiment observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

METRICS_SCHEMA_VERSION = 1


class RunCategory(StrEnum):
    """
    Identify the purpose of a run and the namespace of its artifacts.
    """

    PRE_EXPERIMENT = "pre_experiment_configuration"
    REDUCED_VALIDATION = "reduced_budget_end_to_end_validation"
    REPORTED = "reported_experiments"


class MetricScope(StrEnum):
    """
    Identify whether a record belongs to training or an isolated evaluation.
    """

    TRAINING = "training"
    EVALUATION = "evaluation"
    REFERENCE = "reference"


class EpisodeOutcome(StrEnum):
    """
    Preserve the environment's mutually exclusive lifecycle outcomes.
    """

    COMPLETED = "completed"
    CRASHED = "crashed"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    """
    Summarize one scalar episode signal without retaining every sample.

    Fields:
        * mean: Arithmetic mean.
        * standard_deviation: Population standard deviation.
        * minimum: Smallest observed value.
        * maximum: Largest observed value.
        * quantiles: Explicitly named quantiles when the run protocol requests them.
    """

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    quantiles: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """
    Store one environment transition without discarding lifecycle semantics.

    Fields:
        * run_category: Category that prevents cross-purpose aggregation.
        * scope: Whether the action belongs to training or an evaluation.
        * episode_index: Episode-local identifier within the run.
        * step_index: Zero-based agent action index within the episode.
        * observation: Frenet or other policy observation before the action.
        * next_observation: Observation returned after the action.
        * action: Normalized throttle/brake and steering action.
        * reward: Environment reward returned for the action.
        * terminated: Whether the environment ended through completion or crash.
        * truncated: Whether the environment reached its time limit.
        * collision: Whether the transition left the track.
        * lap_completed: Whether the transition completed a valid lap.
        * progress: Accumulated signed progress after the action.
        * elapsed_time: Simulated time after the action.
        * circuit_identity: Stable logical circuit identity.
    """

    run_category: RunCategory
    scope: MetricScope
    episode_index: int
    step_index: int
    observation: tuple[float, ...]
    next_observation: tuple[float, ...]
    action: tuple[float, float]
    reward: float
    terminated: bool
    truncated: bool
    collision: bool
    lap_completed: bool
    progress: float
    elapsed_time: float
    circuit_identity: str
    schema_version: int = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic JSON-compatible representation.
        """
        return _record_dict(self)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """
    Summarize one full training or evaluation episode.

    Fields:
        * run_category: Category that prevents cross-scope aggregation.
        * scope: Training, deterministic evaluation, or reference evaluation.
        * episode_index: Episode-local identifier within the run.
        * outcome: Explicit completion, crash, or time-limit result.
        * undiscounted_return: Sum of environment rewards.
        * training_target_total: Discounted target total when an agent defines one.
        * interactions: Calls to RacingEnv.step in this episode.
        * simulated_time: Environment elapsed simulated time.
        * final_progress: Normalized final progress around the circuit.
        * maximum_progress: Largest normalized progress reached in the episode.
        * lap_time: Simulated lap time only when outcome is completed.
        * training_interactions: Run training counter at episode end.
        * evaluation_interactions: Run evaluation/reference counter at episode end.
        * circuit_identity: Stable track identity when the run uses a circuit pool.
        * root_identity: Logical root identity that produced the episode.
        * observation_type: Observation representation used by the policy.
        * circuit_seed: Generated integer circuit seed.
        * circuit_split: Development, training, validation, test, or fixed split.
        * speed: Speed distribution summary.
        * throttle: Normalized throttle/brake distribution summary.
        * absolute_steering: Absolute normalized steering distribution summary.
        * positive_throttle_fraction: Fraction of actions with positive throttle.
        * braking_fraction: Fraction of actions with negative throttle.
        * near_saturated_steering_fraction: Fraction meeting the configured threshold.
    """

    run_category: RunCategory
    scope: MetricScope
    episode_index: int
    outcome: EpisodeOutcome
    undiscounted_return: float
    training_target_total: float | None
    interactions: int
    simulated_time: float
    final_progress: float
    maximum_progress: float
    lap_time: float | None
    training_interactions: int
    evaluation_interactions: int
    circuit_identity: str
    root_identity: int | None = None
    observation_type: str | None = None
    circuit_seed: int | None = None
    circuit_split: str | None = None
    speed: ScalarSummary | None = None
    throttle: ScalarSummary | None = None
    absolute_steering: ScalarSummary | None = None
    positive_throttle_fraction: float | None = None
    braking_fraction: float | None = None
    near_saturated_steering_fraction: float | None = None
    schema_version: int = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic JSON-compatible representation.
        """
        return _record_dict(self)


@dataclass(frozen=True, slots=True)
class UpdateRecord:
    """
    Store diagnostics from one optimizer update.

    Fields:
        * run_category: Category that prevents cross-scope aggregation.
        * update_index: Zero-based optimizer-update identifier.
        * training_interactions: Environment-step budget consumed before the update.
        * actor_loss: Mean actor loss for the update.
        * critic_loss: Mean critic loss where applicable.
        * actor_gradient_norm: Actor norm measured before gradient clipping.
        * critic_gradient_norm: Critic norm measured before gradient clipping.
        * optimization_duration: Exclusive optimization duration.
        * actor_learning_rate: Actor learning rate used for this update.
        * entropy_proxy: Negative mean log-probability diagnostic.
        * log_standard_deviation: Learned policy dispersion parameters.
        * actor_weight_norm: Actor parameter norm before the update.
        * actor_update_norm: Actor parameter-change norm.
        * critic_learning_rate: Critic learning rate where applicable.
        * critic_weight_norm: Critic parameter norm before the update.
        * critic_update_norm: Critic parameter-change norm.
        * explained_variance: Critic explained variance where applicable.
        * approximate_kl: PPO approximate KL where applicable.
        * clip_fraction: PPO clipped-sample fraction where applicable.
        * diagnostics: Algorithm-specific scalar diagnostics.
    """

    run_category: RunCategory
    update_index: int
    training_interactions: int
    actor_loss: float
    critic_loss: float | None
    actor_gradient_norm: float | None
    critic_gradient_norm: float | None
    optimization_duration: float
    actor_learning_rate: float | None = None
    entropy_proxy: float | None = None
    log_standard_deviation: tuple[float, ...] | None = None
    actor_weight_norm: float | None = None
    actor_update_norm: float | None = None
    critic_learning_rate: float | None = None
    critic_weight_norm: float | None = None
    critic_update_norm: float | None = None
    explained_variance: float | None = None
    approximate_kl: float | None = None
    clip_fraction: float | None = None
    diagnostics: dict[str, float | int | None] = field(default_factory=dict)
    scope: MetricScope = MetricScope.TRAINING
    schema_version: int = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic JSON-compatible representation.
        """
        return _record_dict(self)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """
    Link one deterministic/reference episode to its evaluation checkpoint.

    Fields:
        * run_category: Category that prevents cross-scope aggregation.
        * scope: Evaluation or reference, never training.
        * evaluation_index: Zero-based evaluation identifier.
        * training_interactions: Training budget at the evaluation boundary.
        * evaluation_interactions: Cumulative isolated evaluation interactions.
        * episode: Complete episode summary.
        * normalizer_checksum: Frozen normalizer state identity when applicable.
    """

    run_category: RunCategory
    scope: MetricScope
    evaluation_index: int
    training_interactions: int
    evaluation_interactions: int
    episode: EpisodeRecord
    normalizer_checksum: str | None = None
    schema_version: int = METRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.scope is MetricScope.TRAINING:
            raise ValueError("Evaluation records cannot use the training scope.")
        if self.episode.run_category is not self.run_category:
            raise ValueError("Evaluation and episode categories must agree.")

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic JSON-compatible representation.
        """
        return _record_dict(self)


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """
    Store mutually exclusive runtime categories for one reporting interval.

    Fields:
        * collection: Time spent stepping environments.
        * optimization: Time spent in actor/critic optimization.
        * evaluation: Time spent in deterministic/reference evaluation.
        * persistence: Time spent writing metrics or checkpoints.
        * end_to_end: Wall time from manifest acceptance to completion.
    """

    run_category: RunCategory
    scope: MetricScope
    collection: float = 0.0
    optimization: float = 0.0
    evaluation: float = 0.0
    persistence: float = 0.0
    end_to_end: float = 0.0
    schema_version: int = METRICS_SCHEMA_VERSION

    @property
    def training_only(self) -> float:
        """
        Return collection plus optimization time.
        """
        return self.collection + self.optimization

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic JSON-compatible representation.
        """
        record = _record_dict(self)
        record["training_only"] = self.training_only
        return record


@dataclass(frozen=True, slots=True)
class ResourceRecord:
    """
    Store run resource counts and optional process/GPU memory observations.

    Fields:
        * training_interactions: Steps used for learning.
        * evaluation_interactions: Steps used only for isolated evaluation.
        * completed_episodes: Episodes with a completed-lap outcome.
        * optimizer_updates: Completed parameter updates.
        * actor_parameters: Trainable actor parameter count.
        * critic_parameters: Trainable critic parameter count when applicable.
        * peak_process_memory: Peak process memory when available.
        * peak_gpu_memory: Peak GPU memory when available.
    """

    run_category: RunCategory
    scope: MetricScope
    training_interactions: int
    evaluation_interactions: int
    completed_episodes: int
    optimizer_updates: int
    actor_parameters: int
    critic_parameters: int | None
    peak_process_memory: int | None
    peak_gpu_memory: int | None
    schema_version: int = METRICS_SCHEMA_VERSION

    @property
    def total_parameters(self) -> int:
        """
        Return the complete trainable parameter count.
        """
        return self.actor_parameters + (self.critic_parameters or 0)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic JSON-compatible representation.
        """
        record = _record_dict(self)
        record["total_parameters"] = self.total_parameters
        return record


def _record_dict(record: Any) -> dict[str, Any]:
    """
    Convert a record to JSON data while preserving enum values.
    """
    return _json_value(asdict(record))


def _json_value(value: Any) -> Any:
    """
    Convert nested records, enums, and tuples to JSON-compatible values.
    """
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value
