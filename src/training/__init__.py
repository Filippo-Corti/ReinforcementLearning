"""Shared collection, target, evaluation, and resume machinery for learning."""

from circuits import (
    CIRCUIT_IDENTITY_LIMIT,
    CircuitSplit,
    EvaluationCircuit,
    TrainingCircuitSchedule,
    circuit_track_seed,
)
from evaluation import evaluate_deterministic
from normalization import RunningObservationNormalizer

from .buffers import (
    FixedRolloutBuffer,
    GAETargets,
    OnPolicyRollout,
    OnPolicyTensors,
    ReinforceEpisodeBuffer,
    TrainingTransition,
    VectorOnPolicyRollout,
    VectorOnPolicyTensors,
    VectorRolloutBuffer,
    compute_gae_targets,
    compute_vector_gae_targets,
    monte_carlo_return_to_go,
)
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from .engines import (
    A2CTrainingEngine,
    PPOTrainingEngine,
    ReinforceTrainingEngine,
    TrainingCounters,
    TrainingEngine,
    TrainingRunState,
    TrainingUpdate,
)
from .multienvs import (
    MultiEnvironmentManager,
    PersistentRacingVectorEnv,
    RacingWorkerState,
    VectorRacingState,
    vector_info,
    vector_worker_info,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CIRCUIT_IDENTITY_LIMIT",
    "A2CTrainingEngine",
    "CheckpointError",
    "CircuitSplit",
    "EvaluationCircuit",
    "FixedRolloutBuffer",
    "GAETargets",
    "MultiEnvironmentManager",
    "OnPolicyRollout",
    "OnPolicyTensors",
    "PPOTrainingEngine",
    "PersistentRacingVectorEnv",
    "RacingWorkerState",
    "ReinforceEpisodeBuffer",
    "ReinforceTrainingEngine",
    "RunningObservationNormalizer",
    "TrainingCircuitSchedule",
    "TrainingCounters",
    "TrainingEngine",
    "TrainingRunState",
    "TrainingTransition",
    "TrainingUpdate",
    "VectorOnPolicyRollout",
    "VectorOnPolicyTensors",
    "VectorRacingState",
    "VectorRolloutBuffer",
    "circuit_track_seed",
    "compute_gae_targets",
    "compute_vector_gae_targets",
    "evaluate_deterministic",
    "load_checkpoint",
    "monte_carlo_return_to_go",
    "save_checkpoint",
    "vector_info",
    "vector_worker_info",
]
