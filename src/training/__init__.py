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

from .buffers import TrainingTransition, Trajectory
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
    MultiEnvTrainingTransition,
    PersistentRacingVectorEnv,
    RacingWorkerState,
    VectorRacingState,
    VectorRollout,
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
    "MultiEnvTrainingTransition",
    "MultiEnvironmentManager",
    "PPOTrainingEngine",
    "PersistentRacingVectorEnv",
    "RacingWorkerState",
    "ReinforceTrainingEngine",
    "RunningObservationNormalizer",
    "TrainingCircuitSchedule",
    "TrainingCounters",
    "TrainingEngine",
    "TrainingRunState",
    "TrainingTransition",
    "TrainingUpdate",
    "Trajectory",
    "VectorRacingState",
    "VectorRollout",
    "circuit_track_seed",
    "evaluate_deterministic",
    "load_checkpoint",
    "save_checkpoint",
    "vector_info",
    "vector_worker_info",
]
