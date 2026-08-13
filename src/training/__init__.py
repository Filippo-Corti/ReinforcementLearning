"""Shared collection, target, evaluation, and resume machinery for learning."""

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
from .checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from .circuits import (
    CIRCUIT_IDENTITY_LIMIT,
    CircuitSplit,
    EvaluationCircuit,
    TrainingCircuitSchedule,
    circuit_track_seed,
)
from .engines.shared_engine import (
    OnPolicyTrainingEngine,
    TrainingCounters,
    TrainingRunState,
    TrainingUpdate,
)
from .evaluation import evaluate_deterministic
from .normalization import RunningObservationNormalizer
from .policy_evaluation import evaluate_policy_episode
from .vector_environment import (
    PersistentRacingVectorEnv,
    RacingWorkerState,
    VectorRacingState,
    vector_info,
    vector_worker_info,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CIRCUIT_IDENTITY_LIMIT",
    "CheckpointError",
    "CircuitSplit",
    "EvaluationCircuit",
    "FixedRolloutBuffer",
    "GAETargets",
    "OnPolicyRollout",
    "OnPolicyTensors",
    "OnPolicyTrainingEngine",
    "PersistentRacingVectorEnv",
    "RacingWorkerState",
    "ReinforceEpisodeBuffer",
    "RunningObservationNormalizer",
    "TrainingCircuitSchedule",
    "TrainingCounters",
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
    "evaluate_policy_episode",
    "load_checkpoint",
    "monte_carlo_return_to_go",
    "save_checkpoint",
    "vector_info",
    "vector_worker_info",
]
