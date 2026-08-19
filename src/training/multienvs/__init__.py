"""Many racing environments stepped as one, and the episodes that come out.

`vector_env` owns the persistent worker processes and knows nothing about
learning. `manager` drives them for one policy step and turns the result into
transitions and finished episodes, which is what a training loop actually wants.
`episodes` accumulates the in-flight episode each worker is racing.
"""

from .episodes import ActiveEpisode, EpisodeCollector
from .manager import CollectedStep, MultiEnvironmentManager
from .vector_env import (
    PersistentRacingVectorEnv,
    RacingWorkerState,
    VectorRacingState,
    vector_info,
    vector_worker_info,
)

__all__ = [
    "ActiveEpisode",
    "CollectedStep",
    "EpisodeCollector",
    "MultiEnvironmentManager",
    "PersistentRacingVectorEnv",
    "RacingWorkerState",
    "VectorRacingState",
    "vector_info",
    "vector_worker_info",
]
