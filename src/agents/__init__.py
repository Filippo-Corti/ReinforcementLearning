"""Bounded continuous on-policy agents, and the records they exchange.

`types` is the vocabulary an engine and an agent share, `targets` is what each
algorithm asks the collected experience to predict, `models` is what they are
built from, and `implementations` is the algorithms themselves.
"""

from .implementations import A2CAgent, OnPolicyAgent, PPOAgent, ReinforceAgent
from .targets import GAETargets, compute_vector_gae_targets, monte_carlo_return_to_go
from .types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectedActionBatch,
    CollectionMode,
    CompleteEpisodesInput,
    FixedRolloutInput,
)

__all__ = [
    "A2CAgent",
    "AgentUpdateInput",
    "AgentUpdateOutput",
    "CollectedAction",
    "CollectedActionBatch",
    "CollectionMode",
    "CompleteEpisodesInput",
    "FixedRolloutInput",
    "GAETargets",
    "OnPolicyAgent",
    "PPOAgent",
    "ReinforceAgent",
    "compute_vector_gae_targets",
    "monte_carlo_return_to_go",
]
