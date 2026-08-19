"""Neural components the agents are built from.

Models live inside `agents` because that is the only thing they are for: an
actor is a policy and a critic is a value estimate, and neither means anything
without an algorithm training it.
"""

from __future__ import annotations

from .actor import ActorNetwork
from .counts import (
    TrainableParameterCounts,
    agent_parameter_counts,
    trainable_parameter_count,
)
from .critic import CriticNetwork
from .mlp import make_mlp
from .policies import (
    GaussianPolicy,
    Policy,
    PolicySample,
    RandomPolicy,
    ScriptedFrenetPolicy,
)

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "GaussianPolicy",
    "Policy",
    "PolicySample",
    "RandomPolicy",
    "ScriptedFrenetPolicy",
    "TrainableParameterCounts",
    "agent_parameter_counts",
    "make_mlp",
    "trainable_parameter_count",
]
