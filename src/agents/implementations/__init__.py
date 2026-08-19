"""One learning algorithm per file, over the contract they share.

`base` is everything an engine may ask of an agent, plus the few helpers that
are about being an agent in this project rather than about any algorithm.
Each other file is one algorithm and reads as that algorithm alone: what it
collects, what it fits, and how it takes its step.

Nothing here imports `training` at runtime. An algorithm is handed records and
reads their attributes, so it needs their types only to be annotated, and those
imports sit under `TYPE_CHECKING`. That is deliberate: `training` imports the
agent contract for real, and if the arrow pointed both ways at import time the
two packages would deadlock on whichever was loaded first.
"""

from .a2c import A2CAgent
from .base import OnPolicyAgent
from .ppo import PPOAgent
from .reinforce import ReinforceAgent

__all__ = [
    "A2CAgent",
    "OnPolicyAgent",
    "PPOAgent",
    "ReinforceAgent",
]
