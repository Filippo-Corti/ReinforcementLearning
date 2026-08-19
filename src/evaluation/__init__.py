"""Deterministic policy evaluation, independent of how training collects data.

Training uses this package; nothing here reaches back into `training`. What
belongs here is the question "how good is the current policy," asked without
touching the policy, so the answer can never be shaped by how it was measured.
"""

from .deterministic import evaluate_deterministic
from .scheduler import EvaluationScheduler
from .utils import TrajectoryState, trajectory_state

__all__ = [
    "EvaluationScheduler",
    "TrajectoryState",
    "evaluate_deterministic",
    "trajectory_state",
]
