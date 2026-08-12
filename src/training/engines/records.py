"""Readable records returned by the algorithm-specific training engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from recording import EpisodeOutcome


@dataclass(frozen=True, slots=True)
class EducationalEpisodeRecord:
    """
    Summarize one complete training episode for inspection in educational runs.

    Fields:
        * episode_index: Zero-based order of the training episode.
        * circuit_identity: Seed-derived identity of the selected circuit.
        * interactions: Environment transitions collected in the episode.
        * undiscounted_return: Sum of the rewards observed in the episode.
        * outcome: Explicit racing lifecycle outcome.
        * final_progress: Fraction of a lap reached at the episode boundary.
        * maximum_progress: Largest lap fraction reached during the episode.
        * mean_speed: Mean pre-action vehicle speed across the episode.
        * mean_throttle: Mean signed throttle/brake action.
        * mean_throttle_magnitude: Mean absolute throttle/brake action.
        * lap_time: Simulated seconds taken, when the lap was completed.

    A bounded Gaussian policy whose mean output is zero still produces a large
    mean absolute action, because the exploration noise alone saturates through
    `tanh`. Only the signed mean separates a policy that has learned to
    accelerate from one whose throttle is pure noise, so both are recorded.
    """

    episode_index: int
    circuit_identity: str
    interactions: int
    undiscounted_return: float
    outcome: EpisodeOutcome
    final_progress: float
    maximum_progress: float
    mean_speed: float
    mean_throttle: float
    mean_throttle_magnitude: float
    lap_time: float | None


@dataclass(frozen=True, slots=True)
class EducationalUpdateRecord:
    """
    Summarize one completed optimizer update and its input collection.

    Fields:
        * update_index: Zero-based order of the optimizer update.
        * final_episode_index: Most recent episode represented when the update ran.
        * training_interactions: Total environment interactions collected so far.
        * transition_count: Number of transition rows used by the update.
        * diagnostics: Detached values returned by the agent update.
    """

    update_index: int
    final_episode_index: int
    training_interactions: int
    transition_count: int
    diagnostics: dict[str, float | int | None]


@dataclass(slots=True)
class EducationalTrainingHistory:
    """
    Collect episode and optimizer-update records produced during training.

    Fields:
        * episodes: Complete training-episode summaries in collection order.
        * updates: Optimizer-update summaries in application order.
        * training_interactions: Total environment transitions collected.
    """

    episodes: list[EducationalEpisodeRecord] = field(default_factory=list)
    updates: list[EducationalUpdateRecord] = field(default_factory=list)
    training_interactions: int = 0


def racing_outcome(
    terminated: bool, truncated: bool, info: Mapping[str, object]
) -> EpisodeOutcome:
    """
    Convert the environment's explicit episode boundary into its recorded outcome.
    """
    if terminated and bool(info["lap_completed"]):
        return EpisodeOutcome.COMPLETED
    if terminated and bool(info["collision"]):
        return EpisodeOutcome.CRASHED
    if terminated and bool(info["stalled"]):
        return EpisodeOutcome.STALLED
    if truncated:
        return EpisodeOutcome.TIME_LIMIT
    raise ValueError("Racing episode ended without a supported lifecycle outcome.")
