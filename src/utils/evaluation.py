"""Isolated deterministic evaluation for project-owned on-policy agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from agents.types import OnPolicyAgent
from envs.racing import RacingEnv

from .metrics import (
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationRecord,
    MetricScope,
    RunCategory,
    ScalarSummary,
    TransitionRecord,
)
from .normalizers import RunningObservationNormalizer


@dataclass(frozen=True, slots=True)
class DeterministicEvaluation:
    """
    Store one isolated deterministic evaluation and its semantic trajectory.

    Fields:
        * record: Checkpoint-linked evaluation summary.
        * transitions: Ordered action-level trajectory retained for diagnosis.
    """

    record: EvaluationRecord
    transitions: tuple[TransitionRecord, ...]


def evaluate_deterministic(
    environment_factory: Callable[[], RacingEnv],
    agent: OnPolicyAgent,
    normalizer: RunningObservationNormalizer,
    *,
    run_category: RunCategory,
    evaluation_index: int,
    training_interactions: int,
    evaluation_interactions_before: int,
    reset_seed: int | None,
    root_identity: int | None = None,
    circuit_identity: str | None = None,
    circuit_split: str | None = None,
) -> DeterministicEvaluation:
    """
    Evaluate a policy on a fresh environment using only frozen observation statistics.
    """
    environment = environment_factory()
    try:
        observation, _ = environment.reset(seed=reset_seed)
        return _evaluate_episode(
            environment,
            agent,
            normalizer,
            observation=observation,
            run_category=run_category,
            evaluation_index=evaluation_index,
            training_interactions=training_interactions,
            evaluation_interactions_before=evaluation_interactions_before,
            root_identity=root_identity,
            circuit_identity=circuit_identity,
            circuit_split=circuit_split,
        )
    finally:
        environment.close()


def _evaluate_episode(
    environment: RacingEnv,
    agent: OnPolicyAgent,
    normalizer: RunningObservationNormalizer,
    *,
    observation: np.ndarray,
    run_category: RunCategory,
    evaluation_index: int,
    training_interactions: int,
    evaluation_interactions_before: int,
    root_identity: int | None,
    circuit_identity: str | None,
    circuit_split: str | None,
) -> DeterministicEvaluation:
    total_return = 0.0
    maximum_progress = 0.0
    speeds: list[float] = []
    throttles: list[float] = []
    steering: list[float] = []
    transitions: list[TransitionRecord] = []
    resolved_circuit_identity = circuit_identity or str(
        environment.track.generation.seed
    )

    for step_index in range(environment.config.simulation.max_episode_steps):
        normalized = normalizer.normalize_frozen(observation)
        with torch.inference_mode():
            action = agent.deterministic_action(normalized)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        progress = float(info["episode_progress"]) / environment.track.track_length
        maximum_progress = max(maximum_progress, progress)
        speeds.append(float(observation[2]))
        throttles.append(float(action[0]))
        steering.append(abs(float(action[1])))
        transitions.append(
            TransitionRecord(
                run_category=run_category,
                scope=MetricScope.EVALUATION,
                episode_index=evaluation_index,
                step_index=step_index,
                observation=tuple(float(value) for value in observation),
                next_observation=tuple(float(value) for value in next_observation),
                action=(float(action[0]), float(action[1])),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                collision=bool(info["collision"]),
                lap_completed=bool(info["lap_completed"]),
                progress=progress,
                elapsed_time=float(info["elapsed_time"]),
                circuit_identity=resolved_circuit_identity,
            )
        )
        total_return += float(reward)
        observation = next_observation
        if terminated or truncated:
            outcome = _outcome(terminated, truncated, info)
            count = len(transitions)
            episode = EpisodeRecord(
                run_category=run_category,
                scope=MetricScope.EVALUATION,
                episode_index=evaluation_index,
                outcome=outcome,
                undiscounted_return=total_return,
                training_target_total=None,
                interactions=count,
                simulated_time=float(info["elapsed_time"]),
                final_progress=progress,
                maximum_progress=maximum_progress,
                lap_time=(
                    float(info["elapsed_time"])
                    if outcome is EpisodeOutcome.COMPLETED
                    else None
                ),
                training_interactions=training_interactions,
                evaluation_interactions=evaluation_interactions_before + count,
                circuit_identity=resolved_circuit_identity,
                root_identity=root_identity,
                observation_type="frenet",
                circuit_seed=environment.track.generation.seed,
                circuit_split=circuit_split,
                speed=_summary(speeds),
                throttle=_summary(throttles),
                absolute_steering=_summary(steering),
                positive_throttle_fraction=float(np.mean(np.asarray(throttles) > 0)),
                braking_fraction=float(np.mean(np.asarray(throttles) < 0)),
            )
            return DeterministicEvaluation(
                record=EvaluationRecord(
                    run_category=run_category,
                    scope=MetricScope.EVALUATION,
                    evaluation_index=evaluation_index,
                    training_interactions=training_interactions,
                    evaluation_interactions=evaluation_interactions_before + count,
                    episode=episode,
                    normalizer_checksum=normalizer.checksum(),
                ),
                transitions=tuple(transitions),
            )
    raise RuntimeError("RacingEnv did not end within its configured episode limit.")


def _summary(values: list[float]) -> ScalarSummary:
    """
    Return a population scalar summary for one non-empty action-level signal.
    """
    array = np.asarray(values, dtype=np.float64)
    return ScalarSummary(
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )


def _outcome(
    terminated: bool, truncated: bool, info: dict[str, object]
) -> EpisodeOutcome:
    """
    Convert one explicit RacingEnv boundary into a metrics outcome.
    """
    if terminated and bool(info["lap_completed"]):
        return EpisodeOutcome.COMPLETED
    if terminated and bool(info["collision"]):
        return EpisodeOutcome.CRASHED
    if truncated:
        return EpisodeOutcome.TIME_LIMIT
    raise ValueError("Terminal transition lacks a supported RacingEnv outcome.")
