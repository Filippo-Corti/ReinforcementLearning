"""Episode evaluation shared by scripted and random baseline policies."""

from __future__ import annotations

import numpy as np

from envs.racing import RacingEnv
from models import Policy
from recording.records import (
    EpisodeOutcome,
    EpisodeRecord,
    LoggedTransitionRecord,
    MetricScope,
    PolicyEvaluationRecord,
    RunCategory,
    ScalarSummaryRecord,
)

from .evaluation import circuit_geometry_summary, trajectory_state


def evaluate_policy_episode(
    environment: RacingEnv,
    policy: Policy,
    *,
    run_category: RunCategory,
    episode_index: int = 0,
    training_interactions: int = 0,
    evaluation_interactions_before: int = 0,
    reset_seed: int | None = None,
    circuit_identity: str | None = None,
    root_identity: int | None = None,
    circuit_split: str | None = None,
) -> PolicyEvaluationRecord:
    """
    Run one baseline-policy episode and retain its summary and raw trajectory.
    """
    observation, _ = environment.reset(seed=reset_seed)
    total_return = 0.0
    maximum_progress = 0.0
    transitions: list[LoggedTransitionRecord] = []
    speeds: list[float] = []
    throttles: list[float] = []
    absolute_steering: list[float] = []
    resolved_circuit_identity = circuit_identity or str(
        environment.track.generation.seed
    )
    for step_index in range(environment.config.simulation.max_episode_steps):
        current_state = trajectory_state(environment, observation)
        action = policy.action(observation)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        speeds.append(float(observation[2]))
        throttles.append(float(action[0]))
        absolute_steering.append(abs(float(action[1])))
        progress = float(info["episode_progress"]) / environment.track.track_length
        maximum_progress = max(maximum_progress, progress)
        transitions.append(
            LoggedTransitionRecord(
                run_category=run_category,
                scope=MetricScope.REFERENCE,
                episode_index=episode_index,
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
                position=current_state.position,
                heading=current_state.heading,
                current_curvature=current_state.current_curvature,
                preview_curvature=current_state.preview_curvature,
                speed=current_state.speed,
                lateral_acceleration_proxy=(current_state.lateral_acceleration_proxy),
            )
        )
        total_return += float(reward)
        observation = next_observation
        if terminated or truncated:
            outcome = _episode_outcome(terminated, truncated, info)
            interactions = len(transitions)
            episode = EpisodeRecord(
                run_category=run_category,
                scope=MetricScope.REFERENCE,
                episode_index=episode_index,
                outcome=outcome,
                undiscounted_return=total_return,
                training_target_total=None,
                interactions=interactions,
                simulated_time=float(info["elapsed_time"]),
                final_progress=progress,
                maximum_progress=maximum_progress,
                lap_time=(
                    float(info["elapsed_time"])
                    if outcome is EpisodeOutcome.COMPLETED
                    else None
                ),
                training_interactions=training_interactions,
                evaluation_interactions=evaluation_interactions_before + interactions,
                circuit_identity=resolved_circuit_identity,
                root_identity=root_identity,
                observation_type="frenet",
                circuit_seed=environment.track.generation.seed,
                circuit_split=circuit_split,
                speed=_scalar_summary(speeds),
                throttle=_scalar_summary(throttles),
                absolute_steering=_scalar_summary(absolute_steering),
                positive_throttle_fraction=float(np.mean(np.asarray(throttles) > 0)),
                braking_fraction=float(np.mean(np.asarray(throttles) < 0)),
                circuit_geometry=circuit_geometry_summary(environment),
            )
            return PolicyEvaluationRecord(
                episode=episode, transitions=tuple(transitions)
            )
    raise RuntimeError("RacingEnv did not end within its configured episode limit.")


def _scalar_summary(values: list[float]) -> ScalarSummaryRecord:
    """
    Summarize one non-empty episode signal using population dispersion.
    """
    array = np.asarray(values, dtype=np.float64)
    return ScalarSummaryRecord(
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        quantiles={
            "q25": float(np.quantile(array, 0.25)),
            "q50": float(np.quantile(array, 0.50)),
            "q75": float(np.quantile(array, 0.75)),
            "q90": float(np.quantile(array, 0.90)),
        },
    )


def _episode_outcome(
    terminated: bool,
    truncated: bool,
    info: dict[str, object],
) -> EpisodeOutcome:
    """
    Convert the environment's terminal flags into one explicit outcome.
    """
    if terminated and bool(info["lap_completed"]):
        return EpisodeOutcome.COMPLETED
    if terminated and bool(info["collision"]):
        return EpisodeOutcome.CRASHED
    if terminated and bool(info["stalled"]):
        return EpisodeOutcome.STALLED
    if truncated:
        return EpisodeOutcome.TIME_LIMIT
    raise ValueError("Terminal transition lacks a supported RacingEnv outcome.")
