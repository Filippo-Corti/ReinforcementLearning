"""Isolated deterministic evaluation for project-owned on-policy agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from agents.types import OnPolicyAgent
from configs import ObservationRepresentation
from envs.observations import FrenetObservation
from envs.racing import RacingEnv
from recording.records import (
    CircuitGeometrySummaryRecord,
    DeterministicEvaluationRecord,
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationRecord,
    LoggedTransitionRecord,
    MetricScope,
    RunCategory,
    ScalarSummaryRecord,
)

from .normalization import RunningObservationNormalizer


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
    observation_type: ObservationRepresentation = ObservationRepresentation.FRENET,
    collection_duration: float = 0.0,
    optimization_duration: float = 0.0,
    near_saturated_steering_threshold: float | None = None,
) -> DeterministicEvaluationRecord:
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
            observation_type=observation_type,
            collection_duration=collection_duration,
            optimization_duration=optimization_duration,
            near_saturated_steering_threshold=near_saturated_steering_threshold,
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
    observation_type: ObservationRepresentation,
    collection_duration: float,
    optimization_duration: float,
    near_saturated_steering_threshold: float | None,
) -> DeterministicEvaluationRecord:
    total_return = 0.0
    maximum_progress = 0.0
    speeds: list[float] = []
    throttles: list[float] = []
    steering: list[float] = []
    transitions: list[LoggedTransitionRecord] = []
    resolved_circuit_identity = circuit_identity or str(
        environment.track.generation.seed
    )

    for step_index in range(environment.config.simulation.max_episode_steps):
        current_state = trajectory_state(environment, observation)
        normalized = normalizer.normalize(observation)
        with torch.inference_mode():
            action = agent.deterministic_action(normalized)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        progress = float(info["episode_progress"]) / environment.track.track_length
        maximum_progress = max(maximum_progress, progress)
        speeds.append(float(observation[2]))
        throttles.append(float(action[0]))
        steering.append(abs(float(action[1])))
        transitions.append(
            LoggedTransitionRecord(
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
                position=current_state.position,
                heading=current_state.heading,
                current_curvature=current_state.current_curvature,
                preview_curvature=current_state.preview_curvature,
                speed=current_state.speed,
                lateral_acceleration_proxy=current_state.lateral_acceleration_proxy,
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
                observation_type=observation_type.value,
                circuit_seed=environment.track.generation.seed,
                circuit_split=circuit_split,
                speed=_summary(speeds),
                throttle=_summary(throttles),
                absolute_steering=_summary(steering),
                positive_throttle_fraction=float(np.mean(np.asarray(throttles) > 0)),
                braking_fraction=float(np.mean(np.asarray(throttles) < 0)),
                near_saturated_steering_fraction=(
                    None
                    if near_saturated_steering_threshold is None
                    else float(
                        np.mean(
                            np.asarray(steering) >= near_saturated_steering_threshold
                        )
                    )
                ),
                circuit_geometry=circuit_geometry_summary(environment),
            )
            return DeterministicEvaluationRecord(
                record=EvaluationRecord(
                    run_category=run_category,
                    scope=MetricScope.EVALUATION,
                    evaluation_index=evaluation_index,
                    training_interactions=training_interactions,
                    evaluation_interactions=evaluation_interactions_before + count,
                    episode=episode,
                    normalizer_checksum=normalizer.checksum(),
                    collection_duration=collection_duration,
                    optimization_duration=optimization_duration,
                ),
                transitions=tuple(transitions),
            )
    raise RuntimeError("RacingEnv did not end within its configured episode limit.")


@dataclass(frozen=True, slots=True)
class TrajectoryState:
    """
    Preserve pre-action geometry and vehicle state for one retained trajectory row.

    Fields:
        * position: Cartesian vehicle position.
        * heading: Vehicle heading in radians.
        * current_curvature: Centerline curvature at the closest projection.
        * preview_curvature: Frenet preview curvature when present.
        * speed: Vehicle speed.
        * lateral_acceleration_proxy: Speed squared times absolute curvature.
    """

    position: tuple[float, float]
    heading: float
    current_curvature: float
    preview_curvature: float | None
    speed: float
    lateral_acceleration_proxy: float


def circuit_geometry_summary(environment: RacingEnv) -> CircuitGeometrySummaryRecord:
    """
    Summarize one frozen sampled circuit for later stratified analysis.
    """
    absolute_curvature = np.abs(environment.track.curvature)
    return CircuitGeometrySummaryRecord(
        track_length=float(environment.track.track_length),
        absolute_curvature=ScalarSummaryRecord(
            mean=float(np.mean(absolute_curvature)),
            standard_deviation=float(np.std(absolute_curvature)),
            minimum=float(np.min(absolute_curvature)),
            maximum=float(np.max(absolute_curvature)),
            quantiles={
                "q25": float(np.quantile(absolute_curvature, 0.25)),
                "q50": float(np.quantile(absolute_curvature, 0.50)),
                "q75": float(np.quantile(absolute_curvature, 0.75)),
                "q90": float(np.quantile(absolute_curvature, 0.90)),
            },
        ),
    )


def trajectory_state(
    environment: RacingEnv, observation: np.ndarray
) -> TrajectoryState:
    """
    Read the current vehicle and projected circuit geometry before an action.
    """
    if environment.state is None:
        raise RuntimeError(
            "RacingEnv must have active vehicle state during evaluation."
        )
    state = environment.state
    projection = environment.track_with_geometry.centerline_projector.project(
        state.position()
    )
    s = float(
        environment.track.s[projection.segment_index]
        + projection.fraction
        * environment.track_with_geometry.centerline_projector.lengths[
            projection.segment_index
        ]
    )
    current_curvature = environment.track_with_geometry.curvature(s)
    preview_curvature = (
        float(observation[-1])
        if observation.shape == (FrenetObservation.DIMENSIONS,)
        else None
    )
    return TrajectoryState(
        position=(float(state.x), float(state.y)),
        heading=float(state.heading),
        current_curvature=current_curvature,
        preview_curvature=preview_curvature,
        speed=float(state.speed),
        lateral_acceleration_proxy=float(state.speed**2 * abs(current_curvature)),
    )


def _summary(values: list[float]) -> ScalarSummaryRecord:
    """
    Return a population scalar summary for one non-empty action-level signal.
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
    if terminated and bool(info["stalled"]):
        return EpisodeOutcome.STALLED
    if truncated:
        return EpisodeOutcome.TIME_LIMIT
    raise ValueError("Terminal transition lacks a supported RacingEnv outcome.")
