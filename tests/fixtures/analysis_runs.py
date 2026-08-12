"""Synthetic complete run directories with analytically simple outcomes."""

from __future__ import annotations

from pathlib import Path

from recording import (
    CircuitGeometrySummaryRecord,
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationRecord,
    LoggedTransitionRecord,
    MetricScope,
    ResourceRecord,
    RunCategory,
    RunDirectory,
    ScalarSummaryRecord,
    TimingRecord,
    UpdateRecord,
)


def write_analysis_run(
    path: Path,
    *,
    root_identity: int,
    actor_name: str = "small",
    algorithm: str = "ppo",
    observation_type: str = "frenet",
    outcomes: tuple[EpisodeOutcome, ...],
    return_offset: float = 0.0,
) -> RunDirectory:
    """
    Write one complete four-checkpoint run with a retained final trajectory.
    """
    if len(outcomes) != 4:
        raise ValueError("synthetic analysis runs require four outcomes.")
    run = RunDirectory.create(
        path,
        category=RunCategory.REPORTED,
        run_id=f"{algorithm}-{actor_name}-{observation_type}-{root_identity}",
        manifest={
            "purpose": "synthetic_analysis",
            "algorithm": algorithm,
            "root_seed": root_identity,
        },
        config={
            "training": {"actor": {"name": actor_name}},
            "environment": {},
        },
        metadata={"synthetic": True},
    )
    geometry = CircuitGeometrySummaryRecord(
        track_length=100.0,
        absolute_curvature=ScalarSummaryRecord(
            mean=0.02,
            standard_deviation=0.01,
            minimum=0.0,
            maximum=0.04,
            quantiles={"q25": 0.01, "q50": 0.02, "q75": 0.03, "q90": 0.036},
        ),
    )
    final_record: EvaluationRecord | None = None
    for index, outcome in enumerate(outcomes):
        boundary = (index + 1) * 10
        completed = outcome is EpisodeOutcome.COMPLETED
        episode = EpisodeRecord(
            run_category=RunCategory.REPORTED,
            scope=MetricScope.EVALUATION,
            episode_index=index,
            outcome=outcome,
            undiscounted_return=return_offset + boundary,
            training_target_total=None,
            interactions=4,
            simulated_time=90.0 if completed else 100.0,
            final_progress=1.0 if completed else boundary / 100.0,
            maximum_progress=1.0 if completed else boundary / 100.0,
            lap_time=90.0 if completed else None,
            training_interactions=boundary,
            evaluation_interactions=(index + 1) * 4,
            circuit_identity="fixed",
            root_identity=root_identity,
            observation_type=observation_type,
            circuit_seed=123,
            circuit_split="fixed",
            circuit_geometry=geometry,
        )
        record = EvaluationRecord(
            run_category=RunCategory.REPORTED,
            scope=MetricScope.EVALUATION,
            evaluation_index=index,
            training_interactions=boundary,
            evaluation_interactions=(index + 1) * 4,
            episode=episode,
            normalizer_checksum="normalizer",
            collection_duration=boundary * 0.08,
            optimization_duration=boundary * 0.02,
        )
        run.append("evaluations", record)
        run.append(
            "episodes",
            EpisodeRecord(
                run_category=RunCategory.REPORTED,
                scope=MetricScope.TRAINING,
                episode_index=index,
                outcome=outcome,
                undiscounted_return=return_offset + boundary / 2,
                training_target_total=None,
                interactions=10,
                simulated_time=100.0,
                final_progress=episode.final_progress,
                maximum_progress=episode.maximum_progress,
                lap_time=episode.lap_time,
                training_interactions=boundary,
                evaluation_interactions=index * 4,
                circuit_identity="fixed",
                root_identity=root_identity,
                observation_type=observation_type,
                circuit_seed=123,
                circuit_split="fixed",
                circuit_geometry=geometry,
            ),
        )
        run.append(
            "updates",
            UpdateRecord(
                run_category=RunCategory.REPORTED,
                update_index=index,
                training_interactions=boundary,
                actor_loss=-float(boundary),
                critic_loss=float(boundary) / 2,
                actor_gradient_norm=1.0,
                critic_gradient_norm=2.0,
                optimization_duration=0.2,
                entropy_proxy=0.5,
                actor_weight_norm=3.0,
                actor_update_norm=0.1,
                critic_weight_norm=4.0,
                critic_update_norm=0.2,
                explained_variance=0.25,
                approximate_kl=0.01,
                clip_fraction=0.05,
                diagnostics={"ratio_mean": 1.0},
            ),
        )
        final_record = record
    if final_record is None:
        raise RuntimeError("synthetic run did not create its final evaluation.")
    run.write_trajectory(
        "final",
        {
            "evaluation": final_record.to_dict(),
            "transitions": [
                _transition(root_identity, index, curvature, outcomes[-1]).to_dict()
                for index, curvature in enumerate((0.005, 0.015, 0.025, 0.035))
            ],
        },
    )
    timing = TimingRecord(
        run_category=RunCategory.REPORTED,
        scope=MetricScope.TRAINING,
        collection=3.2,
        optimization=0.8,
        evaluation=0.4,
        persistence=0.1,
        end_to_end=4.5,
    )
    resources = ResourceRecord(
        run_category=RunCategory.REPORTED,
        scope=MetricScope.TRAINING,
        training_interactions=40,
        evaluation_interactions=16,
        completed_episodes=sum(
            outcome is EpisodeOutcome.COMPLETED for outcome in outcomes
        ),
        optimizer_updates=4,
        actor_parameters=10,
        critic_parameters=5,
        peak_process_memory=None,
    )
    run.complete(
        {
            "training_interactions": 40,
            "evaluation_interactions": 16,
            "timing": timing.to_dict(),
            "resources": resources.to_dict(),
        }
    )
    return run


def _transition(
    root_identity: int,
    step_index: int,
    curvature: float,
    outcome: EpisodeOutcome,
) -> LoggedTransitionRecord:
    final = step_index == 3
    return LoggedTransitionRecord(
        run_category=RunCategory.REPORTED,
        scope=MetricScope.EVALUATION,
        episode_index=3,
        step_index=step_index,
        observation=(0.0, 0.0, 10.0, curvature),
        next_observation=(0.0, 0.0, 10.0, curvature),
        action=(0.2 + root_identity * 0.01, 0.1 * step_index),
        reward=1.0,
        terminated=final and outcome is not EpisodeOutcome.TIME_LIMIT,
        truncated=final and outcome is EpisodeOutcome.TIME_LIMIT,
        collision=final and outcome is EpisodeOutcome.CRASHED,
        lap_completed=final and outcome is EpisodeOutcome.COMPLETED,
        progress=(step_index + 1) / 4,
        elapsed_time=float(step_index + 1),
        circuit_identity="fixed",
        position=(float(step_index), 0.0),
        heading=0.0,
        current_curvature=curvature,
        preview_curvature=curvature,
        speed=10.0 + step_index,
        lateral_acceleration_proxy=(10.0 + step_index) ** 2 * curvature,
    )
