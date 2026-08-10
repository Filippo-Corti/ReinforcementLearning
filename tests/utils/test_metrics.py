"""Tests for semantic experiment metric records."""

from __future__ import annotations

import json

import pytest

from utils.metrics import (
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationRecord,
    MetricScope,
    RunCategory,
)


@pytest.mark.parametrize(
    "outcome",
    (EpisodeOutcome.COMPLETED, EpisodeOutcome.CRASHED, EpisodeOutcome.TIME_LIMIT),
)
def test_episode_records_preserve_distinct_environment_outcomes(
    outcome: EpisodeOutcome,
) -> None:
    record = EpisodeRecord(
        run_category=RunCategory.REDUCED_VALIDATION,
        scope=MetricScope.REFERENCE,
        episode_index=0,
        outcome=outcome,
        undiscounted_return=1.0,
        training_target_total=None,
        interactions=3,
        simulated_time=0.12,
        final_progress=0.2,
        maximum_progress=0.3,
        lap_time=None,
        training_interactions=0,
        evaluation_interactions=3,
        circuit_identity="test",
    )

    assert json.loads(json.dumps(record.to_dict()))["outcome"] == outcome.value


def test_evaluation_rejects_training_scope_and_category_mixing() -> None:
    episode = EpisodeRecord(
        run_category=RunCategory.PRE_EXPERIMENT,
        scope=MetricScope.EVALUATION,
        episode_index=0,
        outcome=EpisodeOutcome.TIME_LIMIT,
        undiscounted_return=0.0,
        training_target_total=None,
        interactions=1,
        simulated_time=0.04,
        final_progress=0.0,
        maximum_progress=0.0,
        lap_time=None,
        training_interactions=0,
        evaluation_interactions=1,
        circuit_identity="test",
    )

    with pytest.raises(ValueError, match="categories"):
        EvaluationRecord(
            run_category=RunCategory.REPORTED,
            scope=MetricScope.EVALUATION,
            evaluation_index=0,
            training_interactions=0,
            evaluation_interactions=1,
            episode=episode,
        )
    with pytest.raises(ValueError, match="training scope"):
        EvaluationRecord(
            run_category=RunCategory.PRE_EXPERIMENT,
            scope=MetricScope.TRAINING,
            evaluation_index=0,
            training_interactions=0,
            evaluation_interactions=1,
            episode=episode,
        )
