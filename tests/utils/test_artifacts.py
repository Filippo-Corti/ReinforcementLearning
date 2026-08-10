"""Tests for versioned, safely persisted run artifacts."""

from __future__ import annotations

import json

import pytest

from utils.artifacts import ArtifactError, IncompleteRunError, RunDirectory
from utils.metrics import EpisodeOutcome, EpisodeRecord, MetricScope, RunCategory


def _create_run(
    tmp_path, category: RunCategory = RunCategory.REDUCED_VALIDATION
) -> RunDirectory:
    return RunDirectory.create(
        tmp_path / "run",
        category=category,
        run_id="test-run",
        manifest={"purpose": "test"},
        config={"value": 1},
        metadata={"machine": None},
    )


def _episode(category: RunCategory) -> EpisodeRecord:
    return EpisodeRecord(
        run_category=category,
        scope=MetricScope.REFERENCE,
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


def test_incomplete_runs_and_partial_jsonl_writes_are_rejected(tmp_path) -> None:
    run = _create_run(tmp_path)
    (run.path / "episodes.jsonl").write_text("{broken\n", encoding="utf-8")

    with pytest.raises(IncompleteRunError):
        RunDirectory.open(run.path, require_complete=True)
    with pytest.raises(ArtifactError, match="invalid episodes"):
        run.records("episodes")


def test_run_category_and_schema_mismatches_are_rejected(tmp_path) -> None:
    run = _create_run(tmp_path, RunCategory.PRE_EXPERIMENT)
    with pytest.raises(ArtifactError, match="reported_experiments"):
        RunDirectory.open(run.path, expected_category=RunCategory.REPORTED)
    with pytest.raises(ArtifactError, match="schema"):
        run.append("episodes", {"schema_version": 99})
    with pytest.raises(ArtifactError, match="category"):
        run.append("episodes", {"schema_version": 1})
    with pytest.raises(ArtifactError, match="category"):
        mismatched = _episode(RunCategory.REPORTED)
        run.append("episodes", mismatched)


def test_completed_run_round_trips_canonical_jsonl(tmp_path) -> None:
    run = _create_run(tmp_path)
    run.append("episodes", _episode(RunCategory.REDUCED_VALIDATION))
    run.complete({"training_interactions": 0, "evaluation_interactions": 1})

    reopened = RunDirectory.open(run.path, require_complete=True)
    assert reopened.records("episodes")[0]["outcome"] == "time_limit"
    assert reopened.require_complete()["training_interactions"] == 0
    assert (
        json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))["run_id"]
        == "test-run"
    )


def test_completion_rejects_a_partial_metric_append(tmp_path) -> None:
    run = _create_run(tmp_path)
    (run.path / "episodes.jsonl").write_text('{"schema_version":1}', encoding="utf-8")

    with pytest.raises(ArtifactError, match="partial append"):
        run.complete({"training_interactions": 0})
