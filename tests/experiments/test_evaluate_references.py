"""Tests for the standalone reference-evaluation command and reusable API."""

from __future__ import annotations

from experiments.evaluate_references import parse_arguments, run_reference_evaluation
from utils.artifacts import RunDirectory
from utils.metrics import RunCategory


def test_reference_evaluator_writes_complete_separated_deterministic_artifacts(
    tmp_path,
) -> None:
    first = run_reference_evaluation(
        seed=5,
        run_path=tmp_path / "first",
        references=("random", "scripted"),
    )
    second = run_reference_evaluation(
        seed=5,
        run_path=tmp_path / "second",
        references=("random", "scripted"),
    )

    assert [evaluation.episode.to_dict() for evaluation in first] == [
        evaluation.episode.to_dict() for evaluation in second
    ]
    assert [transition.action for transition in first[0].transitions] == [
        transition.action for transition in second[0].transitions
    ]
    run = RunDirectory.open(
        tmp_path / "first",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )
    completion = run.require_complete()
    assert completion["training_interactions"] == 0
    assert completion["evaluation_interactions"] == sum(
        evaluation.episode.interactions for evaluation in first
    )
    assert (run.path / "reference_track.json").is_file()
    assert completion["timing"]["evaluation"] > 0.0
    assert completion["timing"]["persistence"] > 0.0
    assert completion["timing"]["end_to_end"] >= completion["timing"]["evaluation"]


def test_reference_cli_requires_seed_and_accepts_output_alias() -> None:
    parsed = parse_arguments(["--seed", "2", "--output", "new-run"])

    assert parsed.seed == 2
    assert parsed.run_path == "new-run"
