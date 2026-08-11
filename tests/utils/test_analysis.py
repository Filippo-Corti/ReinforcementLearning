"""Tests for deterministic root-level experiment analysis."""

from __future__ import annotations

import math

import pytest

from recording import EpisodeOutcome, RunCategory
from tests.fixtures.analysis_runs import write_analysis_run
from utils.analysis import (
    cell_summary_rows,
    convergence_summary,
    curvature_control_rows,
    descriptive_statistics,
    load_recorded_runs,
    normalized_curve_area,
    paired_circuit_difference_rows,
    paired_difference_rows,
    representative_run_ids,
    run_summary_rows,
    stratify_circuit_geometry,
)


def test_known_curve_area_convergence_and_censoring() -> None:
    curve = [
        {
            "training_interactions": interaction,
            "training_duration": interaction / 10,
            "mean_return": value,
            "lap_target_rate": qualified,
            "completion_rate": qualified,
            "median_progress": value,
        }
        for interaction, value, qualified in (
            (10, 0.0, False),
            (20, 1.0, True),
            (30, 1.0, True),
            (40, 1.0, True),
        )
    ]

    assert normalized_curve_area(curve, "mean_return") == pytest.approx(5 / 6)
    assert convergence_summary(curve, experiment=1) == {
        "converged": True,
        "censored": False,
        "convergence_interactions": 20,
        "convergence_duration": 2.0,
        "restricted_convergence_interactions": 20,
        "restricted_convergence_duration": 2.0,
    }
    for row in curve:
        row["lap_target_rate"] = 0.0
    assert convergence_summary(curve, experiment=1)["censored"] is True
    assert (
        convergence_summary(curve, experiment=1)["restricted_convergence_interactions"]
        == 40
    )


def test_exact_root_statistics_and_pairing_ignore_row_order() -> None:
    statistics = descriptive_statistics([1.0, 3.0])
    assert statistics.mean == 2.0
    assert statistics.sample_standard_deviation == pytest.approx(math.sqrt(2.0))
    assert statistics.confidence_interval_low == pytest.approx(1.075)
    assert statistics.confidence_interval_high == pytest.approx(2.925)

    rows = [
        {"algorithm": "ppo", "actor_name": actor, "root_identity": root, "score": score}
        for actor, root, score in (
            ("large", 1, 8.0),
            ("small", 0, 2.0),
            ("small", 1, 3.0),
            ("large", 0, 5.0),
        )
    ]
    differences = paired_difference_rows(
        rows,
        condition_key="actor_name",
        left="small",
        right="large",
        fixed_keys=("algorithm",),
        metrics=("score",),
    )

    assert [(row["root_identity"], row["score"]) for row in differences] == [
        (0, -3.0),
        (1, -5.0),
    ]


def test_recorded_failures_denominators_and_representative_tie(tmp_path) -> None:
    write_analysis_run(
        tmp_path / "z-root-zero",
        root_identity=0,
        outcomes=(
            EpisodeOutcome.CRASHED,
            EpisodeOutcome.COMPLETED,
            EpisodeOutcome.COMPLETED,
            EpisodeOutcome.COMPLETED,
        ),
    )
    write_analysis_run(
        tmp_path / "a-root-one",
        root_identity=1,
        outcomes=(EpisodeOutcome.CRASHED,) * 4,
        return_offset=20.0,
    )
    runs = load_recorded_runs(tmp_path, category=RunCategory.REPORTED)
    summaries = run_summary_rows(runs, experiment=1)
    cells = cell_summary_rows(summaries, experiment=1)

    assert [run.root_identity for run in runs] == [0, 1]
    assert [row["censored"] for row in summaries] == [False, True]
    assert cells[0]["completed_lap_count"] == 1
    assert cells[0]["completed_lap_denominator"] == 2
    assert representative_run_ids(summaries, experiment=1) == {
        ("ppo", "small"): summaries[0]["run_id"]
    }
    curvature = curvature_control_rows(runs, summaries, experiment=1)
    assert sum(int(row["sample_count"]) for row in curvature) == 4
    assert {row["curvature_bin"] for row in curvature} == {"q1", "q2", "q3", "q4"}


def test_circuit_pairing_and_geometry_bins_preserve_circuit_identity() -> None:
    rows = [
        {
            "root_identity": 0,
            "circuit_identity": circuit,
            "circuit_split": "test",
            "observation_type": observation,
            "outcome": outcome,
            "return": return_,
            "maximum_progress": progress,
            "circuit_length": length,
            "curvature_q90": curvature,
        }
        for circuit, length, curvature, observation, outcome, return_, progress in (
            ("b", 120.0, 0.03, "lidar", "crashed", 1.0, 0.4),
            ("a", 80.0, 0.01, "frenet", "completed", 5.0, 1.0),
            ("b", 120.0, 0.03, "frenet", "completed", 4.0, 1.0),
            ("a", 80.0, 0.01, "lidar", "completed", 3.0, 1.0),
        )
    ]

    differences = paired_circuit_difference_rows(rows)
    assert [(row["circuit_identity"], row["return"]) for row in differences] == [
        ("a", 2.0),
        ("b", 3.0),
    ]
    strata = stratify_circuit_geometry(
        rows,
        length_edges=(100.0,),
        curvature_edges=(0.02,),
    )
    assert {(row["circuit_identity"], row["length_bin"]) for row in strata} == {
        ("a", 0),
        ("b", 1),
    }
