"""Integration tests for deterministic analysis bundle generation."""

from __future__ import annotations

import json

import matplotlib.image as mpimg

from experiments.analyze_results import analyze_results, parse_arguments
from recording import EpisodeOutcome, RunCategory
from tests.fixtures.analysis_runs import write_analysis_run


def test_analysis_bundle_is_order_independent_and_repeatable(tmp_path) -> None:
    results = tmp_path / "results"
    write_analysis_run(
        results / "later-name",
        root_identity=0,
        outcomes=(
            EpisodeOutcome.CRASHED,
            EpisodeOutcome.COMPLETED,
            EpisodeOutcome.COMPLETED,
            EpisodeOutcome.COMPLETED,
        ),
    )
    write_analysis_run(
        results / "earlier-name",
        root_identity=1,
        outcomes=(EpisodeOutcome.CRASHED,) * 4,
        return_offset=20.0,
    )

    first = tmp_path / "analysis-first"
    second = tmp_path / "analysis-second"
    first_manifest = analyze_results(
        results_root=results,
        output_directory=first,
        experiment=1,
        category=RunCategory.REPORTED,
    )
    second_manifest = analyze_results(
        results_root=results,
        output_directory=second,
        experiment=1,
        category=RunCategory.REPORTED,
    )

    assert first_manifest == second_manifest
    for first_path in sorted(first.glob("*.json")) + sorted(first.glob("*.csv")):
        assert first_path.read_bytes() == (second / first_path.name).read_bytes()
    for first_path in sorted(first.glob("*.png")):
        first_image = mpimg.imread(first_path)
        second_image = mpimg.imread(second / first_path.name)
        assert first_image.shape == second_image.shape
        assert (first_image == second_image).all()

    run_rows = json.loads((first / "run_summaries.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    assert [row["root_identity"] for row in run_rows] == [0, 1]
    cell_rows = json.loads((first / "cell_summaries.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    assert cell_rows[0]["completed_lap_count"] == 1
    assert cell_rows[0]["completed_lap_denominator"] == 2


def test_analysis_cli_requires_explicit_inputs() -> None:
    parsed = parse_arguments(
        [
            "--results-root",
            "results",
            "--output",
            "analysis",
            "--experiment",
            "2",
        ]
    )

    assert parsed.results_root == "results"
    assert parsed.experiment == 2
    assert parsed.run_category == RunCategory.REPORTED.value
