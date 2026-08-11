"""Regenerate experiment tables and figures from complete recorded runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from recording import RunCategory, RunRecordingError
from utils.analysis import (
    TableRow,
    cell_summary_rows,
    curvature_control_rows,
    descriptive_statistics,
    evaluation_rows,
    final_split_rows,
    generalization_gap_rows,
    learning_curve_rows,
    load_recorded_runs,
    paired_circuit_difference_rows,
    paired_difference_rows,
    run_inventory_rows,
    run_summary_rows,
    stratify_circuit_geometry,
    training_episode_rows,
    update_rows,
)
from utils.plotting import (
    plot_circuit_geometry,
    plot_convergence_resources,
    plot_curvature_controls,
    plot_learning_curves,
    plot_optimization_diagnostics,
    plot_task_outcomes,
    save_figure,
)

ANALYSIS_SCHEMA_VERSION = 1
_PAIR_METRICS = (
    "final_completion_rate",
    "final_crash_rate",
    "final_mean_return",
    "final_mean_progress",
    "return_auc",
    "progress_auc",
    "restricted_convergence_interactions",
    "restricted_convergence_duration",
    "training_duration",
    "end_to_end_duration",
)


def analyze_results(
    *,
    results_root: str | Path,
    output_directory: str | Path,
    experiment: int,
    category: RunCategory = RunCategory.REPORTED,
    geometry_specification: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate inputs and write every deterministic table and requested figure.
    """
    if experiment not in (1, 2):
        raise ValueError("experiment must be 1 or 2.")
    runs = load_recorded_runs(results_root, category=category)
    if not runs:
        raise RunRecordingError("analysis found no complete recorded runs.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    evaluations = evaluation_rows(runs)
    curves = learning_curve_rows(runs, experiment=experiment)
    summaries = run_summary_rows(runs, experiment=experiment)
    cells = cell_summary_rows(summaries, experiment=experiment)
    updates = update_rows(runs)
    curvature = curvature_control_rows(
        runs,
        summaries,
        experiment=experiment,
    )
    if not curvature:
        raise RunRecordingError(
            "analysis requires retained final trajectories for curvature results."
        )
    paired = _experiment_pairs(summaries, experiment)
    paired_summaries = _paired_summary_rows(paired)

    tables: dict[str, list[dict[str, Any]]] = {
        "run_inventory": run_inventory_rows(runs),
        "training_episodes": training_episode_rows(runs),
        "evaluation_outcomes": evaluations,
        "learning_curves": curves,
        "run_summaries": summaries,
        "cell_summaries": cells,
        "paired_differences": paired,
        "paired_summaries": paired_summaries,
        "optimization_diagnostics": updates,
        "curvature_controls": curvature,
    }
    if experiment == 2:
        split_rows = final_split_rows(runs)
        final_evaluations = _final_evaluation_rows(runs, evaluations)
        tables.update(
            {
                "final_split_summaries": split_rows,
                "generalization_gaps": generalization_gap_rows(split_rows),
                "paired_circuit_differences": paired_circuit_difference_rows(
                    final_evaluations
                ),
            }
        )
        if geometry_specification is not None:
            specification = _read_json(Path(geometry_specification))
            tables["geometry_strata"] = stratify_circuit_geometry(
                final_evaluations,
                length_edges=_number_tuple(specification, "length_edges"),
                curvature_edges=_number_tuple(
                    specification,
                    "curvature_edges",
                ),
            )

    table_paths: list[Path] = []
    for name, rows in sorted(tables.items()):
        table_paths.extend(_write_table(output, name, rows))

    figure_paths = {
        "learning_curves.png": plot_learning_curves(curves),
        "task_outcomes.png": plot_task_outcomes(cells, summaries),
        "convergence_resources.png": plot_convergence_resources(summaries),
        "optimization_diagnostics.png": plot_optimization_diagnostics(updates),
        "curvature_controls.png": plot_curvature_controls(curvature),
    }
    if experiment == 2:
        figure_paths["circuit_geometry.png"] = plot_circuit_geometry(
            _final_evaluation_rows(runs, evaluations)
        )
    for name, figure in figure_paths.items():
        save_figure(figure, output / name)

    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": experiment,
        "run_category": category.value,
        "conventions": {
            "aggregation_unit": "training_root",
            "curve_area": "trapezoid_first_to_final_divided_by_recorded_span",
            "confidence_interval": "exhaustive_root_bootstrap_2.5_97.5_percentiles",
            "convergence": "first_of_three_consecutive_qualifying_checkpoints",
            "final_window_evaluations": 3,
            "representative_trajectory": "nearest_cell_median_final_return_then_lower_root",
            "curvature_bins": "circuit_sample_absolute_curvature_quartiles",
        },
        "inputs": [
            {
                "run_id": run.directory.run_id,
                "checksum": run.data_checksum,
                "manifest_checksum": run.directory.manifest_checksum(),
            }
            for run in runs
        ],
        "tables": {path.name: _file_checksum(path) for path in sorted(table_paths)},
        "figures": sorted(figure_paths),
    }
    _write_json(output / "analysis_manifest.json", manifest)
    return manifest


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse explicit analysis inputs, experiment identity, and output location.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", required=True, type=int, choices=(1, 2))
    parser.add_argument(
        "--run-category",
        choices=tuple(category.value for category in RunCategory),
        default=RunCategory.REPORTED.value,
    )
    parser.add_argument(
        "--geometry-specification",
        help="JSON with frozen length_edges and curvature_edges for Experiment 2.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Generate the complete deterministic analysis bundle.
    """
    parsed = parse_arguments(arguments)
    manifest = analyze_results(
        results_root=parsed.results_root,
        output_directory=parsed.output,
        experiment=parsed.experiment,
        category=RunCategory(parsed.run_category),
        geometry_specification=parsed.geometry_specification,
    )
    print(
        f"Analyzed {len(manifest['inputs'])} runs for Experiment "
        f"{manifest['experiment']}."
    )
    return 0


def _experiment_pairs(summaries: list[TableRow], experiment: int) -> list[TableRow]:
    pairs: list[TableRow] = []
    if experiment == 1:
        for left, right in (
            ("small", "medium"),
            ("small", "large"),
            ("medium", "large"),
        ):
            pairs.extend(
                paired_difference_rows(
                    summaries,
                    condition_key="actor_name",
                    left=left,
                    right=right,
                    fixed_keys=("algorithm",),
                    metrics=_PAIR_METRICS,
                )
            )
        for left, right in (
            ("reinforce", "a2c"),
            ("reinforce", "ppo"),
            ("a2c", "ppo"),
        ):
            pairs.extend(
                paired_difference_rows(
                    summaries,
                    condition_key="algorithm",
                    left=left,
                    right=right,
                    fixed_keys=("actor_name",),
                    metrics=_PAIR_METRICS,
                )
            )
    else:
        pairs.extend(
            paired_difference_rows(
                summaries,
                condition_key="observation_type",
                left="frenet",
                right="lidar",
                fixed_keys=("algorithm", "actor_name"),
                metrics=_PAIR_METRICS,
            )
        )
    return sorted(pairs, key=_row_sort_key)


def _paired_summary_rows(rows: list[TableRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[TableRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("algorithm", "")),
                str(row.get("actor_name", "")),
                str(row["contrast"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for identity, group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "algorithm": identity[0] or None,
            "actor_name": identity[1] or None,
            "contrast": identity[2],
            "root_count": len(group),
        }
        for metric in _PAIR_METRICS:
            summary[metric] = asdict(
                descriptive_statistics([float(row[metric]) for row in group])
            )
        output.append(summary)
    return output


def _final_evaluation_rows(
    runs: tuple[Any, ...], rows: list[TableRow]
) -> list[TableRow]:
    boundaries = {
        run.directory.run_id: int(run.completion["training_interactions"])
        for run in runs
    }
    return [
        row
        for row in rows
        if int(row["training_interactions"]) == boundaries[str(row["run_id"])]
    ]


def _write_table(
    output: Path, name: str, rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    json_path = output / f"{name}.json"
    csv_path = output / f"{name}.csv"
    _write_json(
        json_path,
        {"schema_version": ANALYSIS_SCHEMA_VERSION, "rows": rows},
    )
    flattened = [_flatten(row) for row in rows]
    fields = sorted({key for row in flattened for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in flattened:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    return json_path, csv_path


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in sorted(data.items()):
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            output.update(_flatten(value, name))
        else:
            output[name] = value
    return output


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, data: Any) -> None:
    payload = json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.write_text(f"{payload}\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("geometry specification must be a JSON object.")
    return data


def _number_tuple(data: dict[str, Any], key: str) -> tuple[float, ...]:
    values = data.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in values
    ):
        raise ValueError(f"geometry specification {key} must be a numeric list.")
    result = tuple(float(value) for value in values)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"geometry specification {key} must be strictly increasing.")
    return result


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_sort_key(row: TableRow) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in sorted(row))


if __name__ == "__main__":
    raise SystemExit(main())
