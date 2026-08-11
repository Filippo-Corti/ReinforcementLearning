"""Deterministic root-level analysis of recorded reinforcement-learning runs."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from recording import RunCategory, RunDirectory, RunRecordingError

Experiment = Literal[1, 2]
TableValue = Any
TableRow = dict[str, TableValue]


@dataclass(frozen=True, slots=True)
class RecordedRun:
    """
    Hold one validated complete run and every analysis-relevant record stream.

    Fields:
        * directory: Validated run-directory interface.
        * manifest: Immutable run identity and seed document.
        * config: Complete environment and learning configuration.
        * metadata: Recorded software, hardware, and repository context.
        * completion: Atomic completion, timing, and resource document.
        * episodes: Training episode records.
        * updates: Optimizer update records.
        * evaluations: Deterministic evaluation records.
        * trajectories: Selected deterministic trajectory documents.
        * data_checksum: Checksum over every authoritative analysis input.
    """

    directory: RunDirectory
    manifest: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]
    completion: dict[str, Any]
    episodes: tuple[dict[str, Any], ...]
    updates: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    trajectories: tuple[dict[str, Any], ...]
    data_checksum: str

    @property
    def algorithm(self) -> str:
        """
        Return the recorded algorithm identity.
        """
        value = self.manifest.get("algorithm")
        if not isinstance(value, str):
            raise RunRecordingError(f"run {self.directory.run_id} lacks an algorithm.")
        return value

    @property
    def root_identity(self) -> int:
        """
        Return the paired training-root identity.
        """
        value = self.manifest.get("root_seed")
        if not isinstance(value, int):
            raise RunRecordingError(
                f"run {self.directory.run_id} lacks an integer root identity."
            )
        return value

    @property
    def actor_name(self) -> str:
        """
        Return the named actor size stored in the training configuration.
        """
        training = _mapping(self.config, "training")
        actor = _mapping(training, "actor")
        value = actor.get("name")
        if not isinstance(value, str):
            raise RunRecordingError(f"run {self.directory.run_id} lacks actor name.")
        return value

    @property
    def observation_type(self) -> str:
        """
        Return the policy observation identity recorded with its outcomes.
        """
        for record in self.evaluations:
            episode = _mapping(record, "episode")
            value = episode.get("observation_type")
            if isinstance(value, str):
                return value
        for record in self.episodes:
            value = record.get("observation_type")
            if isinstance(value, str):
                return value
        raise RunRecordingError(
            f"run {self.directory.run_id} lacks an observation identity."
        )


@dataclass(frozen=True, slots=True)
class DescriptiveStatistics:
    """
    Describe independent root values without hiding their small sample size.

    Fields:
        * count: Number of independent roots.
        * mean: Arithmetic mean.
        * sample_standard_deviation: Sample standard deviation.
        * median: Median root value.
        * first_quartile: Root-level first quartile.
        * third_quartile: Root-level third quartile.
        * minimum: Smallest root value.
        * maximum: Largest root value.
        * confidence_interval_low: Exact root-bootstrap lower percentile.
        * confidence_interval_high: Exact root-bootstrap upper percentile.
    """

    count: int
    mean: float
    sample_standard_deviation: float
    median: float
    first_quartile: float
    third_quartile: float
    minimum: float
    maximum: float
    confidence_interval_low: float
    confidence_interval_high: float


def load_recorded_runs(
    root: str | Path,
    *,
    category: RunCategory = RunCategory.REPORTED,
) -> tuple[RecordedRun, ...]:
    """
    Load complete runs in canonical identity order, independent of discovery order.
    """
    root_path = Path(root)
    directories = sorted(
        (path.parent for path in root_path.rglob("manifest.json")),
        key=lambda path: path.as_posix(),
    )
    runs: list[RecordedRun] = []
    for directory in directories:
        run = RunDirectory.open(
            directory,
            expected_category=category,
            require_complete=True,
        )
        trajectories = tuple(
            _read_json(path)
            for path in sorted((directory / "trajectories").glob("*.json"))
        )
        runs.append(
            RecordedRun(
                directory=run,
                manifest=_read_json(directory / "manifest.json"),
                config=_read_json(directory / "config.json"),
                metadata=_read_json(directory / "metadata.json"),
                completion=run.require_complete(),
                episodes=tuple(run.records("episodes")),
                updates=tuple(run.records("updates")),
                evaluations=tuple(run.records("evaluations")),
                trajectories=trajectories,
                data_checksum=_run_data_checksum(directory),
            )
        )
    return tuple(sorted(runs, key=_run_sort_key))


def run_inventory_rows(runs: tuple[RecordedRun, ...]) -> list[dict[str, Any]]:
    """
    Preserve complete configuration, provenance, and resources beside each run.
    """
    return [
        {
            **_identity_row(run),
            "manifest_checksum": run.directory.manifest_checksum(),
            "manifest": run.manifest,
            "config": run.config,
            "metadata": run.metadata,
            "completion": run.completion,
        }
        for run in runs
    ]


def evaluation_rows(runs: tuple[RecordedRun, ...]) -> list[TableRow]:
    """
    Flatten deterministic outcomes while retaining root and circuit identities.
    """
    rows: list[TableRow] = []
    for run in runs:
        for record in run.evaluations:
            episode = _mapping(record, "episode")
            geometry = episode.get("circuit_geometry")
            curvature = (
                _mapping(_mapping(geometry, "absolute_curvature"), "quantiles")
                if isinstance(geometry, dict)
                else {}
            )
            rows.append(
                {
                    **_identity_row(run),
                    "evaluation_index": _integer(record, "evaluation_index"),
                    "training_interactions": _integer(record, "training_interactions"),
                    "training_duration": _number(record, "training_duration"),
                    "circuit_identity": _string(episode, "circuit_identity"),
                    "circuit_seed": _optional_integer(episode, "circuit_seed"),
                    "circuit_split": _optional_string(episode, "circuit_split"),
                    "outcome": _string(episode, "outcome"),
                    "return": _number(episode, "undiscounted_return"),
                    "final_progress": _number(episode, "final_progress"),
                    "maximum_progress": _number(episode, "maximum_progress"),
                    "lap_time": _optional_number(episode, "lap_time"),
                    "interactions": _integer(episode, "interactions"),
                    "simulated_time": _number(episode, "simulated_time"),
                    "circuit_length": (
                        _number(geometry, "track_length")
                        if isinstance(geometry, dict)
                        else None
                    ),
                    "curvature_q25": _optional_mapping_number(curvature, "q25"),
                    "curvature_q50": _optional_mapping_number(curvature, "q50"),
                    "curvature_q75": _optional_mapping_number(curvature, "q75"),
                    "curvature_q90": _optional_mapping_number(curvature, "q90"),
                    **_episode_signal_fields(episode),
                }
            )
    return sorted(rows, key=_table_sort_key)


def training_episode_rows(runs: tuple[RecordedRun, ...]) -> list[TableRow]:
    """
    Flatten every training episode without changing its run or circuit identity.
    """
    rows: list[TableRow] = []
    for run in runs:
        for record in run.episodes:
            if record.get("scope") != "training":
                continue
            rows.append(
                {
                    **_identity_row(run),
                    "episode_index": _integer(record, "episode_index"),
                    "training_interactions": _integer(record, "training_interactions"),
                    "evaluation_interactions": _integer(
                        record, "evaluation_interactions"
                    ),
                    "circuit_identity": _string(record, "circuit_identity"),
                    "circuit_seed": _optional_integer(record, "circuit_seed"),
                    "circuit_split": _optional_string(record, "circuit_split"),
                    "outcome": _string(record, "outcome"),
                    "return": _number(record, "undiscounted_return"),
                    "training_target_total": _optional_number(
                        record, "training_target_total"
                    ),
                    "interactions": _integer(record, "interactions"),
                    "simulated_time": _number(record, "simulated_time"),
                    "final_progress": _number(record, "final_progress"),
                    "maximum_progress": _number(record, "maximum_progress"),
                    "lap_time": _optional_number(record, "lap_time"),
                    **_episode_signal_fields(record),
                }
            )
    return sorted(rows, key=_table_sort_key)


def learning_curve_rows(
    runs: tuple[RecordedRun, ...], *, experiment: Experiment
) -> list[TableRow]:
    """
    Aggregate circuits within a root before exposing checkpoint learning curves.
    """
    rows = evaluation_rows(runs)
    grouped: dict[tuple[str, int], list[TableRow]] = defaultdict(list)
    for row in rows:
        if experiment == 2 and row["circuit_split"] != "validation":
            continue
        grouped[(str(row["run_id"]), int(row["training_interactions"]))].append(row)

    curves: list[TableRow] = []
    by_id = {run.directory.run_id: run for run in runs}
    for (run_id, interactions), group in sorted(grouped.items()):
        run = by_id[run_id]
        outcomes = [str(row["outcome"]) for row in group]
        lap_target_count = sum(
            row["outcome"] == "completed"
            and row["lap_time"] is not None
            and float(row["lap_time"]) <= 100.0
            for row in group
        )
        returns = _float_array(group, "return")
        progress = _float_array(group, "maximum_progress")
        durations = _float_array(group, "training_duration")
        curves.append(
            {
                **_identity_row(run),
                "training_interactions": interactions,
                "training_duration": float(np.max(durations)),
                "evaluation_count": len(group),
                "completion_rate": outcomes.count("completed") / len(group),
                "lap_target_rate": lap_target_count / len(group),
                "crash_rate": outcomes.count("crashed") / len(group),
                "mean_return": float(np.mean(returns)),
                "mean_progress": float(np.mean(progress)),
                "median_progress": float(np.median(progress)),
            }
        )
    return sorted(curves, key=_table_sort_key)


def run_summary_rows(
    runs: tuple[RecordedRun, ...], *, experiment: Experiment
) -> list[TableRow]:
    """
    Compute one independent summary row per training root.
    """
    curves = learning_curve_rows(runs, experiment=experiment)
    curves_by_run: dict[str, list[TableRow]] = defaultdict(list)
    for row in curves:
        curves_by_run[str(row["run_id"])].append(row)
    evaluations = evaluation_rows(runs)
    evaluations_by_run: dict[str, list[TableRow]] = defaultdict(list)
    for row in evaluations:
        evaluations_by_run[str(row["run_id"])].append(row)

    summaries: list[TableRow] = []
    for run in runs:
        run_curves = sorted(
            curves_by_run[run.directory.run_id],
            key=lambda row: int(row["training_interactions"]),
        )
        if not run_curves:
            raise RunRecordingError(
                f"run {run.directory.run_id} has no applicable evaluation curve."
            )
        if len(run_curves) < 3:
            raise RunRecordingError(
                f"run {run.directory.run_id} needs at least three evaluation checkpoints."
            )
        final_interactions = int(run.completion["training_interactions"])
        final_rows = [
            row
            for row in evaluations_by_run[run.directory.run_id]
            if int(row["training_interactions"]) == final_interactions
            and (experiment == 1 or row["circuit_split"] == "test")
        ]
        if not final_rows:
            raise RunRecordingError(
                f"run {run.directory.run_id} lacks final primary evaluation outcomes."
            )
        convergence = convergence_summary(run_curves, experiment=experiment)
        outcomes = [str(row["outcome"]) for row in final_rows]
        lap_times = [
            float(row["lap_time"]) for row in final_rows if row["lap_time"] is not None
        ]
        timing = _mapping(run.completion, "timing")
        resources = _mapping(run.completion, "resources")
        training_time = _number(timing, "training_only")
        window = run_curves[-3:]
        summaries.append(
            {
                **_identity_row(run),
                "final_evaluation_count": len(final_rows),
                "final_completion_count": outcomes.count("completed"),
                "final_completion_rate": outcomes.count("completed") / len(final_rows),
                "final_crash_count": outcomes.count("crashed"),
                "final_crash_rate": outcomes.count("crashed") / len(final_rows),
                "final_mean_return": float(np.mean(_float_array(final_rows, "return"))),
                "final_mean_progress": float(
                    np.mean(_float_array(final_rows, "maximum_progress"))
                ),
                "completed_lap_time_mean": (
                    float(np.mean(lap_times)) if lap_times else None
                ),
                "completed_lap_count": len(lap_times),
                "completed_lap_denominator": len(final_rows),
                "return_auc": normalized_curve_area(run_curves, "mean_return"),
                "progress_auc": normalized_curve_area(run_curves, "mean_progress"),
                **convergence,
                "episodes_to_convergence": _episodes_to_convergence(
                    run, convergence["convergence_interactions"]
                ),
                "restricted_episodes_to_convergence": _episodes_to_convergence(
                    run, convergence["restricted_convergence_interactions"]
                ),
                "final_window_return_change": float(
                    window[-1]["mean_return"] - window[0]["mean_return"]
                ),
                "final_window_return_range": float(
                    max(float(row["mean_return"]) for row in window)
                    - min(float(row["mean_return"]) for row in window)
                ),
                "final_return_minus_best": float(
                    run_curves[-1]["mean_return"]
                    - max(float(row["mean_return"]) for row in run_curves)
                ),
                "collection_duration": _number(timing, "collection"),
                "optimization_duration": _number(timing, "optimization"),
                "evaluation_duration": _number(timing, "evaluation"),
                "persistence_duration": _number(timing, "persistence"),
                "training_duration": training_time,
                "end_to_end_duration": _number(timing, "end_to_end"),
                "training_interactions": final_interactions,
                "collection_throughput": final_interactions
                / _number(timing, "collection"),
                "training_throughput": final_interactions / training_time,
                "actor_parameters": _integer(resources, "actor_parameters"),
                "critic_parameters": _optional_integer(resources, "critic_parameters"),
                "total_parameters": _integer(resources, "total_parameters"),
                "peak_process_memory": _optional_integer(
                    resources, "peak_process_memory"
                ),
                "peak_gpu_memory": _optional_integer(resources, "peak_gpu_memory"),
            }
        )
    return sorted(summaries, key=_table_sort_key)


def convergence_summary(curve: list[TableRow], *, experiment: Experiment) -> TableRow:
    """
    Find the first of three consecutive qualifying evaluation checkpoints.
    """
    ordered = sorted(curve, key=lambda row: int(row["training_interactions"]))
    qualified = [
        (
            float(row["lap_target_rate"]) == 1.0
            if experiment == 1
            else float(row["completion_rate"]) >= 0.75
            and float(row["median_progress"]) >= 0.95
        )
        for row in ordered
    ]
    for index in range(len(qualified) - 2):
        if all(qualified[index : index + 3]):
            first = ordered[index]
            return {
                "converged": True,
                "censored": False,
                "convergence_interactions": int(first["training_interactions"]),
                "convergence_duration": float(first["training_duration"]),
                "restricted_convergence_interactions": int(
                    first["training_interactions"]
                ),
                "restricted_convergence_duration": float(first["training_duration"]),
            }
    final = ordered[-1]
    return {
        "converged": False,
        "censored": True,
        "convergence_interactions": None,
        "convergence_duration": None,
        "restricted_convergence_interactions": int(final["training_interactions"]),
        "restricted_convergence_duration": float(final["training_duration"]),
    }


def normalized_curve_area(curve: list[TableRow], metric: str) -> float:
    """
    Return trapezoidal area divided by the recorded evaluation interaction span.
    """
    ordered = sorted(curve, key=lambda row: int(row["training_interactions"]))
    if len(ordered) < 2:
        raise ValueError("normalized curve area requires at least two evaluations.")
    interactions = _float_array(ordered, "training_interactions")
    if len(np.unique(interactions)) != len(interactions):
        raise ValueError("a root learning curve cannot repeat an interaction boundary.")
    values = _float_array(ordered, metric)
    return float(
        np.trapezoid(values, interactions) / (interactions[-1] - interactions[0])
    )


def descriptive_statistics(values: list[float]) -> DescriptiveStatistics:
    """
    Compute root dispersion and an exhaustive percentile bootstrap interval.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("descriptive values must be one non-empty finite vector.")
    if len(array) > 6:
        raise ValueError("exhaustive root bootstrap supports at most six roots.")
    bootstrap_means = np.fromiter(
        (
            float(np.mean(array[list(indices)]))
            for indices in itertools.product(range(len(array)), repeat=len(array))
        ),
        dtype=np.float64,
    )
    return DescriptiveStatistics(
        count=len(array),
        mean=float(np.mean(array)),
        sample_standard_deviation=(
            float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        ),
        median=float(np.median(array)),
        first_quartile=float(np.quantile(array, 0.25)),
        third_quartile=float(np.quantile(array, 0.75)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        confidence_interval_low=float(np.quantile(bootstrap_means, 0.025)),
        confidence_interval_high=float(np.quantile(bootstrap_means, 0.975)),
    )


def cell_summary_rows(
    summaries: list[TableRow], *, experiment: Experiment
) -> list[dict[str, Any]]:
    """
    Summarize root-level cells and retain explicit completion denominators.
    """
    keys = ("algorithm", "actor_name") if experiment == 1 else ("observation_type",)
    grouped: dict[tuple[TableValue, ...], list[TableRow]] = defaultdict(list)
    for row in summaries:
        grouped[tuple(row[key] for key in keys)].append(row)
    metrics = (
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
        "collection_throughput",
        "training_throughput",
    )
    output: list[dict[str, Any]] = []
    for identity, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        result: dict[str, Any] = dict(zip(keys, identity, strict=True))
        result["root_count"] = len(rows)
        result["converged_root_count"] = sum(bool(row["converged"]) for row in rows)
        result["censored_root_count"] = sum(bool(row["censored"]) for row in rows)
        result["final_completed_root_count"] = sum(
            float(row["final_completion_rate"]) == 1.0 for row in rows
        )
        result["completed_lap_count"] = sum(
            int(row["completed_lap_count"]) for row in rows
        )
        result["completed_lap_denominator"] = sum(
            int(row["completed_lap_denominator"]) for row in rows
        )
        lap_times = [
            float(row["completed_lap_time_mean"])
            for row in rows
            if row["completed_lap_time_mean"] is not None
        ]
        result["completed_lap_time"] = (
            asdict(descriptive_statistics(lap_times)) if lap_times else None
        )
        for metric in metrics:
            result[metric] = asdict(
                descriptive_statistics([float(row[metric]) for row in rows])
            )
        output.append(result)
    return output


def paired_difference_rows(
    summaries: list[TableRow],
    *,
    condition_key: str,
    left: str,
    right: str,
    fixed_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[TableRow]:
    """
    Pair conditions by root identity and fixed design factors, never row order.
    """
    indexed: dict[tuple[TableValue, ...], TableRow] = {}
    for row in summaries:
        key = tuple(row[name] for name in fixed_keys) + (
            row["root_identity"],
            row[condition_key],
        )
        if key in indexed:
            raise ValueError(f"duplicate paired run identity: {key}.")
        indexed[key] = row
    identities = sorted(
        {
            tuple(row[name] for name in fixed_keys) + (row["root_identity"],)
            for row in summaries
        },
        key=str,
    )
    differences: list[TableRow] = []
    for identity in identities:
        left_row = indexed.get(identity + (left,))
        right_row = indexed.get(identity + (right,))
        if left_row is None or right_row is None:
            continue
        result: TableRow = dict(zip(fixed_keys, identity[:-1], strict=True))
        result.update(
            {
                "root_identity": identity[-1],
                "contrast": f"{left}_minus_{right}",
            }
        )
        for metric in metrics:
            result[metric] = float(left_row[metric]) - float(right_row[metric])
        differences.append(result)
    return sorted(differences, key=_table_sort_key)


def paired_circuit_difference_rows(rows: list[TableRow]) -> list[TableRow]:
    """
    Pair final Frenet and LiDAR circuit outcomes by root and circuit identity.
    """
    indexed: dict[tuple[int, str, str | None, str], TableRow] = {}
    for row in rows:
        key = (
            int(row["root_identity"]),
            str(row["circuit_identity"]),
            None if row["circuit_split"] is None else str(row["circuit_split"]),
            str(row["observation_type"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate circuit pairing identity: {key}.")
        indexed[key] = row
    identities = sorted({key[:-1] for key in indexed}, key=str)
    output: list[TableRow] = []
    for identity in identities:
        frenet = indexed.get(identity + ("frenet",))
        lidar = indexed.get(identity + ("lidar",))
        if frenet is None or lidar is None:
            continue
        output.append(
            {
                "root_identity": identity[0],
                "circuit_identity": identity[1],
                "circuit_split": identity[2],
                "contrast": "frenet_minus_lidar",
                "completion": float(frenet["outcome"] == "completed")
                - float(lidar["outcome"] == "completed"),
                "crash": float(frenet["outcome"] == "crashed")
                - float(lidar["outcome"] == "crashed"),
                "return": float(frenet["return"]) - float(lidar["return"]),
                "maximum_progress": float(frenet["maximum_progress"])
                - float(lidar["maximum_progress"]),
            }
        )
    return sorted(output, key=_table_sort_key)


def representative_run_ids(
    summaries: list[TableRow], *, experiment: Experiment
) -> dict[tuple[TableValue, ...], str]:
    """
    Select the lower-root tie nearest each cell's median final return.
    """
    keys = ("algorithm", "actor_name") if experiment == 1 else ("observation_type",)
    grouped: dict[tuple[TableValue, ...], list[TableRow]] = defaultdict(list)
    for row in summaries:
        grouped[tuple(row[key] for key in keys)].append(row)
    selected: dict[tuple[TableValue, ...], str] = {}
    for identity, rows in grouped.items():
        median = float(np.median(_float_array(rows, "final_mean_return")))
        chosen = min(
            rows,
            key=lambda row: (
                abs(float(row["final_mean_return"]) - median),
                int(row["root_identity"]),
            ),
        )
        selected[identity] = str(chosen["run_id"])
    return selected


def curvature_control_rows(
    runs: tuple[RecordedRun, ...],
    summaries: list[TableRow],
    *,
    experiment: Experiment,
) -> list[TableRow]:
    """
    Aggregate every retained final-trajectory row into circuit curvature quartiles.
    """
    selected = set(representative_run_ids(summaries, experiment=experiment).values())
    output: list[TableRow] = []
    for run in runs:
        if run.directory.run_id not in selected:
            continue
        final_interactions = int(run.completion["training_interactions"])
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for trajectory in run.trajectories:
            evaluation = trajectory.get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            if _integer(evaluation, "training_interactions") != final_interactions:
                continue
            episode = _mapping(evaluation, "episode")
            transitions = trajectory.get("transitions")
            if not isinstance(transitions, list):
                raise RunRecordingError("trajectory transitions must be a list.")
            geometry = _mapping(episode, "circuit_geometry")
            quantiles = _mapping(_mapping(geometry, "absolute_curvature"), "quantiles")
            edges = (
                _number(quantiles, "q25"),
                _number(quantiles, "q50"),
                _number(quantiles, "q75"),
            )
            for transition in transitions:
                if not isinstance(transition, dict):
                    raise RunRecordingError("trajectory row must be an object.")
                curvature = abs(_number(transition, "current_curvature"))
                bin_index = int(np.searchsorted(edges, curvature, side="right"))
                grouped[(f"q{bin_index + 1}", _string(episode, "outcome"))].append(
                    transition
                )
        for (curvature_bin, outcome), transitions in sorted(grouped.items()):
            output.append(
                {
                    **_identity_row(run),
                    "curvature_bin": curvature_bin,
                    "outcome": outcome,
                    "sample_count": len(transitions),
                    "mean_speed": float(
                        np.mean([_number(row, "speed") for row in transitions])
                    ),
                    "mean_throttle": float(
                        np.mean(
                            [_number(row, "action", index=0) for row in transitions]
                        )
                    ),
                    "mean_absolute_steering": float(
                        np.mean(
                            [
                                abs(_number(row, "action", index=1))
                                for row in transitions
                            ]
                        )
                    ),
                }
            )
    return sorted(output, key=_table_sort_key)


def update_rows(runs: tuple[RecordedRun, ...]) -> list[TableRow]:
    """
    Flatten optimizer diagnostics with their independent run identities.
    """
    rows: list[TableRow] = []
    for run in runs:
        for record in run.updates:
            diagnostics = record.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                raise RunRecordingError("update diagnostics must be an object.")
            row: TableRow = {
                **_identity_row(run),
                "update_index": _integer(record, "update_index"),
                "training_interactions": _integer(record, "training_interactions"),
            }
            for name in (
                "actor_loss",
                "critic_loss",
                "actor_gradient_norm",
                "critic_gradient_norm",
                "optimization_duration",
                "actor_learning_rate",
                "critic_learning_rate",
                "entropy_proxy",
                "actor_weight_norm",
                "actor_update_norm",
                "critic_weight_norm",
                "critic_update_norm",
                "explained_variance",
                "approximate_kl",
                "clip_fraction",
            ):
                row[name] = _optional_number(record, name)
            log_standard_deviation = record.get("log_standard_deviation")
            if isinstance(log_standard_deviation, list):
                for index, value in enumerate(log_standard_deviation):
                    if not isinstance(value, int | float) or isinstance(value, bool):
                        raise RunRecordingError(
                            "log-standard-deviation values must be numeric."
                        )
                    row[f"log_standard_deviation_{index}"] = float(value)
            for name, value in diagnostics.items():
                if name in row:
                    continue
                if value is not None and (
                    not isinstance(value, int | float) or isinstance(value, bool)
                ):
                    raise RunRecordingError(
                        f"update diagnostic {name!r} must be numeric or null."
                    )
                row[name] = value
            rows.append(row)
    return sorted(rows, key=_table_sort_key)


def generalization_gap_rows(summaries_by_split: list[TableRow]) -> list[TableRow]:
    """
    Compute validation/test and training-reference/test gaps within each root.
    """
    indexed = {(row["run_id"], row["circuit_split"]): row for row in summaries_by_split}
    output: list[TableRow] = []
    for run_id in sorted({str(row["run_id"]) for row in summaries_by_split}):
        test = indexed.get((run_id, "test"))
        if test is None:
            continue
        for source in ("validation", "training_reference"):
            source_row = indexed.get((run_id, source))
            if source_row is None:
                continue
            output.append(
                {
                    **{
                        key: source_row[key]
                        for key in (
                            "run_id",
                            "algorithm",
                            "actor_name",
                            "observation_type",
                            "root_identity",
                        )
                    },
                    "contrast": f"{source}_minus_test",
                    "completion_rate": float(source_row["completion_rate"])
                    - float(test["completion_rate"]),
                    "crash_rate": float(source_row["crash_rate"])
                    - float(test["crash_rate"]),
                    "mean_return": float(source_row["mean_return"])
                    - float(test["mean_return"]),
                    "mean_progress": float(source_row["mean_progress"])
                    - float(test["mean_progress"]),
                }
            )
    return sorted(output, key=_table_sort_key)


def final_split_rows(runs: tuple[RecordedRun, ...]) -> list[TableRow]:
    """
    Aggregate final Experiment-2 circuits within root and split.
    """
    evaluations = evaluation_rows(runs)
    final_by_run = {
        run.directory.run_id: int(run.completion["training_interactions"])
        for run in runs
    }
    grouped: dict[tuple[str, str], list[TableRow]] = defaultdict(list)
    for row in evaluations:
        split = row["circuit_split"]
        if (
            split is None
            or int(row["training_interactions"]) != final_by_run[str(row["run_id"])]
        ):
            continue
        grouped[(str(row["run_id"]), str(split))].append(row)
    by_id = {run.directory.run_id: run for run in runs}
    output: list[TableRow] = []
    for (run_id, split), rows in sorted(grouped.items()):
        outcomes = [str(row["outcome"]) for row in rows]
        output.append(
            {
                **_identity_row(by_id[run_id]),
                "circuit_split": split,
                "circuit_count": len(rows),
                "completion_count": outcomes.count("completed"),
                "completion_rate": outcomes.count("completed") / len(rows),
                "crash_count": outcomes.count("crashed"),
                "crash_rate": outcomes.count("crashed") / len(rows),
                "mean_return": float(np.mean(_float_array(rows, "return"))),
                "mean_progress": float(np.mean(_float_array(rows, "maximum_progress"))),
            }
        )
    return sorted(output, key=_table_sort_key)


def stratify_circuit_geometry(
    rows: list[TableRow],
    *,
    length_edges: tuple[float, ...],
    curvature_edges: tuple[float, ...],
) -> list[TableRow]:
    """
    Apply explicitly supplied frozen circuit bins without creating pseudo-replicates.
    """
    output: list[TableRow] = []
    for row in rows:
        length = row.get("circuit_length")
        curvature = row.get("curvature_q90")
        if length is None or curvature is None:
            raise ValueError("geometry stratification requires length and curvature.")
        enriched = dict(row)
        enriched["length_bin"] = int(
            np.searchsorted(length_edges, float(length), side="right")
        )
        enriched["curvature_bin"] = int(
            np.searchsorted(curvature_edges, float(curvature), side="right")
        )
        output.append(enriched)
    return sorted(output, key=_table_sort_key)


def _identity_row(run: RecordedRun) -> TableRow:
    return {
        "run_id": run.directory.run_id,
        "run_checksum": run.data_checksum,
        "algorithm": run.algorithm,
        "actor_name": run.actor_name,
        "observation_type": run.observation_type,
        "root_identity": run.root_identity,
    }


def _episode_signal_fields(episode: dict[str, Any]) -> TableRow:
    fields: TableRow = {
        "positive_throttle_fraction": _optional_number(
            episode, "positive_throttle_fraction"
        ),
        "braking_fraction": _optional_number(episode, "braking_fraction"),
        "near_saturated_steering_fraction": _optional_number(
            episode, "near_saturated_steering_fraction"
        ),
    }
    for name in ("speed", "throttle", "absolute_steering"):
        summary = episode.get(name)
        if summary is None:
            continue
        if not isinstance(summary, dict):
            raise RunRecordingError(f"episode field {name!r} must be an object.")
        for statistic in ("mean", "standard_deviation", "minimum", "maximum"):
            fields[f"{name}_{statistic}"] = _number(summary, statistic)
        quantiles = _mapping(summary, "quantiles")
        for quantile, value in sorted(quantiles.items()):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise RunRecordingError(
                    f"episode {name} quantile {quantile!r} must be numeric."
                )
            fields[f"{name}_{quantile}"] = float(value)
    return fields


def _episodes_to_convergence(
    run: RecordedRun, convergence_interactions: TableValue
) -> int | None:
    if convergence_interactions is None:
        return None
    boundary = int(convergence_interactions)
    return sum(
        record.get("scope") == "training"
        and _integer(record, "training_interactions") <= boundary
        for record in run.episodes
    )


def _run_data_checksum(directory: Path) -> str:
    digest = hashlib.sha256()
    paths = [
        directory / name
        for name in (
            "manifest.json",
            "config.json",
            "metadata.json",
            "episodes.jsonl",
            "updates.jsonl",
            "evaluations.jsonl",
            "completion.json",
        )
    ] + sorted((directory / "trajectories").glob("*.json"))
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_sort_key(run: RecordedRun) -> tuple[str, str, str, int, str]:
    return (
        run.algorithm,
        run.actor_name,
        run.observation_type,
        run.root_identity,
        run.directory.run_id,
    )


def _table_sort_key(row: TableRow) -> tuple[str, ...]:
    keys = (
        "algorithm",
        "actor_name",
        "observation_type",
        "root_identity",
        "run_id",
        "training_interactions",
        "evaluation_index",
        "update_index",
        "circuit_split",
        "circuit_identity",
        "contrast",
        "curvature_bin",
        "outcome",
    )
    return tuple("" if row.get(key) is None else str(row.get(key)) for key in keys)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunRecordingError(f"invalid analysis input: {path}") from error
    if not isinstance(data, dict):
        raise RunRecordingError(f"analysis input is not an object: {path}")
    return data


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise RunRecordingError(f"record field {key!r} must be an object.")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise RunRecordingError(f"record field {key!r} must be a string.")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise RunRecordingError(f"record field {key!r} must be a string or null.")
    return value


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunRecordingError(f"record field {key!r} must be an integer.")
    return value


def _optional_integer(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise RunRecordingError(f"record field {key!r} must be an integer or null.")
    return value


def _number(data: dict[str, Any], key: str, *, index: int | None = None) -> float:
    value = data.get(key)
    if index is not None:
        if not isinstance(value, list) or index >= len(value):
            raise RunRecordingError(f"record field {key!r} lacks vector index {index}.")
        value = value[index]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RunRecordingError(f"record field {key!r} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise RunRecordingError(f"record field {key!r} must be finite.")
    return result


def _optional_number(data: dict[str, Any], key: str) -> float | None:
    return None if data.get(key) is None else _number(data, key)


def _optional_mapping_number(data: dict[str, Any], key: str) -> float | None:
    return None if data.get(key) is None else _number(data, key)


def _float_array(rows: list[TableRow], key: str) -> np.ndarray:
    values = [row[key] for row in rows]
    if any(value is None for value in values):
        raise ValueError(f"table metric {key!r} contains null values.")
    return np.asarray(values, dtype=np.float64)
