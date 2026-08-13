"""Deterministic plots generated from root-level analysis tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .analysis import TableRow

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_learning_curves(rows: list[TableRow]) -> Figure:
    """
    Plot root-mean return and progress with sample-standard-deviation bands.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, condition_rows in _condition_groups(rows).items():
        by_boundary: dict[int, list[TableRow]] = defaultdict(list)
        for row in condition_rows:
            by_boundary[int(row["training_interactions"])].append(row)
        x = np.asarray(sorted(by_boundary), dtype=np.float64)
        for axis, metric, title in (
            (axes[0], "mean_return", "Evaluation return"),
            (axes[1], "mean_progress", "Normalized progress"),
        ):
            means = np.asarray(
                [
                    np.mean([float(row[metric]) for row in by_boundary[int(point)]])
                    for point in x
                ]
            )
            deviations = np.asarray(
                [
                    _sample_standard_deviation(
                        [float(row[metric]) for row in by_boundary[int(point)]]
                    )
                    for point in x
                ]
            )
            line = axis.plot(x, means, label=label)[0]
            axis.fill_between(
                x,
                means - deviations,
                means + deviations,
                color=line.get_color(),
                alpha=0.18,
            )
            axis.set_title(title)
            axis.set_xlabel("Training interactions")
    axes[1].set_ylim(top=max(1.0, axes[1].get_ylim()[1]))
    axes[0].set_ylabel("Root mean ± sample SD")
    axes[1].legend(fontsize="small")
    return figure


def plot_task_outcomes(
    rows: list[dict[str, Any]], root_rows: list[TableRow] | None = None
) -> Figure:
    """
    Plot final completion, progress, and return for each experiment cell.
    """
    ordered = sorted(rows, key=_cell_label)
    labels = [_cell_label(row) for row in ordered]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    metrics = (
        ("final_completion_rate", "Completion rate"),
        ("final_mean_progress", "Final progress"),
        ("final_mean_return", "Final return"),
    )
    x = np.arange(len(ordered))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        means = [float(row[metric]["mean"]) for row in ordered]
        lower = [
            mean - float(row[metric]["confidence_interval_low"])
            for mean, row in zip(means, ordered, strict=True)
        ]
        upper = [
            float(row[metric]["confidence_interval_high"]) - mean
            for mean, row in zip(means, ordered, strict=True)
        ]
        axis.errorbar(x, means, yerr=(lower, upper), fmt="o", capsize=3)
        if root_rows is not None:
            cell_positions = {label: index for index, label in enumerate(labels)}
            for root in root_rows:
                position = next(
                    (
                        cell_positions[label]
                        for label in _cell_label_candidates(root)
                        if label in cell_positions
                    ),
                    None,
                )
                if position is not None:
                    axis.scatter(
                        position,
                        float(root[metric]),
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=3,
                    )
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    return figure


def plot_convergence_resources(rows: list[TableRow]) -> Figure:
    """
    Plot final performance and convergence cost against actor parameter count.
    """
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for label, condition_rows in _condition_groups(rows).items():
        parameters = np.asarray(
            [float(row["actor_parameters"]) for row in condition_rows]
        )
        progress = np.asarray(
            [float(row["final_mean_progress"]) for row in condition_rows]
        )
        convergence = np.asarray(
            [
                float(row["restricted_convergence_interactions"])
                for row in condition_rows
            ]
        )
        duration = np.asarray(
            [float(row["end_to_end_duration"]) for row in condition_rows]
        )
        axes[0].scatter(parameters, progress, label=label, alpha=0.8)
        axes[1].scatter(parameters, convergence, label=label, alpha=0.8)
        axes[2].scatter(parameters, duration, label=label, alpha=0.8)
    axes[0].set_title("Performance versus capacity")
    axes[0].set_ylabel("Final normalized progress")
    axes[1].set_title("Convergence or censoring boundary")
    axes[1].set_ylabel("Training interactions")
    axes[2].set_title("End-to-end runtime")
    axes[2].set_ylabel("Seconds")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Actor parameters")
    axes[2].legend(fontsize="small")
    return figure


def plot_optimization_diagnostics(rows: list[TableRow]) -> Figure:
    """
    Plot recorded loss, gradient dispersion, critic fit, and PPO clipping.

    The two dispersion panels are the ones that speak to estimator variance
    rather than to optimization progress. Prefer the cosine when comparing
    algorithms: the ratio is dominated by whichever sub-batch gradient was
    largest, and REINFORCE's magnitudes are heavy-tailed.
    """
    figure, axes = plt.subplots(2, 5, figsize=(19, 7), constrained_layout=True)
    panels = (
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("actor_gradient_norm", "Actor gradient norm"),
        ("entropy_proxy", "Entropy proxy"),
        ("gradient_cosine_similarity", "Gradient agreement (cosine)"),
        ("gradient_signal_to_noise", "Gradient signal-to-noise"),
        ("explained_variance", "Explained variance"),
        ("approximate_kl", "Approximate KL"),
        ("clip_fraction", "PPO clip fraction"),
        ("ratio_mean", "PPO ratio mean"),
    )
    for label, condition_rows in _condition_groups(rows).items():
        by_boundary: dict[int, list[TableRow]] = defaultdict(list)
        for row in condition_rows:
            by_boundary[int(row["training_interactions"])].append(row)
        for axis, (metric, title) in zip(axes.flat, panels, strict=True):
            points: list[tuple[int, float]] = []
            for boundary, boundary_rows in sorted(by_boundary.items()):
                values = [
                    float(row[metric])
                    for row in boundary_rows
                    if row.get(metric) is not None
                ]
                if values:
                    points.append((boundary, float(np.mean(values))))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [float(point[1]) for point in points],
                    label=label,
                    alpha=0.75,
                )
            axis.set_title(title)
            axis.set_xlabel("Training interactions")
    axes[0, 0].legend(fontsize="small")
    return figure


def plot_curvature_controls(rows: list[TableRow]) -> Figure:
    """
    Plot speed, throttle, and steering by curvature quartile without outcome filtering.
    """
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    bins = ("q1", "q2", "q3", "q4")
    for label, condition_rows in _condition_groups(rows).items():
        for axis, metric, title in (
            (axes[0], "mean_speed", "Speed"),
            (axes[1], "mean_throttle", "Throttle / brake"),
            (axes[2], "mean_absolute_steering", "Absolute steering"),
        ):
            values: list[float] = []
            for curvature_bin in bins:
                selected = [
                    row
                    for row in condition_rows
                    if row["curvature_bin"] == curvature_bin
                ]
                weights = [int(row["sample_count"]) for row in selected]
                values.append(
                    float(
                        np.average(
                            [float(row[metric]) for row in selected],
                            weights=weights,
                        )
                    )
                    if selected
                    else np.nan
                )
            axis.plot(bins, values, marker="o", label=label)
            axis.set_title(title)
            axis.set_xlabel("Absolute-curvature quartile")
    axes[0].legend(fontsize="small")
    return figure


def plot_circuit_geometry(rows: list[TableRow]) -> Figure:
    """
    Plot held-out progress against recorded circuit length and curvature.
    """
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    markers = {"completed": "o", "crashed": "x", "time_limit": "^"}
    for label, condition_rows in _condition_groups(rows).items():
        for outcome, marker in markers.items():
            selected = [row for row in condition_rows if row["outcome"] == outcome]
            if not selected:
                continue
            axes[0].scatter(
                [float(row["circuit_length"]) for row in selected],
                [float(row["maximum_progress"]) for row in selected],
                marker=marker,
                label=f"{label}/{outcome}",
                alpha=0.75,
            )
            axes[1].scatter(
                [float(row["curvature_q90"]) for row in selected],
                [float(row["maximum_progress"]) for row in selected],
                marker=marker,
                label=f"{label}/{outcome}",
                alpha=0.75,
            )
    axes[0].set_xlabel("Circuit length")
    axes[1].set_xlabel("90th-percentile absolute curvature")
    for axis in axes:
        axis.set_ylabel("Maximum normalized progress")
    axes[0].legend(fontsize="x-small")
    return figure


def save_figure(figure: Figure, path: str | Path) -> None:
    """
    Save one plot with fixed rendering metadata and close its resources.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=150,
        metadata={"Software": "reinforcement-learning-car-racing"},
    )
    plt.close(figure)


def _condition_groups(rows: list[TableRow]) -> dict[str, list[TableRow]]:
    groups: dict[str, list[TableRow]] = defaultdict(list)
    for row in rows:
        groups[_condition_label(row)].append(row)
    return dict(sorted(groups.items()))


def _condition_label(row: TableRow) -> str:
    observation = row.get("observation_type")
    algorithm = row.get("algorithm")
    actor = row.get("actor_name")
    return "/".join(str(value) for value in (algorithm, actor, observation) if value)


def _cell_label(row: dict[str, Any]) -> str:
    return "/".join(
        str(row[key])
        for key in ("algorithm", "actor_name", "observation_type")
        if row.get(key) is not None
    )


def _cell_label_candidates(row: dict[str, Any]) -> tuple[str, ...]:
    """
    Return labels for complete, actor-size, and observation-only cell designs.
    """
    return (
        _cell_label(row),
        "/".join(str(row[key]) for key in ("algorithm", "actor_name") if row.get(key)),
        str(row.get("observation_type", "")),
    )


def _sample_standard_deviation(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
