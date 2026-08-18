"""Summarize the discount-horizon grid, paired by seed.

Two things decide how this reads the runs.

The first is that a single checkpoint says very little. Evaluation here is one
deterministic episode per checkpoint, so a run that is learning steadily still
produces a jagged curve, and reading the last point alone measures where the
jag landed. Every end-of-training figure below is therefore the mean over the
final quarter of checkpoints.

The second is pairing. Runs of this task differ enormously from seed to seed —
far more than the discount is expected to move anything — so the quantity that
carries the signal is the *within-pair difference*, and the spread across seeds
within one condition is reported beside it as the scale that difference has to
beat.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from configs import Algorithm

# Fraction of the run treated as "the end", for both evaluations and updates.
TAIL_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class RunSummary:
    """
    One run reduced to the numbers the comparison rests on.

    Fields:
        * name: Directory name of the run.
        * algorithm: Learning algorithm.
        * discount: The discount used.
        * seed: Run seed.
        * metrics: Scalar metrics, absent where a run cannot supply one.
        * curve_interactions: Checkpoint positions of the evaluation curve.
        * curve_progress: Evaluation progress at each checkpoint.
        * curve_return: Evaluation return at each checkpoint.
    """

    name: str
    algorithm: Algorithm
    discount: float
    seed: int
    metrics: dict[str, float]
    curve_interactions: list[float]
    curve_progress: list[float]
    curve_return: list[float]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read one JSON-lines file, or nothing if it does not exist.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tail(values: Sequence[float]) -> list[float]:
    """
    Return the final quarter of a sequence, and never fewer than one value.
    """
    if not values:
        return []
    count = max(1, round(len(values) * TAIL_FRACTION))
    return list(values[-count:])


def _mean(values: Sequence[float]) -> float | None:
    """
    Return the mean, or nothing when there is nothing to average.
    """
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def summarize_run(
    path: Path, algorithm: Algorithm, discount: float, seed: int
) -> RunSummary:
    """
    Reduce one run directory to its metrics and its evaluation curve.
    """
    evaluations = _read_jsonl(path / "evaluations.jsonl")
    updates = _read_jsonl(path / "updates.jsonl")
    episodes = _read_jsonl(path / "episodes.jsonl")

    interactions = [float(row["training_interactions"]) for row in evaluations]
    progress = [float(row["episode"]["final_progress"]) for row in evaluations]
    returns = [float(row["episode"]["undiscounted_return"]) for row in evaluations]
    outcomes = [str(row["episode"]["outcome"]) for row in evaluations]
    lap_times = [
        float(row["episode"]["lap_time"])
        for row in evaluations
        if row["episode"].get("lap_time") is not None
    ]

    # Mean speed says directly what a lap time says indirectly, and it is
    # defined on every episode rather than only on the completed ones.
    speeds = [
        float(row["episode"]["speed"]["mean"])
        for row in evaluations
        if isinstance(row["episode"].get("speed"), dict)
    ]

    metrics: dict[str, float] = {}
    _store(metrics, "final_speed", _mean(_tail(speeds)))
    _store(metrics, "final_progress", _mean(_tail(progress)))
    _store(metrics, "final_return", _mean(_tail(returns)))
    _store(metrics, "mean_progress", _mean(progress))
    tail_outcomes = _tail(
        [1.0 if outcome == "completed" else 0.0 for outcome in outcomes]
    )
    _store(metrics, "final_completion", _mean(tail_outcomes))
    if lap_times:
        # The fastest lap of a run is a minimum over forty noisy checkpoints, so
        # it partly measures how lucky the best checkpoint got. The mean over
        # the final quarter is the same question asked of a settled policy, and
        # it is the one the comparison should rest on.
        _store(metrics, "best_lap_time", min(lap_times))
        _store(metrics, "final_lap_time", _mean(_tail(lap_times)))

    # Where the policy first completes a lap, as a learning-speed measure that
    # does not depend on where the run happened to finish.
    for position, outcome in enumerate(outcomes):
        if outcome == "completed":
            _store(metrics, "first_completion", interactions[position])
            break

    _summarize_updates(metrics, updates)
    _summarize_episodes(metrics, episodes)

    return RunSummary(
        name=path.name,
        algorithm=algorithm,
        discount=discount,
        seed=seed,
        metrics=metrics,
        curve_interactions=interactions,
        curve_progress=progress,
        curve_return=returns,
    )


def _store(metrics: dict[str, float], name: str, value: float | None) -> None:
    """
    Record a metric only when the run actually produced one.
    """
    if value is not None and math.isfinite(value):
        metrics[name] = float(value)


def _summarize_updates(
    metrics: dict[str, float], updates: list[dict[str, Any]]
) -> None:
    """
    Add the optimizer-side diagnostics that show what the discount changed.

    These are the mechanism, not the outcome. A discount below one contracts the
    critic's bootstrap and shortens the span of the Monte Carlo return, so if it
    matters at all it should be visible here before it is visible in a lap time.
    """
    if not updates:
        return

    def column(name: str) -> list[float]:
        return [
            float(row[name])
            for row in updates
            if isinstance(row.get(name), (int, float))
        ]

    def diagnostic(name: str) -> list[float]:
        return [
            float(row["diagnostics"][name])
            for row in updates
            if isinstance(row.get("diagnostics", {}).get(name), (int, float))
        ]

    _store(metrics, "explained_variance", _mean(_tail(column("explained_variance"))))
    _store(metrics, "critic_loss", _mean(_tail(column("critic_loss"))))
    _store(
        metrics,
        "log_standard_deviation",
        _mean(_tail(column("log_standard_deviation"))),
    )
    _store(
        metrics,
        "gradient_signal_to_noise",
        _mean(_tail(diagnostic("gradient_signal_to_noise"))),
    )
    _store(
        metrics,
        "value_target_sd",
        _mean(_tail(diagnostic("value_target_standard_deviation"))),
    )
    _store(metrics, "value_target_mean", _mean(_tail(diagnostic("value_target_mean"))))
    _store(
        metrics,
        "advantage_sd",
        _mean(_tail(diagnostic("advantage_standard_deviation"))),
    )
    _store(metrics, "return_sd", _mean(_tail(diagnostic("return_standard_deviation"))))
    _store(metrics, "return_mean", _mean(_tail(diagnostic("return_mean"))))


def _summarize_episodes(
    metrics: dict[str, float], episodes: list[dict[str, Any]]
) -> None:
    """
    Add the training-episode outcome mix.

    The time-limit fraction is the one that matters for an undiscounted run:
    truncation is where a finite-horizon return has to be continued by a
    bootstrap rather than ended, and it is the only boundary at which the two
    discounts are not simply rescalings of each other.
    """
    training = [row for row in episodes if row.get("scope") == "training"] or episodes
    if not training:
        return
    outcomes = [str(row.get("outcome")) for row in training]
    total = float(len(outcomes))
    for outcome in ("time_limit", "completed", "crashed", "stalled"):
        _store(
            metrics,
            f"train_{outcome}_fraction",
            outcomes.count(outcome) / total,
        )
    _store(metrics, "train_episode_count", total)


def load_grid(
    output_root: Path,
    algorithms: Sequence[Algorithm],
    seeds: Sequence[int],
    discounts: Sequence[float] = (0.9995, 1.0),
) -> list[RunSummary]:
    """
    Load every finished run of the grid.
    """
    summaries: list[RunSummary] = []
    for algorithm in algorithms:
        for seed in seeds:
            for discount in discounts:
                label = "undiscounted" if discount == 1.0 else "discounted"
                path = output_root / f"{algorithm.value}_{label}_seed{seed}"
                if not (path / "completion.json").exists():
                    continue
                summaries.append(summarize_run(path, algorithm, discount, seed))
    return summaries


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """
    One metric compared within pairs that share a seed.

    Fields:
        * metric: Name of the metric compared.
        * pairs: Seed and its (discounted, undiscounted) values.
        * difference_mean: Mean of undiscounted minus discounted.
        * difference_sd: Sample standard deviation of those differences.
        * half_width: Half-width of the 95% interval on the mean difference.
        * seed_sd: Spread across seeds within a condition, pooled over both.
        * favouring_undiscounted: How many pairs the undiscounted run wins.
    """

    metric: str
    pairs: tuple[tuple[int, float, float], ...]
    difference_mean: float
    difference_sd: float
    half_width: float
    seed_sd: float
    favouring_undiscounted: int

    @property
    def relative_to_seed_noise(self) -> float:
        """
        Return the mean difference measured in units of seed-to-seed spread.

        This is the number that decides whether a difference means anything
        here. A shift much smaller than the spread between seeds cannot be
        acted on, however consistent its sign.
        """
        return self.difference_mean / self.seed_sd if self.seed_sd > 0.0 else math.nan


def compare_pairs(
    summaries: Sequence[RunSummary], metric: str
) -> PairedComparison | None:
    """
    Compare one metric within every complete pair of one algorithm.
    """
    by_seed: dict[int, dict[float, float]] = {}
    for summary in summaries:
        if metric in summary.metrics:
            by_seed.setdefault(summary.seed, {})[summary.discount] = summary.metrics[
                metric
            ]

    pairs = tuple(
        (seed, values[0.9995], values[1.0])
        for seed, values in sorted(by_seed.items())
        if 0.9995 in values and 1.0 in values
    )
    if len(pairs) < 2:
        return None

    discounted = np.asarray([pair[1] for pair in pairs])
    undiscounted = np.asarray([pair[2] for pair in pairs])
    differences = undiscounted - discounted
    count = differences.size
    difference_sd = float(np.std(differences, ddof=1))
    # Student t for a small paired sample; with five seeds this interval is
    # wide, and saying so is the point of reporting it.
    critical = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(
        count, 2.0
    )
    half_width = critical * difference_sd / math.sqrt(count) if count else math.nan
    seed_sd = float(
        math.sqrt(0.5 * (np.var(discounted, ddof=1) + np.var(undiscounted, ddof=1)))
    )
    return PairedComparison(
        metric=metric,
        pairs=pairs,
        difference_mean=float(np.mean(differences)),
        difference_sd=difference_sd,
        half_width=half_width,
        seed_sd=seed_sd,
        favouring_undiscounted=int(np.sum(differences > 0.0)),
    )


OUTCOME_METRICS = (
    "final_speed",
    "final_progress",
    "final_return",
    "final_completion",
    "mean_progress",
    "final_lap_time",
    "best_lap_time",
    "first_completion",
)
MECHANISM_METRICS = (
    "explained_variance",
    "critic_loss",
    "value_target_sd",
    "advantage_sd",
    "return_sd",
    "gradient_signal_to_noise",
    "log_standard_deviation",
    "train_time_limit_fraction",
    "train_completed_fraction",
)


def analyse(
    *,
    output_root: Path,
    algorithms: Sequence[Algorithm],
    seeds: Sequence[int],
    figure_path: Path = Path("outputs/discount_horizon.png"),
    summary_path: Path = Path("outputs/discount_horizon_summary.json"),
) -> int:
    """
    Print the paired comparison, write it as JSON, and draw the curves.
    """
    summaries = load_grid(output_root, algorithms, seeds)
    if not summaries:
        print(f"no finished runs under {output_root}")
        return 1

    report: dict[str, Any] = {"runs": len(summaries), "algorithms": {}}
    for algorithm in algorithms:
        of_algorithm = [row for row in summaries if row.algorithm is algorithm]
        if not of_algorithm:
            continue
        complete_pairs = len(
            {row.seed for row in of_algorithm if row.discount == 0.9995}
            & {row.seed for row in of_algorithm if row.discount == 1.0}
        )
        print(
            f"\n{'=' * 78}\n{algorithm.value.upper()}  ({complete_pairs} complete pairs)\n{'=' * 78}"
        )
        entries: dict[str, Any] = {}
        for group, metrics in (
            ("outcome", OUTCOME_METRICS),
            ("mechanism", MECHANISM_METRICS),
        ):
            printed_header = False
            for metric in metrics:
                comparison = compare_pairs(of_algorithm, metric)
                if comparison is None:
                    continue
                if not printed_header:
                    print(f"\n  {group}")
                    print(
                        f"    {'metric':<26} {'gamma=0.9995':>13} {'gamma=1':>11} "
                        f"{'difference':>12} {'95% CI':>17} {'d/seed sd':>10} {'wins':>5}"
                    )
                    printed_header = True
                discounted = float(np.mean([pair[1] for pair in comparison.pairs]))
                undiscounted = float(np.mean([pair[2] for pair in comparison.pairs]))
                print(
                    f"    {metric:<26} {discounted:>13.4g} {undiscounted:>11.4g} "
                    f"{comparison.difference_mean:>+12.4g} "
                    f"{f'+/- {comparison.half_width:.3g}':>17} "
                    f"{comparison.relative_to_seed_noise:>10.2f} "
                    f"{comparison.favouring_undiscounted:>3}/{len(comparison.pairs)}"
                )
                entries[metric] = {
                    "discounted_mean": discounted,
                    "undiscounted_mean": undiscounted,
                    "difference_mean": comparison.difference_mean,
                    "difference_sd": comparison.difference_sd,
                    "confidence_half_width": comparison.half_width,
                    "seed_sd": comparison.seed_sd,
                    "difference_over_seed_sd": comparison.relative_to_seed_noise,
                    "pairs_favouring_undiscounted": comparison.favouring_undiscounted,
                    "pair_count": len(comparison.pairs),
                    "per_seed": [
                        {"seed": seed, "discounted": low, "undiscounted": high}
                        for seed, low, high in comparison.pairs
                    ],
                }
        report["algorithms"][algorithm.value] = {
            "complete_pairs": complete_pairs,
            "metrics": entries,
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nsummary  {summary_path}")

    draw_curves(summaries, algorithms, figure_path)
    print(f"figure   {figure_path}")
    return 0


def draw_curves(
    summaries: Sequence[RunSummary],
    algorithms: Sequence[Algorithm],
    figure_path: Path,
) -> None:
    """
    Draw both conditions' learning curves and the diagnostic that explains them.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    present = [
        algorithm
        for algorithm in algorithms
        if any(row.algorithm is algorithm for row in summaries)
    ]
    figure, axes = plt.subplots(
        2,
        len(present),
        figsize=(5.2 * len(present), 8.0),
        constrained_layout=True,
        squeeze=False,
    )
    colours = {0.9995: "#00798c", 1.0: "#d1495b"}
    labels = {0.9995: "gamma = 0.9995", 1.0: "gamma = 1"}

    for column, algorithm in enumerate(present):
        for row, (attribute, title) in enumerate(
            (("curve_progress", "lap progress"), ("curve_return", "evaluation return"))
        ):
            axis = axes[row][column]
            for discount in (0.9995, 1.0):
                runs = [
                    run
                    for run in summaries
                    if run.algorithm is algorithm and run.discount == discount
                ]
                if not runs:
                    continue
                length = min(len(getattr(run, attribute)) for run in runs)
                if length == 0:
                    continue
                x = np.asarray(runs[0].curve_interactions[:length])
                stack = np.asarray([getattr(run, attribute)[:length] for run in runs])
                mean = stack.mean(axis=0)
                for series in stack:
                    axis.plot(
                        x, series, color=colours[discount], alpha=0.16, linewidth=0.8
                    )
                axis.plot(
                    x,
                    mean,
                    color=colours[discount],
                    linewidth=2.0,
                    label=labels[discount],
                )
            axis.set_title(f"{algorithm.value} — {title}", fontsize=10)
            axis.set_xlabel("training interactions")
            axis.grid(alpha=0.15)
            if column == 0 and row == 0:
                axis.legend(fontsize=8)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def draw_credit_weights(
    figure_path: Path = Path("outputs/discount_credit_weights.png"),
) -> None:
    """
    Draw what the discount actually changes in each algorithm's estimator.

    The two algorithms that use GAE weight a temporal-difference error k steps
    ahead by ``(gamma * lambda) ** k``, so the trace parameter already truncates
    the horizon long before the discount could: at lambda = 0.95 the span is
    twenty steps whether gamma is 0.9995 or one. REINFORCE has no such trace and
    weights a reward k steps later by ``gamma ** k`` over the whole episode,
    which is where a discount of 0.9995 does visible work.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from configs import A2CConfig, SimulationConfig

    lam = A2CConfig().gae_lambda
    horizon = SimulationConfig().max_episode_steps
    steps = np.arange(horizon + 1)
    colours = {0.9995: "#00798c", 1.0: "#d1495b"}

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for discount in (0.9995, 1.0):
        label = f"gamma = {discount:g}"
        axes[0].plot(
            steps[:81],
            (discount * lam) ** steps[:81],
            color=colours[discount],
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            steps, discount**steps, color=colours[discount], linewidth=2.0, label=label
        )

    axes[0].set_title(
        f"A2C and PPO: GAE weight (gamma*lambda)^k, lambda = {lam}\n"
        "the trace, not the discount, sets the span",
        fontsize=10,
    )
    axes[1].set_title(
        "REINFORCE: Monte Carlo weight gamma^k over a whole episode\n"
        "the only place the discount does visible work",
        fontsize=10,
    )
    for axis in axes:
        axis.set_xlabel("steps ahead k")
        axis.set_ylabel("weight")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=9)
    axes[1].axvline(horizon, color="#33363d", linestyle=":", linewidth=1.0)
    axes[1].annotate(
        f"T_max = {horizon}",
        xy=(horizon, 0.5),
        xytext=(-6.0, 0.0),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color="#33363d",
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


CONTROL_METRICS = (
    "final_speed",
    "final_lap_time",
    "final_return",
    "final_completion",
    "final_progress",
    "explained_variance",
    "critic_loss",
    "value_target_sd",
)


def analyse_control(
    *,
    output_root: Path,
    seeds: Sequence[int],
    critic_rates: Sequence[float],
) -> int:
    """
    Report whether a cooler critic recovers what the undiscounted PPO run lost.

    The reference rows are the two grid arms restricted to the same seeds, so
    every number in the table is an average over one set of runs and the
    comparison is not smuggling in seeds the control never trained on.
    """
    rows: list[tuple[str, list[RunSummary]]] = []
    for discount, label in ((0.9995, "gamma=0.9995"), (1.0, "gamma=1")):
        suffix = "undiscounted" if discount == 1.0 else "discounted"
        runs = [
            summarize_run(
                output_root / f"ppo_{suffix}_seed{seed}", Algorithm.PPO, discount, seed
            )
            for seed in seeds
            if (output_root / f"ppo_{suffix}_seed{seed}" / "completion.json").exists()
        ]
        rows.append((f"{label}, critic 1e-2 (grid)", runs))

    for rate in critic_rates:
        runs = [
            summarize_run(
                output_root / f"ppo_undiscounted_critic{rate:g}_seed{seed}",
                Algorithm.PPO,
                1.0,
                seed,
            )
            for seed in seeds
            if (
                output_root
                / f"ppo_undiscounted_critic{rate:g}_seed{seed}"
                / "completion.json"
            ).exists()
        ]
        rows.append((f"gamma=1, critic {rate:g}", runs))

    header = f"    {'condition':<32} {'n':>2}  " + "  ".join(
        f"{metric.replace('final_', ''):>12}" for metric in CONTROL_METRICS
    )
    print(f"\n{'=' * len(header)}\nPPO CRITIC-RATE CONTROL\n{'=' * len(header)}")
    print(header)
    for label, runs in rows:
        if not runs:
            print(f"    {label:<32} {0:>2}  (no finished runs)")
            continue
        values = []
        for metric in CONTROL_METRICS:
            present = [run.metrics[metric] for run in runs if metric in run.metrics]
            values.append(f"{np.mean(present):>12.4g}" if present else f"{'-':>12}")
        print(f"    {label:<32} {len(runs):>2}  " + "  ".join(values))
    return 0
