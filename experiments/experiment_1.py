"""Run Experiment 1: actor-network size on one fixed circuit.

The protocol is in `docs/EXPERIMENT.md` and the reported outcome is written up
in `docs/EXPERIMENT_1.md`. This file is the executable half: it builds the
matrix that document describes, runs it, and turns the recorded runs into the
tables and figures the write-up reads from.

    python experiments/experiment_1.py plan       # enumerate, run nothing
    python experiments/experiment_1.py run        # train the matrix
    python experiments/experiment_1.py analyze    # tables and figures only

`run` is resumable and contract-checked: an interrupted matrix continues where
it stopped, and a run recorded under superseded constants is re-run instead of
being silently reused. `--rehearsal` swaps the protocol budget for a short one
and writes under a different category and seed namespace, so a rehearsal can
neither be mistaken for a result nor share randomness with one.

Three algorithms times three actor sizes times five roots is 45 runs of two
million interactions, roughly nine hours on eight workers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from analyze_results import analyze_results
from matrix import RunSpecification, execute, learning_contract, summarize
from reporting import read_table
from train import run_a2c_training, run_ppo_training, run_reinforce_training

from configs import (
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    A2CConfig,
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    PPOConfig,
    ReinforceConfig,
    physical_cpu_count,
)
from recording import RunCategory
from utils.analysis import ppo_actor_selection_rows, selected_ppo_actor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_PATH = PROJECT_ROOT / "tracks" / "experiment_1.json"

ALGORITHMS = ("reinforce", "a2c", "ppo")

# The size ladder is the scientific factor. The critic is held at (64, 64) for
# A2C and PPO by `run_*_training`, so only the actor's capacity varies.
ACTORS = {
    "small": SMALL_ACTOR_CONFIG,
    "medium": MEDIUM_ACTOR_CONFIG,
    "large": LARGE_ACTOR_CONFIG,
}

# Selected before the experiment by the configuration check in EXPERIMENT.md.
# The rate travels with the algorithm rather than with the size: holding it
# fixed across the ladder is what makes the comparison about capacity instead
# of about tuning.
ACTOR_LEARNING_RATE = {"reinforce": 1e-3, "a2c": 1e-3, "ppo": 3e-4}
CRITIC_LEARNING_RATE = {"a2c": 3e-3, "ppo": 1e-2}


@dataclass(frozen=True, slots=True)
class Scale:
    """
    The budget, schedule and recording namespace one invocation runs under.

    Fields:
        * budget: Training interactions allowed per run.
        * evaluation_interval: Interactions between deterministic evaluations.
        * roots: Random roots, one run each.
        * category: Recording category, which also fixes the seed namespace.
    """

    budget: int
    evaluation_interval: int
    roots: tuple[int, ...]
    category: RunCategory

    @property
    def results_root(self) -> Path:
        """
        Return where this scale's runs are recorded.
        """
        return PROJECT_ROOT / "results" / self.category.value / "experiment_1"

    @property
    def analysis_root(self) -> Path:
        """
        Return where this scale's tables and figures are written.
        """
        return (
            PROJECT_ROOT / "results" / "analysis" / self.category.value / "experiment_1"
        )


PROTOCOL = Scale(2_000_000, 50_000, (0, 1, 2, 3, 4), RunCategory.REPORTED)
REHEARSAL = Scale(60_000, 5_000, (0, 1), RunCategory.REDUCED_VALIDATION)


def specifications(scale: Scale) -> list[RunSpecification]:
    """
    Enumerate every cell of the matrix, without starting any of them.
    """
    execution = ExecutionConfig(environment_workers=physical_cpu_count())
    steering = LoggingConfig().near_saturated_steering_threshold
    runs: list[RunSpecification] = []
    for algorithm in ALGORITHMS:
        for actor_name, actor_config in ACTORS.items():
            for root in scale.roots:
                run_id = f"{algorithm}-{actor_name}-frenet-seed-{root}"
                path = scale.results_root / run_id
                common: dict[str, Any] = {
                    "seed": root,
                    "track_path": TRACK_PATH,
                    "run_path": path,
                    "actor_config": actor_config,
                    "actor_learning_rate": ACTOR_LEARNING_RATE[algorithm],
                    "training_interaction_budget": scale.budget,
                    "evaluation_interval": scale.evaluation_interval,
                    "execution_config": execution,
                    "near_saturated_steering_threshold": steering,
                    "run_category": scale.category,
                }
                if algorithm == "reinforce":
                    launch = partial(run_reinforce_training, **common)
                else:
                    runner = (
                        run_a2c_training if algorithm == "a2c" else run_ppo_training
                    )
                    launch = partial(
                        runner,
                        critic_learning_rate=CRITIC_LEARNING_RATE[algorithm],
                        **common,
                    )
                runs.append(RunSpecification(run_id, path, launch))
    return runs


def contract() -> dict[str, Any]:
    """
    Return the constants a completed run must have been produced under.
    """
    return learning_contract(
        EnvironmentConfig(), ReinforceConfig(), A2CConfig(), PPOConfig()
    )


def analyze(scale: Scale) -> dict[str, Any]:
    """
    Write every table and figure, then record the PPO actor Experiment 2 uses.
    """
    manifest = analyze_results(
        results_root=scale.results_root,
        output_directory=scale.analysis_root,
        experiment=1,
        category=scale.category,
    )
    print(f"analyzed {len(manifest['inputs'])} runs -> {scale.analysis_root}")

    summaries = read_table(scale.analysis_root, "run_summaries")
    selection = ppo_actor_selection_rows(summaries)
    selected = selected_ppo_actor(selection)
    (scale.analysis_root / "ppo_actor_selection.json").write_text(
        json.dumps(
            {"selected_actor": selected, "candidates": selection},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(scale, summaries, selection, selected)
    return manifest


def _report(
    scale: Scale,
    summaries: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    selected: str,
) -> None:
    """
    Print the digest a person running this from a terminal wants to see.

    Deliberately ASCII: a Windows console encodes stdout as cp1252, and this is
    exactly the output a reader would copy a number out of.
    """
    cells = read_table(scale.analysis_root, "cell_summaries")
    header = (
        "Algorithm",
        "Actor",
        "Roots",
        "Completion",
        "Return",
        "SD",
        "95% interval",
        "Progress",
        "Converged",
        "Lap time",
    )
    widths = (11, 7, 5, 10, 9, 7, 18, 8, 9, 8)
    print()
    print("| " + " | ".join(f"{n:<{w}}" for n, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in sorted(cells, key=lambda row: (row["algorithm"], row["actor_name"])):
        # Every aggregate is a statistic rather than a number: a mean over five
        # roots without its spread is the one summary this experiment must not
        # report, because whether the roots agree is the question.
        completion = row["final_completion_rate"]
        returns = row["final_mean_return"]
        progress = row["final_mean_progress"]
        interval = (
            f"{returns['confidence_interval_low']:.1f}"
            f" to {returns['confidence_interval_high']:.1f}"
        )
        # Absent whenever a cell completed no lap: there is no lap to time.
        lap = row["completed_lap_time"]
        lap_time = "-" if lap is None else format(float(lap["mean"]), ".2f")
        print(
            f"| {row['algorithm']:<11} | {row['actor_name']:<7} "
            f"| {row['root_count']:<5} | {completion['mean']:<10.3f} "
            f"| {returns['mean']:<9.2f} "
            f"| {returns['sample_standard_deviation']:<7.2f} "
            f"| {interval:<18} | {progress['mean']:<8.3f} "
            f"| {row['converged_root_count']:<9} "
            f"| {lap_time:<8} |"
        )
    print()
    for row in selection:
        print(
            f"ppo {row['actor_name']:<7} return {row['mean_final_return']:>8.2f}  "
            f"deficit {row['mean_paired_deficit']:>7.2f}  "
            f"admitted {row['admitted']!s:<5}  selected {row['selected']}"
        )
    print(f"\nExperiment 2 will use the {selected!r} actor.")
    print(f"figures and tables: {scale.analysis_root}")
    print(f"runs analyzed: {len(summaries)}")


def main() -> int:
    """
    Plan, run or analyze the matrix at the requested scale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Short budget, two roots, written as reduced validation.",
    )
    parsed = parser.parse_args()
    scale = REHEARSAL if parsed.rehearsal else PROTOCOL

    runs = specifications(scale)
    print(f"budget {scale.budget:,} | roots {scale.roots} | {scale.category.value}")
    print(f"{len(runs)} runs -> {scale.results_root}")

    if parsed.command == "plan":
        for run in runs:
            state = "done" if (run.path / "completion.json").is_file() else "pending"
            print(f"  {state:<8} {run.run_id}")
        return 0

    if parsed.command == "run":
        failures = summarize(execute(runs, contract=contract()))
        if failures:
            return 1

    analyze(scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
