"""Run the learning configuration check from `docs/EXPERIMENT.md`.

The course equations do not determine learning rates, so the protocol picks
each algorithm's rate from a small finite grid by a rule written down before the
runs happen. This executes that grid and applies that rule.

It is a *pre-experiment* check: its runs are development evidence, never
observations in either reported experiment, and they are written under the
pre-experiment category so the recording schema refuses to pool them with
reported results.

    python experiments/calibrate_learning_rates.py run
    python experiments/calibrate_learning_rates.py select

`run` is resumable and contract-checked, so it can be interrupted, and a run
recorded under superseded constants is re-run rather than reused.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from matrix import RunSpecification, execute, learning_contract, summarize
from train import run_a2c_training, run_ppo_training, run_reinforce_training

from configs import (
    MEDIUM_ACTOR_CONFIG,
    A2CConfig,
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    PPOConfig,
    ReinforceConfig,
    physical_cpu_count,
)
from recording import RunCategory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_PATH = PROJECT_ROOT / "tracks" / "experiment_1.json"
RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / RunCategory.PRE_EXPERIMENT.value
    / "learning_configuration_check"
)

# Three dedicated roots, as the protocol specifies.
ROOTS = (0, 1, 2)

# The allowance is per algorithm. A2C's was raised to 750,000 by the 2026-08-12
# amendment because all four of its candidates were indistinguishable at
# 250,000, and a selection between indistinguishable candidates is a coin flip.
ALLOWANCE = {"reinforce": 250_000, "a2c": 750_000, "ppo": 250_000}
EVALUATION_INTERVAL = 25_000


@dataclass(frozen=True, slots=True)
class Candidate:
    """
    One grid point: an algorithm and the rates it is being tried at.

    Fields:
        * algorithm: Which learner the rates belong to.
        * actor_rate: Actor optimizer step size.
        * critic_rate: Critic step size, or `None` for actor-only REINFORCE.
    """

    algorithm: str
    actor_rate: float
    critic_rate: float | None

    @property
    def name(self) -> str:
        """
        Return a filesystem-safe identity for this grid point.
        """
        critic = "none" if self.critic_rate is None else f"{self.critic_rate:g}"
        return f"{self.algorithm}-actor-{self.actor_rate:g}-critic-{critic}"


# The actor-critic pairs. The 2026-08-20 amendment added the three at actor
# 1e-3: the original grid capped both actor-critic algorithms at 3e-4 while
# offering REINFORCE 1e-3, so A2C was never allowed the rate that turned out to
# be the one it needed. The grid stays shared between A2C and PPO, as the
# protocol requires, so neither is offered an option the other is denied.
ACTOR_CRITIC_PAIRS: tuple[tuple[float, float], ...] = (
    (1e-4, 3e-4),
    (3e-4, 1e-3),
    (3e-4, 3e-3),
    (3e-4, 1e-2),
    (1e-3, 1e-3),
    (1e-3, 3e-3),
    (1e-3, 1e-2),
)

CANDIDATES: tuple[Candidate, ...] = (
    *(Candidate("reinforce", rate, None) for rate in (1e-4, 3e-4, 1e-3)),
    *(
        Candidate(algorithm, actor, critic)
        for algorithm in ("a2c", "ppo")
        for actor, critic in ACTOR_CRITIC_PAIRS
    ),
)


def specifications() -> list[RunSpecification]:
    """
    Enumerate every run of the grid, without starting any of them.
    """
    execution = ExecutionConfig(environment_workers=physical_cpu_count())
    steering = LoggingConfig().near_saturated_steering_threshold
    runs: list[RunSpecification] = []
    for candidate in CANDIDATES:
        for root in ROOTS:
            run_id = f"{candidate.name}-seed-{root}"
            path = RESULTS_ROOT / run_id
            common: dict[str, Any] = {
                "seed": root,
                "track_path": TRACK_PATH,
                "run_path": path,
                # The grid is run at the medium actor only; the chosen rate is
                # then used for all three sizes, so the size is not a factor.
                "actor_config": MEDIUM_ACTOR_CONFIG,
                "actor_learning_rate": candidate.actor_rate,
                "training_interaction_budget": ALLOWANCE[candidate.algorithm],
                "evaluation_interval": EVALUATION_INTERVAL,
                "execution_config": execution,
                "near_saturated_steering_threshold": steering,
                "run_category": RunCategory.PRE_EXPERIMENT,
            }
            if candidate.algorithm == "reinforce":
                launch = partial(run_reinforce_training, **common)
            else:
                runner = (
                    run_a2c_training
                    if candidate.algorithm == "a2c"
                    else run_ppo_training
                )
                launch = partial(
                    runner, critic_learning_rate=candidate.critic_rate, **common
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


def _final_rows(run: Path) -> list[dict[str, Any]]:
    """
    Return the evaluations recorded at a run's final interaction count.
    """
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    final = int(completion["training_interactions"])
    rows = [
        json.loads(line)
        for line in (run / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [row for row in rows if int(row["training_interactions"]) == final]


def outcomes() -> list[dict[str, Any]]:
    """
    Score every candidate by the recorded rule, from the raw run records.

    The rule is lexicographic: laps completed, then mean final progress with
    each run clamped to one, then mean final return, then the smaller rates.
    Clamping matters because a completed lap overshoots by wherever its last
    step landed, and that overshoot measures the step rather than the driving.
    """
    scored: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        laps = 0
        progress: list[float] = []
        returns: list[float] = []
        missing: list[str] = []
        for root in ROOTS:
            run = RESULTS_ROOT / f"{candidate.name}-seed-{root}"
            if not (run / "completion.json").is_file():
                missing.append(str(root))
                continue
            for row in _final_rows(run):
                episode = row.get("episode", row)
                laps += int(episode["outcome"] == "completed")
                progress.append(min(1.0, float(episode["maximum_progress"])))
                returns.append(float(episode["undiscounted_return"]))
        scored.append(
            {
                "algorithm": candidate.algorithm,
                "actor_rate": candidate.actor_rate,
                "critic_rate": candidate.critic_rate,
                "allowance": ALLOWANCE[candidate.algorithm],
                "roots": len(ROOTS) - len(missing),
                "missing_roots": ",".join(missing),
                "completed_laps": laps,
                "mean_progress": sum(progress) / len(progress) if progress else 0.0,
                "mean_return": sum(returns) / len(returns) if returns else 0.0,
            }
        )
    return scored


def selected(scored: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Apply the lexicographic rule within each algorithm.
    """
    winners: dict[str, dict[str, Any]] = {}
    for algorithm in ("reinforce", "a2c", "ppo"):
        rows = [row for row in scored if row["algorithm"] == algorithm]
        if not rows:
            continue
        winners[algorithm] = max(
            rows,
            key=lambda row: (
                row["completed_laps"],
                row["mean_progress"],
                row["mean_return"],
                -float(row["actor_rate"]),
                -float(row["critic_rate"] or 0.0),
            ),
        )
    return winners


def report(scored: list[dict[str, Any]]) -> None:
    """
    Print the grid and the selection as a table that can be pasted into a report.

    Deliberately ASCII. A Windows console encodes stdout as cp1252, so an
    em-dash for the absent critic rate arrives as a replacement character in
    exactly the log a reader would copy the table out of.
    """
    winners = selected(scored)
    header = ("Algorithm", "Actor", "Critic", "Allowance", "Laps", "Progress", "Return")
    widths = (13, 8, 8, 9, 5, 8, 9)
    print(
        "| "
        + " | ".join(f"{name:<{width}}" for name, width in zip(header, widths))
        + " |"
    )
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for row in scored:
        chosen = winners.get(str(row["algorithm"])) is row
        name = f"**{row['algorithm']}**" if chosen else str(row["algorithm"])
        critic = "-" if row["critic_rate"] is None else f"{row['critic_rate']:g}"
        cells = (
            f"{name:<13}",
            f"{row['actor_rate']:>8g}",
            f"{critic:>8}",
            f"{row['allowance']:>9,}",
            f"{row['completed_laps']:>2}/{row['roots']:<2}",
            f"{row['mean_progress']:>8.3f}",
            f"{row['mean_return']:>9.2f}",
        )
        print("| " + " | ".join(cells) + " |")
    print()
    for algorithm, row in winners.items():
        critic = "-" if row["critic_rate"] is None else f"{row['critic_rate']:g}"
        print(f"selected {algorithm}: actor {row['actor_rate']:g}, critic {critic}")


def main() -> int:
    """
    Run the grid or score it, depending on the requested command.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "select", "plan"))
    parsed = parser.parse_args()

    runs = specifications()
    if parsed.command == "plan":
        for run in runs:
            print(run.run_id)
        print(f"\n{len(runs)} runs -> {RESULTS_ROOT}")
        return 0
    if parsed.command == "run":
        failures = summarize(execute(runs, contract=contract()))
        if failures:
            return 1

    scored = outcomes()
    report(scored)
    (RESULTS_ROOT / "selection.json").write_text(
        json.dumps({"grid": scored, "selected": selected(scored)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
