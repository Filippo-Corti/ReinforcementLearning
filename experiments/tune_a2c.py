"""Find an A2C configuration that learns, and record why the old one did not.

The learning configuration check selected A2C's rates from a grid that never
offered its actor the rate REINFORCE won with, and the selected candidate then
failed to complete a lap in three roots at 750,000 interactions. The recorded
diagnostics say why, and the reason is not the actor rate:

    value target spread      10.24
    value prediction spread   1.62
    explained variance       +0.07

The critic is predicting very nearly a constant. With a state-independent
critic the bootstrap cancels, `delta = r + V - V = r`, and the only thing left
setting the credit horizon is the GAE trace: `1/(1 - gamma*lambda)` is 20 agent
steps, or 0.8 seconds, against a lap of roughly 660 steps. The completion reward
then reaches almost nothing that earned it.

The critic is that bad because it takes **one** Adam step per 2048-transition
rollout -- 976 steps across a full run, where PPO's minibatch loop takes 124,928
from the same data. So this sweep moves the three knobs that are A2C's own
configuration and change how far its critic can travel:

* the actor rate, which the original grid capped below REINFORCE's winner;
* the critic rate; and
* the rollout length, which sets how many updates a fixed budget buys.

None of these touches shared actor-critic logic, so none of them implies a
matching change in PPO.

    python experiments/tune_a2c.py screen     # one root per candidate
    python experiments/tune_a2c.py confirm    # three roots for the shortlist
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from matrix import RunSpecification, execute, learning_contract, summarize
from train import run_a2c_training

from configs import (
    MEDIUM_ACTOR_CONFIG,
    A2CConfig,
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    physical_cpu_count,
)
from recording import RunCategory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_PATH = PROJECT_ROOT / "tracks" / "experiment_1.json"
RESULTS_ROOT = (
    PROJECT_ROOT / "results" / RunCategory.PRE_EXPERIMENT.value / "a2c_tuning"
)

BUDGET = 750_000
EVALUATION_INTERVAL = 25_000


@dataclass(frozen=True, slots=True)
class Candidate:
    """
    One A2C configuration to try.

    Fields:
        * actor_rate: Actor optimizer step size.
        * critic_rate: Critic optimizer step size.
        * rollout: Transitions collected before each single update.
    """

    actor_rate: float
    critic_rate: float
    rollout: int

    @property
    def name(self) -> str:
        """
        Return a filesystem-safe identity for this configuration.
        """
        return f"a2c-{self.actor_rate:g}-{self.critic_rate:g}-r{self.rollout}"

    @property
    def critic_steps(self) -> int:
        """
        Return how many optimizer steps the critic takes across a full run.
        """
        return 2_000_000 // self.rollout


# The first row is the currently selected configuration, carried so the sweep
# reports the thing it is trying to beat rather than assuming it.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(3e-4, 1e-2, 2048),
    Candidate(1e-3, 1e-2, 2048),
    Candidate(3e-4, 1e-2, 512),
    Candidate(1e-3, 1e-2, 512),
    Candidate(3e-4, 3e-2, 2048),
    Candidate(1e-3, 3e-2, 512),
    Candidate(3e-4, 1e-2, 256),
)

# Filled in from the screen; the shortlist is confirmed on three roots.
SHORTLIST: tuple[Candidate, ...] = (
    Candidate(1e-3, 1e-2, 512),
    Candidate(3e-4, 1e-2, 512),
    Candidate(1e-3, 1e-2, 2048),
)


def specifications(
    candidates: tuple[Candidate, ...], roots: int
) -> list[RunSpecification]:
    """
    Enumerate the sweep without starting any of it.
    """
    execution = ExecutionConfig(environment_workers=physical_cpu_count())
    steering = LoggingConfig().near_saturated_steering_threshold
    runs: list[RunSpecification] = []
    for candidate in candidates:
        for root in range(roots):
            run_id = f"{candidate.name}-seed-{root}"
            path = RESULTS_ROOT / run_id
            runs.append(
                RunSpecification(
                    run_id,
                    path,
                    partial(
                        run_a2c_training,
                        seed=root,
                        track_path=TRACK_PATH,
                        run_path=path,
                        actor_config=MEDIUM_ACTOR_CONFIG,
                        actor_learning_rate=candidate.actor_rate,
                        critic_learning_rate=candidate.critic_rate,
                        a2c_config=replace(
                            A2CConfig(), transitions_per_rollout=candidate.rollout
                        ),
                        training_interaction_budget=BUDGET,
                        evaluation_interval=EVALUATION_INTERVAL,
                        execution_config=execution,
                        near_saturated_steering_threshold=steering,
                        run_category=RunCategory.PRE_EXPERIMENT,
                    ),
                )
            )
    return runs


def _scores(candidate: Candidate, roots: int) -> dict[str, Any]:
    """
    Read a candidate's final evaluations and its last recorded critic fit.
    """
    laps, progress, returns, explained, spread = 0, [], [], [], []
    for root in range(roots):
        run = RESULTS_ROOT / f"{candidate.name}-seed-{root}"
        if not (run / "completion.json").is_file():
            continue
        final = int(
            json.loads((run / "completion.json").read_text(encoding="utf-8"))[
                "training_interactions"
            ]
        )
        for line in (
            (run / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            if int(row["training_interactions"]) != final:
                continue
            episode = row.get("episode", row)
            laps += int(episode["outcome"] == "completed")
            progress.append(min(1.0, float(episode["maximum_progress"])))
            returns.append(float(episode["undiscounted_return"]))
        updates = [
            json.loads(line)
            for line in (run / "updates.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for update in updates[-20:]:
            explained.append(float(update["explained_variance"]))
            spread.append(
                float(update["diagnostics"]["value_prediction_standard_deviation"])
                / max(
                    1e-9,
                    float(update["diagnostics"]["value_target_standard_deviation"]),
                )
            )
    return {
        "name": candidate.name,
        "actor_rate": candidate.actor_rate,
        "critic_rate": candidate.critic_rate,
        "rollout": candidate.rollout,
        "critic_steps_per_2m": candidate.critic_steps,
        "roots": len(progress),
        "laps": laps,
        "mean_progress": sum(progress) / len(progress) if progress else 0.0,
        "mean_return": sum(returns) / len(returns) if returns else 0.0,
        "explained_variance": sum(explained) / len(explained) if explained else 0.0,
        "prediction_over_target_spread": sum(spread) / len(spread) if spread else 0.0,
    }


def report(candidates: tuple[Candidate, ...], roots: int) -> list[dict[str, Any]]:
    """
    Print every candidate with the critic diagnostics that explain its score.
    """
    rows = [
        _scores(candidate, roots) for candidate in CANDIDATES if candidate in candidates
    ]
    header = (
        "Actor",
        "Critic",
        "Rollout",
        "Critic steps",
        "Laps",
        "Progress",
        "Return",
        "ExplVar",
        "Pred/Tgt",
    )
    widths = (8, 8, 7, 12, 6, 8, 9, 8, 8)
    print("| " + " | ".join(f"{n:<{w}}" for n, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(
            f"| {row['actor_rate']:<8g} | {row['critic_rate']:<8g} "
            f"| {row['rollout']:<7} | {row['critic_steps_per_2m']:<12,} "
            f"| {row['laps']:>2}/{row['roots']:<3} | {row['mean_progress']:<8.3f} "
            f"| {row['mean_return']:<9.2f} | {row['explained_variance']:<8.3f} "
            f"| {row['prediction_over_target_spread']:<8.3f} |"
        )
    return rows


def main() -> int:
    """
    Screen every candidate on one root, or confirm the shortlist on three.
    """
    parser = argparse.ArgumentParser(description="Tune A2C's own configuration.")
    parser.add_argument("command", choices=("screen", "confirm", "report"))
    parser.add_argument("--roots", type=int, default=None)
    parsed = parser.parse_args()

    candidates = CANDIDATES if parsed.command in ("screen", "report") else SHORTLIST
    roots = parsed.roots or (1 if parsed.command == "screen" else 3)

    if parsed.command != "report":
        failures = summarize(
            execute(
                specifications(candidates, roots),
                contract=learning_contract(EnvironmentConfig()),
            )
        )
        if failures:
            return 1

    rows = report(candidates, roots)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / f"{parsed.command}.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
