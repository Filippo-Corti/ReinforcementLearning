"""Compare the discounted horizon against no discounting at all.

Superseded, and kept because it is the evidence behind a decision. The contract
it interrogates -- ``gamma = 0.9995`` with a lap-time bonus of 100 -- was
replaced on 2026-08-20 by ``gamma = 1`` with a bonus of 140, so re-running this
now compares two arms of a reward function the project no longer uses. The
findings and the corrections to their first analysis are archived in
``docs/old-plans/discount-horizon-study.md``.

The learning contract then fixed ``gamma = 0.9995`` for all three algorithms, but the
task it describes is finite-horizon: every episode ends by crashing, stalling,
finishing, or reaching the step cap. A discount is therefore not needed to make
the return converge, which raises the question of what it is doing and whether
``gamma = 1`` would do the same or better.

This runs the grid that answers it. Each algorithm is trained at both discounts
from the same seeds, on the same fixed circuit and actor, so the two conditions
of a pair differ in one number and in nothing else. Pairing matters here more
than the sample size does: this task's seed-to-seed spread is large enough that
an unpaired comparison of a handful of runs would measure the seeds.

Run the grid, then analyse it::

    python experiments/compare_discount_horizon.py run --shard 0 --shards 2
    python experiments/compare_discount_horizon.py analyse
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import run_a2c_training, run_ppo_training, run_reinforce_training

from configs import (
    MEDIUM_ACTOR_CONFIG,
    A2CConfig,
    Algorithm,
    PPOConfig,
    ReinforceConfig,
)
from recording.records import RunCategory

DEFAULT_ROOT = Path("results/pre_experiment_configuration/discount_horizon")
TRACK_PATH = Path("tracks/experiment_1.json")

# The discount is the only thing under test, so everything else is held at what
# the reported experiments will use: the calibrated rates, the medium actor, the
# fixed Experiment 1 circuit, and the Frenet observation.
DISCOUNTS: tuple[float, ...] = (0.9995, 1.0)
CALIBRATED_RATES: dict[Algorithm, dict[str, float]] = {
    Algorithm.REINFORCE: {"actor_learning_rate": 1e-3},
    Algorithm.A2C: {"actor_learning_rate": 3e-4, "critic_learning_rate": 1e-2},
    Algorithm.PPO: {"actor_learning_rate": 3e-4, "critic_learning_rate": 1e-2},
}


@dataclass(frozen=True, slots=True)
class Cell:
    """
    One run of the grid.

    Fields:
        * algorithm: Learning algorithm under test.
        * discount: The discount this run uses.
        * seed: Run seed, shared by both discounts of a pair.
    """

    algorithm: Algorithm
    discount: float
    seed: int

    @property
    def name(self) -> str:
        """
        Return the directory name identifying this run.
        """
        discount = "undiscounted" if self.discount == 1.0 else "discounted"
        return f"{self.algorithm.value}_{discount}_seed{self.seed}"


def grid(algorithms: Sequence[Algorithm], seeds: Sequence[int]) -> list[Cell]:
    """
    Enumerate the grid with the two discounts of a pair adjacent.

    Adjacency is deliberate: a pair that runs back to back on one machine shares
    whatever that machine was doing at the time, so a comparison within a pair
    cannot pick up a drift that a comparison across the grid could.
    """
    return [
        Cell(algorithm, discount, seed)
        for algorithm in algorithms
        for seed in seeds
        for discount in DISCOUNTS
    ]


def train_cell(
    cell: Cell,
    *,
    budget: int,
    output_root: Path,
    critic_learning_rate: float | None = None,
    name: str | None = None,
) -> float:
    """
    Train one cell and return the wall-clock seconds it took.

    ``critic_learning_rate`` overrides the calibrated one, which the control
    below needs: the calibrated rate was chosen at ``gamma = 0.9995``, and an
    undiscounted run regresses its critic on visibly larger targets.
    """
    run_path = output_root / (name or cell.name)
    rates = dict(CALIBRATED_RATES[cell.algorithm])
    if critic_learning_rate is not None and "critic_learning_rate" in rates:
        rates["critic_learning_rate"] = critic_learning_rate
    shared: dict[str, Any] = {
        "seed": cell.seed,
        "track_path": TRACK_PATH,
        "run_path": run_path,
        "actor_config": MEDIUM_ACTOR_CONFIG,
        "training_interaction_budget": budget,
        "run_category": RunCategory.PRE_EXPERIMENT,
        **rates,
    }

    started = perf_counter()
    if cell.algorithm is Algorithm.REINFORCE:
        run_reinforce_training(
            reinforce_config=ReinforceConfig(discount=cell.discount), **shared
        )
    elif cell.algorithm is Algorithm.A2C:
        run_a2c_training(a2c_config=A2CConfig(discount=cell.discount), **shared)
    else:
        run_ppo_training(ppo_config=PPOConfig(discount=cell.discount), **shared)
    return perf_counter() - started


def run_grid(
    *,
    algorithms: Sequence[Algorithm],
    seeds: Sequence[int],
    budget: int,
    output_root: Path,
    shard: int,
    shards: int,
) -> int:
    """
    Train every cell of one shard, skipping those already complete.
    """
    # Shard by pair, never by cell. Splitting the grid cell by cell would put
    # every discounted run on one worker and every undiscounted run on the
    # other, so any difference between the two workers - load, thermal state,
    # anything - would arrive as a difference between the two conditions.
    cells = [
        cell
        for index, cell in enumerate(grid(algorithms, seeds))
        if (index // len(DISCOUNTS)) % shards == shard
    ]
    print(f"shard {shard}/{shards}: {len(cells)} runs at {budget:,} interactions")

    for position, cell in enumerate(cells, start=1):
        completion = output_root / cell.name / "completion.json"
        if completion.exists():
            print(f"[{position}/{len(cells)}] {cell.name}: already complete")
            continue
        print(f"[{position}/{len(cells)}] {cell.name}: training", flush=True)
        elapsed = train_cell(cell, budget=budget, output_root=output_root)
        print(
            f"[{position}/{len(cells)}] {cell.name}: {elapsed / 60.0:.1f} min "
            f"({budget / max(elapsed, 1e-9):,.0f} interactions/s)",
            flush=True,
        )
    return 0


def run_critic_rate_control(
    *,
    budget: int,
    output_root: Path,
    seeds: Sequence[int],
    critic_rates: Sequence[float],
    shard: int,
    shards: int,
) -> int:
    """
    Ask whether an undiscounted PPO run only needs a cooler critic.

    The grid varies the discount while holding the critic rate at the value
    calibrated for ``gamma = 0.9995``. An undiscounted return is larger, so its
    regression targets are wider and that one rate is no longer the same rate in
    any meaningful sense. If a lower rate recovers what the undiscounted run
    lost, the grid measured a calibration mismatch rather than the discount.
    """
    runs = [(seed, rate) for seed in seeds for rate in critic_rates]
    # Shard by seed, so one worker never ends up owning one rate outright, for
    # the same reason the grid shards by pair.
    mine = [
        run
        for index, run in enumerate(runs)
        if (index // max(1, len(critic_rates))) % shards == shard
    ]
    print(f"critic-rate control, shard {shard}/{shards}: {len(mine)} runs")

    for position, (seed, rate) in enumerate(mine, start=1):
        name = f"ppo_undiscounted_critic{rate:g}_seed{seed}"
        if (output_root / name / "completion.json").exists():
            print(f"[{position}/{len(mine)}] {name}: already complete")
            continue
        print(f"[{position}/{len(mine)}] {name}: training", flush=True)
        elapsed = train_cell(
            Cell(Algorithm.PPO, 1.0, seed),
            budget=budget,
            output_root=output_root,
            critic_learning_rate=rate,
            name=name,
        )
        print(f"[{position}/{len(mine)}] {name}: {elapsed / 60.0:.1f} min", flush=True)
    return 0


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse the sub-command and its settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "control", "analyse"))
    parser.add_argument("--budget", type=int, default=2_000_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--critic-rates",
        nargs="+",
        type=float,
        default=[3e-3, 6e-3],
        help="critic rates the control tries at gamma = 1",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[algorithm.value for algorithm in Algorithm],
        choices=[algorithm.value for algorithm in Algorithm],
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Run or analyse the discount grid.
    """
    parsed = parse_arguments(arguments)
    algorithms = [Algorithm(name) for name in parsed.algorithms]
    seeds = list(range(parsed.seeds))

    if parsed.command == "control":
        return run_critic_rate_control(
            budget=parsed.budget,
            output_root=parsed.output_root,
            seeds=seeds,
            critic_rates=parsed.critic_rates,
            shard=parsed.shard,
            shards=parsed.shards,
        )

    if parsed.command == "run":
        return run_grid(
            algorithms=algorithms,
            seeds=seeds,
            budget=parsed.budget,
            output_root=parsed.output_root,
            shard=parsed.shard,
            shards=parsed.shards,
        )

    from discount_analysis import analyse, analyse_control

    status = analyse(output_root=parsed.output_root, algorithms=algorithms, seeds=seeds)
    analyse_control(
        output_root=parsed.output_root,
        seeds=seeds,
        critic_rates=parsed.critic_rates,
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
