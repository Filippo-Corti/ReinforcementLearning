"""Run Experiment 2: circuit generalization and observation choice.

The protocol is in `docs/EXPERIMENT.md` and the reported outcome is written up
in `docs/EXPERIMENT_2.md`. This file is the executable half: it trains PPO on an
unbounded schedule of generated circuits under two observations, evaluates on
held-out splits, and turns the recorded runs into the tables and figures the
write-up reads from.

    python experiments/experiment_2.py plan       # enumerate, run nothing
    python experiments/experiment_2.py run        # train the matrix
    python experiments/experiment_2.py analyze    # tables and figures only

The actor width is **not** a free choice here: it is whatever Experiment 1
recorded in its `ppo_actor_selection.json`, read at the same run category. That
is what makes the two experiments one study rather than two. `--actor` overrides
it, and says so loudly, for deliberate deviations only.

`run` is resumable and contract-checked. `--rehearsal` swaps the protocol budget
for a short one and writes under a different category and seed namespace, so a
rehearsal can neither be mistaken for a result nor share randomness with one.

Two observations times five roots is 10 runs of two million interactions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from analyze_results import analyze_results
from matrix import RunSpecification, execute, learning_contract, summarize
from reporting import read_table
from train import run_ppo_training

from circuits import CircuitSplit, TrainingCircuitSchedule, load_split_circuits
from configs import (
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    ObservationRepresentation,
    PPOConfig,
    physical_cpu_count,
)
from recording import RunCategory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = PROJECT_ROOT / "tracks" / "experiment_2_splits.json"

ACTORS = {
    "small": SMALL_ACTOR_CONFIG,
    "medium": MEDIUM_ACTOR_CONFIG,
    "large": LARGE_ACTOR_CONFIG,
}
OBSERVATIONS = {
    "frenet": ObservationRepresentation.FRENET,
    "lidar": ObservationRepresentation.LIDAR,
}

# Selected before the experiment by the configuration check in EXPERIMENT.md.
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-2

# Circuits drawn from the training schedule and re-evaluated as a same-family
# reference, so the generalization gap is measured against circuits the agent
# trains on rather than against the single circuit of Experiment 1.
TRAINING_REFERENCE_CIRCUITS = 16


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
        return PROJECT_ROOT / "results" / self.category.value / "experiment_2"

    @property
    def analysis_root(self) -> Path:
        """
        Return where this scale's tables and figures are written.
        """
        return (
            PROJECT_ROOT / "results" / "analysis" / self.category.value / "experiment_2"
        )

    @property
    def experiment_1_analysis(self) -> Path:
        """
        Return where the Experiment 1 actor selection is expected to be.
        """
        return (
            PROJECT_ROOT / "results" / "analysis" / self.category.value / "experiment_1"
        )


PROTOCOL = Scale(2_000_000, 50_000, (0, 1, 2, 3, 4), RunCategory.REPORTED)
REHEARSAL = Scale(60_000, 5_000, (0, 1), RunCategory.REDUCED_VALIDATION)


def selected_actor(scale: Scale, override: str | None) -> str:
    """
    Return the actor width Experiment 1 chose, or a deliberate override.

    Reading this from Experiment 1's recorded selection rather than restating it
    is what keeps the two experiments one study: nobody can quietly run
    Experiment 2 at a width Experiment 1 did not choose.
    """
    if override is not None:
        print(f"OVERRIDDEN: using the {override!r} actor, not the recorded selection.")
        return override
    path = scale.experiment_1_analysis / "ppo_actor_selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no recorded actor selection at {path}. Run experiment_1.py first, "
            "at the same run category."
        )
    recorded = str(json.loads(path.read_text(encoding="utf-8"))["selected_actor"])
    print(f"Experiment 1 selected the {recorded!r} actor.")
    return recorded


def split_circuits(observation: ObservationRepresentation) -> dict[str, Any]:
    """
    Rebuild the frozen validation and test circuits under one observation.

    The circuits must be built per observation rather than once: a circuit
    carries the environment configuration it was built with, so building them
    once under the default would silently hand the LiDAR runs Frenet
    environments.
    """
    environment = replace(EnvironmentConfig(), observation_type=observation)
    return {
        split.value: load_split_circuits(
            SPLITS_PATH, split, environment_config=environment
        )
        for split in (CircuitSplit.VALIDATION, CircuitSplit.TEST)
    }


def specifications(scale: Scale, actor_name: str) -> list[RunSpecification]:
    """
    Enumerate every condition of the matrix, without starting any of them.
    """
    execution = ExecutionConfig(environment_workers=physical_cpu_count())
    steering = LoggingConfig().near_saturated_steering_threshold
    actor_config = ACTORS[actor_name]
    runs: list[RunSpecification] = []
    for name, observation in OBSERVATIONS.items():
        circuits = split_circuits(observation)
        for root in scale.roots:
            run_id = f"ppo-{actor_name}-{name}-seed-{root}"
            path = scale.results_root / run_id
            runs.append(
                RunSpecification(
                    run_id,
                    path,
                    partial(
                        run_ppo_training,
                        seed=root,
                        run_path=path,
                        actor_config=actor_config,
                        actor_learning_rate=ACTOR_LEARNING_RATE,
                        critic_learning_rate=CRITIC_LEARNING_RATE,
                        training_interaction_budget=scale.budget,
                        environment_config=EnvironmentConfig(),
                        evaluation_interval=scale.evaluation_interval,
                        execution_config=execution,
                        near_saturated_steering_threshold=steering,
                        # An unbounded schedule of generated circuits: this is
                        # what makes the experiment about circuits rather than
                        # about one circuit.
                        training_circuit_schedule=TrainingCircuitSchedule(),
                        evaluation_circuits=circuits[CircuitSplit.VALIDATION.value],
                        final_evaluation_circuits=circuits[CircuitSplit.TEST.value],
                        training_reference_circuits=TRAINING_REFERENCE_CIRCUITS,
                        observation=observation,
                        run_category=scale.category,
                    ),
                )
            )
    return runs


def contract() -> dict[str, Any]:
    """
    Return the constants a completed run must have been produced under.
    """
    return learning_contract(EnvironmentConfig(), PPOConfig())


def analyze(scale: Scale) -> dict[str, Any]:
    """
    Write every table and figure, then print the digest.

    The split manifest is passed as the geometry specification so the stratified
    table reads its bin edges from the file that froze them.
    """
    manifest = analyze_results(
        results_root=scale.results_root,
        output_directory=scale.analysis_root,
        experiment=2,
        category=scale.category,
        geometry_specification=SPLITS_PATH,
    )
    print(f"analyzed {len(manifest['inputs'])} runs -> {scale.analysis_root}")
    _report(scale)
    return manifest


def _report(scale: Scale) -> None:
    """
    Print the digest a person running this from a terminal wants to see.

    Deliberately ASCII: a Windows console encodes stdout as cp1252, and this is
    exactly the output a reader would copy a number out of.
    """
    splits = read_table(scale.analysis_root, "final_split_summaries")
    header = ("Observation", "Root", "Split", "Circuits", "Completion", "Progress")
    widths = (12, 5, 20, 9, 11, 9)
    print()
    print("| " + " | ".join(f"{n:<{w}}" for n, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in sorted(
        splits,
        key=lambda row: (
            row["observation_type"],
            row["circuit_split"],
            row["root_identity"],
        ),
    ):
        print(
            f"| {row['observation_type']:<12} | {row['root_identity']:<5} "
            f"| {row['circuit_split']:<20} | {row['circuit_count']:<9} "
            f"| {row['completion_rate']:<11.3f} | {row['mean_progress']:<9.3f} |"
        )

    # The gap rows already hold differences, one per root, named by contrast.
    print()
    for row in sorted(
        read_table(scale.analysis_root, "generalization_gaps"),
        key=lambda row: (
            row["contrast"],
            row["observation_type"],
            row["root_identity"],
        ),
    ):
        print(
            f"{row['contrast']:<32} {row['observation_type']:<7} "
            f"root {row['root_identity']}  "
            f"completion {row['completion_rate']:+.3f}  "
            f"progress {row['mean_progress']:+.3f}  "
            f"return {row['mean_return']:+8.2f}"
        )


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
    parser.add_argument(
        "--actor",
        choices=tuple(ACTORS),
        default=None,
        help="Deviate from the recorded Experiment 1 selection. Deliberate use only.",
    )
    parsed = parser.parse_args()
    scale = REHEARSAL if parsed.rehearsal else PROTOCOL

    if parsed.command == "analyze":
        analyze(scale)
        return 0

    actor_name = selected_actor(scale, parsed.actor)
    runs = specifications(scale, actor_name)
    print(f"budget {scale.budget:,} | roots {scale.roots} | {scale.category.value}")
    print(f"{len(runs)} runs -> {scale.results_root}")

    if parsed.command == "plan":
        for run in runs:
            state = "done" if (run.path / "completion.json").is_file() else "pending"
            print(f"  {state:<8} {run.run_id}")
        return 0

    failures = summarize(execute(runs, contract=contract()))
    if failures:
        return 1

    analyze(scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
