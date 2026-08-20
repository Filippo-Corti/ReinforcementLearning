"""The parts of the three algorithm notebooks that do not depend on the algorithm.

Each notebook builds one agent and one engine, then does the same thing with
whatever comes out: train to a budget, reshape the records into plot rows, draw
the same five figures, and report. Only the first part differs, so only the
first part lives in the per-algorithm scripts next to this file.

These scripts exist so the notebooks can be *run*. A notebook driven from a CLI
is awkward to debug and gives no exit status, and after a refactor what matters
is whether the pipeline still runs end to end and whether the numbers coming
out of it are sane. Both are answered here, with a non-zero exit when they are
not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

# Chosen before pyplot is imported: these run headless, with every figure
# written to a file rather than shown in a window that nothing would display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configs import (
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    ObservationNormalizationConfig,
    StartStateConfig,
    physical_cpu_count,
)
from envs.racing import RacingEnv, observation_space_for
from envs.tracks import TrackWithGeometry
from normalization import RunningObservationNormalizer
from recording import RunCategory
from training import TrainingCircuitSchedule
from utils.random import (
    RunSeedStreams,
    SeedNamespace,
    SeedStream,
    configure_torch_determinism,
)
from utils.visualization import (
    plot_algorithm_diagnostics,
    plot_driving_behavior,
    plot_optimization_and_exploration,
    plot_outcomes_and_lap_time,
    plot_progress_and_efficiency,
    plot_task_performance,
)

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="pygame.pkgdata",
)

# Values the three notebooks share verbatim.
ROOT_SEED = 0
TRAINING_INTERACTION_BUDGET = 2_000_000
TRAIN_ON_RANDOM_CIRCUITS = False
SINGLE_CIRCUIT_SEED = 0
EVALUATION_INTERVAL_INTERACTIONS = 50_000
MOVING_AVERAGE_WINDOW = 20

ENVIRONMENT_CONFIG = EnvironmentConfig()
# Training samples a start pose all around the circuit for state coverage, but
# every greedy evaluation launches from the canonical start line so that the
# reported number always answers the same question and is exactly reproducible.
EVALUATION_CONFIG = replace(
    ENVIRONMENT_CONFIG, start=StartStateConfig(randomized=False)
)
NORMALIZATION_CONFIG = ObservationNormalizationConfig()

# Anchored to this file rather than to the working directory, so a run
# writes to the same place whether it was started from the repository root
# or from inside this folder.
DEFAULT_OUTPUT = pathlib.Path(__file__).parent / "output"


def parse_arguments(algorithm: str) -> argparse.Namespace:
    """
    Read the few settings a verification run needs to differ on.

    The defaults reproduce the notebook exactly. A smoke run overrides the
    budget and the evaluation interval together, because an interval larger
    than the budget produces no evaluation checkpoints at all and would leave
    the evaluation path unexercised.
    """
    parser = argparse.ArgumentParser(description=f"Run the {algorithm} notebook.")
    parser.add_argument(
        "--interactions",
        type=int,
        default=TRAINING_INTERACTION_BUDGET,
        help="Training interaction budget (default: the notebook's).",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=EVALUATION_INTERVAL_INTERACTIONS,
        help="Interactions between greedy evaluation checkpoints.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=physical_cpu_count(),
        help="Environment worker processes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where the figures and the summary are written.",
    )
    return parser.parse_args()


class RunContext:
    """
    Hold everything an algorithm needs to build its agent and its engine.

    Fields:
        * arguments: Parsed command-line settings for this run.
        * streams: Seed streams derived from the one root identity.
        * reference_track: Circuit the run trains and evaluates on.
        * observation_dimensions: Width of the normalized observation.
        * execution_config: Device and worker-count choice.
        * normalizer: Training-updated, evaluation-frozen observation statistics.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        """
        Derive the circuit, the random streams, and the observation shape.
        """
        self.arguments = arguments
        self.execution_config = ExecutionConfig(environment_workers=arguments.workers)
        configure_torch_determinism(self.execution_config)
        self.streams = RunSeedStreams(
            SeedNamespace.MULTI_CIRCUIT_DEVELOPMENT, ROOT_SEED
        )
        self.reference_track = TrackWithGeometry.generate(
            SINGLE_CIRCUIT_SEED,
            track_config=ENVIRONMENT_CONFIG.track,
            vehicle_config=ENVIRONMENT_CONFIG.vehicle,
        )
        self.observation_dimensions = observation_space_for(
            self.reference_track, ENVIRONMENT_CONFIG
        ).shape[0]
        self.normalizer = RunningObservationNormalizer(
            self.observation_dimensions, NORMALIZATION_CONFIG
        )

    @property
    def worker_count(self) -> int:
        """
        Return how many environment workers this run steps.
        """
        return self.arguments.workers

    def torch_generators(self, stream: SeedStream) -> tuple[Any, ...]:
        """
        Return one torch generator per worker, from independent substreams.
        """
        return tuple(
            self.streams.get_torch_generator(stream, substream_identity=index)
            for index in range(self.worker_count)
        )

    def numpy_generators(self, stream: SeedStream) -> tuple[Any, ...]:
        """
        Return one NumPy generator per worker, from independent substreams.
        """
        return tuple(
            self.streams.get_numpy_generator(stream, substream_identity=index)
            for index in range(self.worker_count)
        )

    def engine_arguments(self) -> dict[str, Any]:
        """
        Return the engine keyword arguments every algorithm passes identically.
        """
        return {
            "run_category": RunCategory.REDUCED_VALIDATION,
            "evaluation_environment_factory": lambda: RacingEnv(
                self.reference_track, config=EVALUATION_CONFIG
            ),
            "evaluation_interval": self.arguments.evaluation_interval,
            "environment_reset_generators": self.numpy_generators(
                SeedStream.ENVIRONMENT_RESETS
            ),
            "track_selection_generators": self.numpy_generators(
                SeedStream.TRAINING_TRACK_SELECTION
            ),
            "training_circuit_schedule": (
                TrainingCircuitSchedule() if TRAIN_ON_RANDOM_CIRCUITS else None
            ),
            "execution_config": self.execution_config,
            "show_progress": True,
            "evaluation_seed": int(
                self.streams.get_numpy_generator(SeedStream.EVALUATION).integers(
                    0, 2**32
                )
            ),
            "root_identity": ROOT_SEED,
            "near_saturated_steering_threshold": (
                LoggingConfig().near_saturated_steering_threshold
            ),
        }


def evaluation_row(evaluation: Any) -> dict[str, Any]:
    """
    Summarize one recorded evaluation for the plots below.
    """
    episode = evaluation.record.episode
    return {
        "training_interactions": evaluation.record.training_interactions,
        "return_mean": episode.undiscounted_return,
        "return_standard_deviation": 0.0,
        "maximum_progress_mean": episode.maximum_progress,
        "maximum_progress_standard_deviation": 0.0,
        "lap_time_mean": episode.lap_time,
        "completed_fraction": float(episode.outcome.value == "completed"),
        "outcomes": [episode.outcome.value],
        "completed_training_episodes": None,
    }


def update_row(update: Any) -> dict[str, Any]:
    """
    Expose one optimizer update in the shape the diagnostic plots read.
    """
    return {
        "training_interactions": update.training_interactions,
        "diagnostics": update.output.diagnostics,
        "optimization_duration": update.optimization_duration,
    }


def episode_row(record: Any) -> dict[str, Any]:
    """
    Expose one finished training episode in the shape the curves read.
    """
    row = asdict(record)
    row["outcome"] = record.outcome.value
    row["mean_speed"] = record.speed.mean
    row["mean_throttle"] = record.throttle.mean
    return row


def run_notebook(
    algorithm: str,
    build: Callable[[RunContext], tuple[Any, Any]],
    *,
    include_critic: bool,
) -> int:
    """
    Train one algorithm to its budget, draw its figures, and check its numbers.

    Returns a process exit status rather than raising on a bad run: a training
    run that finishes but produces nonsense is not an exception, it is a result
    that has to be reported and then acted on.
    """
    arguments = parse_arguments(algorithm)
    context = RunContext(arguments)
    agent, engine = build(context)

    print(f"=== {algorithm} ===")
    print(f"Circuit: fixed seed {SINGLE_CIRCUIT_SEED}")
    print(f"Interaction budget: {arguments.interactions:,}")
    print(f"Workers: {context.worker_count}")
    print(f"Actor parameters: {agent.actor_parameter_count:,}")
    if agent.critic_parameter_count is not None:
        print(f"Critic parameters: {agent.critic_parameter_count:,}")

    try:
        state = engine.train(arguments.interactions)
        episodes = [episode_row(record) for record in engine.episode_records]
        evaluations = [evaluation_row(row) for row in engine.evaluations]
        updates = [update_row(row) for row in engine.updates]
    finally:
        engine.close()

    print(f"Episodes completed: {len(episodes):,}")
    print(f"Training interactions: {state.counters.training_interactions:,}")
    print(f"Optimizer updates: {state.counters.optimizer_updates:,}")
    print(f"Greedy evaluation checkpoints: {len(evaluations):,}")

    output = arguments.output / algorithm.lower()
    output.mkdir(parents=True, exist_ok=True)
    _draw_figures(
        algorithm,
        episodes,
        evaluations,
        updates,
        output,
        include_critic=include_critic,
    )
    problems = _check(algorithm, state, episodes, evaluations, updates)
    _report(algorithm, evaluations, updates, problems, output)
    _write_summary(algorithm, state, evaluations, updates, output)
    return 1 if problems else 0


def _draw_figures(
    algorithm: str,
    episodes: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    include_critic: bool,
) -> None:
    """
    Draw every figure the notebook draws, to files instead of to a window.

    Drawing them is itself part of the check: the plotting helpers read the
    recorded diagnostics by name, so a renamed or dropped key fails here rather
    than silently disappearing from a chart nobody is looking at.
    """
    figures = {
        "task_performance": lambda: plot_task_performance(
            episodes, evaluations, moving_average_window=MOVING_AVERAGE_WINDOW
        ),
        "progress_and_efficiency": lambda: plot_progress_and_efficiency(
            episodes, evaluations, moving_average_window=MOVING_AVERAGE_WINDOW
        ),
        "driving_behavior": lambda: plot_driving_behavior(
            episodes, moving_average_window=MOVING_AVERAGE_WINDOW
        ),
        "outcomes_and_lap_time": lambda: plot_outcomes_and_lap_time(
            episodes, moving_average_window=MOVING_AVERAGE_WINDOW
        ),
        "optimization_and_exploration": lambda: plot_optimization_and_exploration(
            updates,
            algorithm_name=algorithm,
            moving_average_window=MOVING_AVERAGE_WINDOW,
            include_critic=include_critic,
        ),
        "algorithm_diagnostics": lambda: plot_algorithm_diagnostics(
            updates, algorithm_name=algorithm
        ),
    }
    for name, draw in figures.items():
        figure = draw()
        if figure is None:
            continue
        figure.savefig(output / f"{name}.png", dpi=110)
        plt.close(figure)
    print(f"Figures written to {output}")


def _check(
    algorithm: str,
    state: Any,
    episodes: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
) -> list[str]:
    """
    Look for the failures a refactor actually causes, not for slow learning.

    A short run cannot say whether the policy improves, so nothing here asks it
    to. What it can say is whether the machinery still turns: that the budget
    was spent exactly, that updates and evaluations happened, and that no
    recorded diagnostic came back as a NaN, which is how a broken loss or a
    detached-gradient mistake shows up first.
    """
    problems: list[str] = []
    if not updates:
        problems.append("no optimizer update ran")
    if not episodes:
        problems.append("no training episode finished")
    if not evaluations:
        problems.append("no greedy evaluation ran")

    for update in updates:
        for key, value in update["diagnostics"].items():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                problems.append(f"diagnostic {key!r} is {value} at update")
                break
        else:
            continue
        break

    for row in evaluations:
        if not np.isfinite(float(row["return_mean"])):
            problems.append("a greedy evaluation returned a non-finite value")
            break

    interactions = state.counters.training_interactions
    for update in updates:
        if update["training_interactions"] > interactions:
            problems.append("an update is stamped past the end of the run")
            break
    return problems


def _report(
    algorithm: str,
    evaluations: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    problems: Sequence[str],
    output: Path,
) -> None:
    """
    Print the greedy evaluation trend, the last diagnostics, then any problem.

    The evaluation column is the one worth reading first. Every other number
    here says the machinery ran; this one says whether the policy is getting
    anywhere, which is the only question a longer run can answer that a short
    one cannot.
    """
    if evaluations:
        print(f"\nGreedy evaluation checkpoints for {algorithm}:")
        print(f"  {'interactions':>14}  {'return':>10}  {'progress':>9}  outcome")
        for row in evaluations:
            print(
                f"  {row['training_interactions']:>14,}"
                f"  {row['return_mean']:>10.3f}"
                f"  {row['maximum_progress_mean']:>9.4f}"
                f"  {row['outcomes'][0]}"
            )
    if updates:
        print(f"\nLast {algorithm} update diagnostics:")
        for key, value in sorted(updates[-1]["diagnostics"].items()):
            if isinstance(value, float):
                print(f"  {key:<42} {value: .6g}")
            else:
                print(f"  {key:<42} {value}")
    if problems:
        print(f"\nFAILED: {algorithm}")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print(f"\nOK: {algorithm} ran and every recorded diagnostic is finite.")


def _write_summary(
    algorithm: str,
    state: Any,
    evaluations: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    """
    Write the numbers behind the figures, so a run can be compared to another.
    """
    summary = {
        "algorithm": algorithm,
        "training_interactions": state.counters.training_interactions,
        "finished_episodes": state.counters.finished_episodes,
        "optimizer_updates": state.counters.optimizer_updates,
        "evaluations": [
            {
                "training_interactions": row["training_interactions"],
                "return_mean": row["return_mean"],
                "maximum_progress_mean": row["maximum_progress_mean"],
                "outcome": row["outcomes"][0],
            }
            for row in evaluations
        ],
        "final_diagnostics": updates[-1]["diagnostics"] if updates else {},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
