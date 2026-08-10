"""Evaluate documented random and scripted references on one saved circuit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from configs import EnvironmentConfig
from envs.racing import RacingEnv
from envs.tracks import TrackWithGeometry, generate_track_file
from models import RandomPolicy, ScriptedFrenetPolicy
from recording import (
    EvaluationRecord,
    MetricScope,
    PolicyEvaluation,
    RunCategory,
    RunDirectory,
    TimingRecord,
    collect_run_metadata,
)
from training import evaluate_policy_episode
from utils.random import RunSeedStreams, SeedNamespace, SeedStream


def run_reference_evaluation(
    *,
    seed: int,
    run_path: str | Path,
    references: Sequence[str] = ("random", "scripted"),
) -> tuple[PolicyEvaluation, ...]:
    """
    Generate, save, and evaluate references without consuming training counters.
    """
    if not references or any(
        reference not in {"random", "scripted"} for reference in references
    ):
        raise ValueError("references must contain only 'random' and/or 'scripted'.")
    run_started = perf_counter()
    streams = RunSeedStreams(SeedNamespace.REDUCED_BUDGET_VALIDATION, seed)
    track_streams = RunSeedStreams(SeedNamespace.EXPERIMENT_1_CIRCUIT_CANDIDATE, 0)
    track_seed = _first_seed(track_streams, SeedStream.TRACK_GENERATION)
    environment_config = EnvironmentConfig()
    run = RunDirectory.create(
        run_path,
        category=RunCategory.REDUCED_VALIDATION,
        run_id=f"reference-seed-{seed}",
        manifest={
            "purpose": "reference_evaluation",
            "root_seed": seed,
            "seed_namespace": streams.namespace.name,
            "reference_track_seed": track_seed,
            "seed_streams": {
                stream.name: _first_seed(streams, stream) for stream in SeedStream
            },
        },
        config={
            "environment": environment_config.to_dict(),
            "references": list(references),
            "scripted_policy": asdict(ScriptedFrenetPolicy()),
        },
        metadata=collect_run_metadata(
            repository=Path(__file__).resolve().parents[1],
            device="cpu",
            environment_workers=1,
        ),
    )
    track_path = run.path / "reference_track.json"
    track = generate_track_file(track_path, seed=track_seed)
    environment = RacingEnv(TrackWithGeometry(track), config=environment_config)
    evaluations: list[PolicyEvaluation] = []
    evaluation_interactions = 0
    evaluation_duration = 0.0
    persistence_duration = 0.0
    try:
        for evaluation_index, reference_name in enumerate(references):
            policy = (
                RandomPolicy(streams.get_numpy_generator(SeedStream.EVALUATION))
                if reference_name == "random"
                else ScriptedFrenetPolicy()
            )
            evaluation_started = perf_counter()
            evaluation = evaluate_policy_episode(
                environment,
                policy,
                run_category=RunCategory.REDUCED_VALIDATION,
                episode_index=evaluation_index,
                evaluation_interactions_before=evaluation_interactions,
                reset_seed=_first_seed(streams, SeedStream.ENVIRONMENT_RESETS),
                circuit_identity=str(track.generation.seed),
                root_identity=seed,
                circuit_split="development",
            )
            evaluation_duration += perf_counter() - evaluation_started
            evaluations.append(evaluation)
            evaluation_interactions += evaluation.episode.interactions
            persistence_started = perf_counter()
            run.append("episodes", evaluation.episode)
            run.append(
                "evaluations",
                EvaluationRecord(
                    run_category=RunCategory.REDUCED_VALIDATION,
                    scope=MetricScope.REFERENCE,
                    evaluation_index=evaluation_index,
                    training_interactions=0,
                    evaluation_interactions=evaluation_interactions,
                    episode=evaluation.episode,
                ),
            )
            run.write_trajectory(
                reference_name,
                {
                    "reference": reference_name,
                    "transitions": [
                        transition.to_dict() for transition in evaluation.transitions
                    ],
                },
            )
            persistence_duration += perf_counter() - persistence_started
    finally:
        environment.close()
    end_to_end_duration = perf_counter() - run_started
    run.complete(
        {
            "training_interactions": 0,
            "evaluation_interactions": evaluation_interactions,
            "timing": TimingRecord(
                run_category=RunCategory.REDUCED_VALIDATION,
                scope=MetricScope.REFERENCE,
                evaluation=evaluation_duration,
                persistence=persistence_duration,
                end_to_end=end_to_end_duration,
            ).to_dict(),
            "reference_outcomes": [
                evaluation.episode.to_dict() for evaluation in evaluations
            ],
        }
    )
    return tuple(evaluations)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse the explicit seed and safe run directory selected by the user.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, help="Reference root seed.")
    parser.add_argument(
        "--run-path",
        "--output",
        dest="run_path",
        required=True,
        help="New empty directory that will receive the recorded run outputs.",
    )
    parser.add_argument(
        "--reference",
        choices=("random", "scripted", "both"),
        default="both",
        help="Reference policy or policies to evaluate.",
    )
    return parser.parse_args(arguments)


def _first_seed(streams: RunSeedStreams, stream: SeedStream) -> int:
    """
    Draw the reproducible first uint32 value from one named NumPy stream.
    """
    return int(
        streams.get_numpy_generator(stream).integers(
            0,
            2**32,
        )
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Run the selected reference evaluations and print their semantic summaries.
    """
    parsed = parse_arguments(arguments)
    references = (
        ("random", "scripted") if parsed.reference == "both" else (parsed.reference,)
    )
    evaluations = run_reference_evaluation(
        seed=parsed.seed,
        run_path=parsed.run_path,
        references=references,
    )
    for evaluation in evaluations:
        episode = evaluation.episode
        print(
            f"{episode.episode_index}: {episode.outcome.value}; "
            f"progress={episode.maximum_progress:.3f}; steps={episode.interactions}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
