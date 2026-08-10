from __future__ import annotations

from pathlib import Path

from configs import (
    SMALL_ACTOR_CONFIG,
    EnvironmentConfig,
    ExecutionConfig,
    SimulationConfig,
)
from experiments.train import parse_arguments, run_reinforce_training
from recording import RunCategory, RunDirectory


def test_train_entry_point_runs_reinforce_and_writes_shared_records(tmp_path) -> None:
    root = Path(__file__).parents[1]
    engine = run_reinforce_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        training_interaction_budget=8,
        environment_config=EnvironmentConfig(
            simulation=SimulationConfig(max_episode_steps=1)
        ),
        execution_config=ExecutionConfig(device="cpu"),
    )
    run = RunDirectory.open(
        tmp_path / "run",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )

    assert engine.state().counters.optimizer_updates == 1
    assert len(run.records("episodes")) == 8
    assert len(run.records("updates")) == 1
    assert run.records("updates")[0]["actor_loss"] is not None
    assert run.records("evaluations") == []
    assert run.require_complete()["timing"]["persistence"] > 0.0


def test_train_cli_requires_learning_rate_and_accepts_output_alias() -> None:
    parsed = parse_arguments(
        [
            "--seed",
            "3",
            "--track",
            "track.json",
            "--output",
            "run",
            "--actor-learning-rate",
            "0.001",
            "--interaction-budget",
            "8",
        ]
    )

    assert parsed.run_path == "run"
    assert parsed.actor_learning_rate == 0.001
