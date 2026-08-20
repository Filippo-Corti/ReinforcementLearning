from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from configs import EnvironmentConfig, PPOConfig
from experiments.matrix import (
    RunSpecification,
    contract_mismatch,
    execute,
    is_complete,
    learning_contract,
    summarize,
)


def _finished_run(directory: Path, *, environment: EnvironmentConfig) -> Path:
    """
    Write the two documents a matrix reads: the config and the completion.
    """
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "environment": environment.to_dict(),
                "training": {"ppo": PPOConfig().to_dict()},
            }
        ),
        encoding="utf-8",
    )
    (directory / "completion.json").write_text("{}", encoding="utf-8")
    return directory


def test_a_finished_run_is_skipped(tmp_path: Path) -> None:
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())
    started: list[str] = []

    outcomes = execute(
        [RunSpecification("run", path, lambda: started.append("run"))],
        report=lambda _: None,
    )

    assert is_complete(path)
    assert [outcome.status for outcome in outcomes] == ["skipped"]
    assert started == []


def test_a_run_recorded_under_other_constants_is_not_reused(tmp_path: Path) -> None:
    """
    The skip is what makes a matrix resumable, and what would silently reuse a
    run from a superseded contract. Changing a reward constant must re-run it.
    """
    superseded = EnvironmentConfig()
    superseded = replace(
        superseded, reward=replace(superseded.reward, lap_time_bonus=100.0)
    )
    path = _finished_run(tmp_path / "run", environment=superseded)
    started: list[str] = []

    outcomes = execute(
        [RunSpecification("run", path, lambda: started.append("run"))],
        contract=learning_contract(EnvironmentConfig(), PPOConfig()),
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["completed"]
    assert started == ["run"]


def test_a_matching_contract_still_skips(tmp_path: Path) -> None:
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())
    started: list[str] = []

    outcomes = execute(
        [RunSpecification("run", path, lambda: started.append("run"))],
        contract=learning_contract(EnvironmentConfig(), PPOConfig()),
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["skipped"]
    assert started == []


def test_the_mismatch_names_the_field_that_changed(tmp_path: Path) -> None:
    superseded = EnvironmentConfig()
    superseded = replace(
        superseded, reward=replace(superseded.reward, lap_time_bonus=100.0)
    )
    path = _finished_run(tmp_path / "run", environment=superseded)

    reason = contract_mismatch(
        path, learning_contract(EnvironmentConfig(), PPOConfig())
    )

    assert reason is not None
    assert "lap_time_bonus" in reason
    assert "100.0" in reason and "140.0" in reason


def test_a_discount_change_is_detected(tmp_path: Path) -> None:
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())

    reason = contract_mismatch(
        path,
        learning_contract(EnvironmentConfig(), replace(PPOConfig(), discount=0.9995)),
    )

    assert reason is not None
    assert "training.ppo.discount" in reason


def test_one_failure_does_not_stop_the_queue(tmp_path: Path) -> None:
    """
    An overnight matrix must finish the runs that can finish.
    """
    started: list[str] = []

    def fail() -> None:
        raise RuntimeError("deliberate")

    outcomes = execute(
        [
            RunSpecification("bad", tmp_path / "bad", fail),
            RunSpecification("good", tmp_path / "good", lambda: started.append("good")),
        ],
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["failed", "completed"]
    assert started == ["good"]
    assert summarize(outcomes, report=lambda _: None) == 1


def test_an_incomplete_directory_is_cleared_before_a_rerun(tmp_path: Path) -> None:
    """
    The recorder refuses a non-empty directory, so debris must go first.
    """
    path = tmp_path / "run"
    path.mkdir()
    (path / "episodes.jsonl").write_text("partial", encoding="utf-8")

    def launch() -> None:
        assert not path.exists(), "the interrupted run's debris was left behind"

    outcomes = execute([RunSpecification("run", path, launch)], report=lambda _: None)

    assert [outcome.status for outcome in outcomes] == ["completed"]
