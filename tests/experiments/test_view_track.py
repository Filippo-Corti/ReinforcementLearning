"""Tests for the interactive track-viewer experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from envs import RacingEnv
from experiments import view_track


def test_parser_accepts_exactly_one_track_source() -> None:
    """
    The viewer requires either a generation seed or a saved track path.
    """
    parser = view_track.build_parser()

    seeded = parser.parse_args(["--seed", "12"])
    saved = parser.parse_args(["--track", "tracks/example.json"])

    assert seeded.seed == 12
    assert seeded.track is None
    assert saved.seed is None
    assert saved.track == Path("tracks/example.json")
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "1", "--track", "track.json"])


def test_build_environment_uses_human_rendering() -> None:
    """
    A seeded viewer environment is configured for an interactive window.
    """
    arguments = argparse.Namespace(seed=4, track=None)

    environment = view_track.build_environment(arguments)

    assert isinstance(environment, RacingEnv)
    assert environment.render_mode == "human"
    assert environment.track is not None
    assert environment.track.generation.seed == 4
    environment.close()


def test_main_delegates_to_viewer_without_running_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Main constructs the selected environment and delegates the blocking loop.
    """
    received: list[RacingEnv] = []
    monkeypatch.setattr(view_track, "run_viewer", received.append)

    result = view_track.main(["--seed", "8"])

    assert result == 0
    assert len(received) == 1
    assert received[0].track is not None
    assert received[0].track.generation.seed == 8
    received[0].close()
