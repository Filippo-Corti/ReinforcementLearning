"""Tests for the interactive manual-driving experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from envs import RacingEnv
from experiments import manual_drive


def test_parser_accepts_exactly_one_track_source() -> None:
    """
    The manual driver requires either a generation seed or a saved track path.
    """
    parser = manual_drive.build_parser()

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


def test_controls_match_documented_action_signs() -> None:
    """
    W and A are positive axes while S and D are negative axes.
    """
    keys = [False] * (max(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d) + 1)
    keys[pygame.K_w] = True
    keys[pygame.K_a] = True

    assert np.array_equal(
        manual_drive.controls_from_keys(keys), np.asarray([1.0, 1.0], dtype=np.float32)
    )

    keys[pygame.K_w] = False
    keys[pygame.K_a] = False
    keys[pygame.K_s] = True
    keys[pygame.K_d] = True
    assert np.array_equal(
        manual_drive.controls_from_keys(keys),
        np.asarray([-1.0, -1.0], dtype=np.float32),
    )


def test_build_environment_uses_human_rendering() -> None:
    """
    A seeded manual-driver environment is configured for an interactive window.
    """
    environment = manual_drive.build_environment(argparse.Namespace(seed=4, track=None))

    assert isinstance(environment, RacingEnv)
    assert environment.render_mode == "human"
    assert environment.track is not None
    assert environment.track.generation.seed == 4
    environment.close()


def test_driver_keeps_terminal_state_visible_until_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A completed episode is rendered but not stepped or reset automatically.
    """

    class FakeClock:
        def tick(self, _: int) -> None:
            pass

    class FakeEnvironment:
        config = SimpleNamespace(
            simulation=SimpleNamespace(agent_timestep=0.04),
        )
        state = type("State", (), {"speed": 3.0})()

        def __init__(self) -> None:
            self.steps = 0
            self.resets = 0
            self.renders = 0
            self.closed = False

        def reset(self) -> tuple[None, dict[str, object]]:
            self.resets += 1
            return None, {"episode_progress": 0.0}

        def step(
            self, _: np.ndarray
        ) -> tuple[None, float, bool, bool, dict[str, object]]:
            self.steps += 1
            return (
                None,
                -20.0,
                True,
                False,
                {
                    "episode_progress": 1.0,
                    "collision": True,
                    "lap_completed": False,
                },
            )

        def render(self) -> None:
            self.renders += 1

        def close(self) -> None:
            self.closed = True

    environment = FakeEnvironment()
    events = [[], [pygame.event.Event(pygame.QUIT)]]
    monkeypatch.setattr(pygame.event, "get", lambda: events.pop(0))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: [False] * 512)
    monkeypatch.setattr(pygame.time, "Clock", FakeClock)
    monkeypatch.setattr(pygame.display, "set_caption", lambda _: None)

    manual_drive.run_driver(environment)  # type: ignore[arg-type]

    assert environment.steps == 1
    assert environment.resets == 1
    assert environment.renders == 2
    assert environment.closed


def test_main_delegates_to_driver_without_running_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Main constructs the selected environment and delegates the blocking loop.
    """
    received: list[RacingEnv] = []
    monkeypatch.setattr(manual_drive, "run_driver", received.append)

    result = manual_drive.main(["--seed", "8"])

    assert result == 0
    assert len(received) == 1
    assert received[0].track is not None
    assert received[0].track.generation.seed == 8
    received[0].close()
