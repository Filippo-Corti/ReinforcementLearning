"""Run the reproducible Phase-1 racing-environment acceptance checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pygame
from gymnasium.utils.env_checker import check_env

from envs import RacingEnv, Track, TrackWithGeometry, generate_track_file

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import manual_drive

REPLAY_ACTIONS = (
    (1.0, 0.0),
    (1.0, 0.2),
    (0.6, -0.3),
    (0.0, 0.0),
    (-0.4, 0.1),
    (0.8, -0.2),
)

AUTOMATED_CHECKS = (
    ("dependency consistency", (sys.executable, "-m", "pip", "check")),
    (
        "formatting",
        (sys.executable, "-m", "black", "--check", "src", "experiments", "tests"),
    ),
    (
        "linting",
        (sys.executable, "-m", "ruff", "check", "src", "experiments", "tests"),
    ),
    (
        "type checking",
        (sys.executable, "-m", "pyright", "src", "tests", "experiments"),
    ),
    (
        "compilation",
        (sys.executable, "-m", "compileall", "-q", "src", "experiments", "tests"),
    ),
    ("test suite", (sys.executable, "-m", "pytest")),
    ("diff whitespace", ("git", "diff", "--check")),
)


@dataclass(frozen=True)
class ReplayTransition:
    """
    Complete externally visible result of one environment transition.

    Fields:
        * observation: Frenet observation returned from the environment.
        * reward: Reward returned from the environment.
        * terminated: Whether the episode reached an MDP terminal state.
        * truncated: Whether the episode reached its time limit.
        * info: Environment diagnostics returned for the transition.
        * state: Pose, speed and front-wheel angle after the transition.
    """

    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: tuple[tuple[str, Any], ...]
    state: tuple[float, float, float, float, float]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the acceptance-runner command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="fixed track seed")
    return parser


def run_automated_checks(repository_root: Path) -> None:
    """
    Run the repository's dependency, static, compilation and test checks.
    """
    for _, command in AUTOMATED_CHECKS:
        subprocess.run(command, cwd=repository_root, check=True)


def generate_save_reload_render(seed: int, directory: Path) -> Track:
    """
    Generate, persist, reload and RGB-render one deterministic circuit.
    """
    track_path = directory / "acceptance_track.json"
    generated = generate_track_file(track_path, seed=seed)
    reloaded = Track.load(track_path)
    if generated.to_dict() != reloaded.to_dict():
        raise AssertionError("saved track changed during the load round trip.")

    environment = RacingEnv(
        TrackWithGeometry.load(track_path),
        render_mode="rgb_array",
    )
    try:
        environment.reset()
        frame = environment.render()
    finally:
        environment.close()
    if frame is None or frame.shape != (800, 800, 3) or frame.dtype != np.uint8:
        raise AssertionError("rgb_array rendering did not return the declared frame.")
    return reloaded


def replay(
    seed: int, actions: Iterable[tuple[float, float]] = REPLAY_ACTIONS
) -> tuple[ReplayTransition, ...]:
    """
    Replay actions on one seeded environment and capture complete outputs.
    """
    environment = RacingEnv(TrackWithGeometry.generate(seed))
    results: list[ReplayTransition] = []
    try:
        # The start pose is sampled, so the seed has to reach reset for the
        # replay to be reproducible.
        environment.reset(seed=seed)
        for action in actions:
            observation, reward, terminated, truncated, info = environment.step(
                np.asarray(action, dtype=np.float32)
            )
            if environment.state is None:
                raise RuntimeError("environment state disappeared during replay.")
            results.append(
                ReplayTransition(
                    observation=tuple(float(value) for value in observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=tuple(sorted(info.items())),
                    state=(
                        environment.state.x,
                        environment.state.y,
                        environment.state.heading,
                        environment.state.speed,
                        environment.state.steering_angle,
                    ),
                )
            )
            if terminated or truncated:
                break
    finally:
        environment.close()
    return tuple(results)


def assert_deterministic_replay(seed: int) -> None:
    """
    Verify that the fixed action sequence produces identical complete outputs.
    """
    first = replay(seed)
    second = replay(seed)
    if first != second:
        raise AssertionError("fixed-seed action replay was not deterministic.")


def run_environment_checker(seed: int) -> None:
    """
    Run Gymnasium's conformance checker on a seeded environment.
    """
    environment = RacingEnv(TrackWithGeometry.generate(seed))
    try:
        check_env(environment, skip_render_check=False)
    finally:
        environment.close()


def manual_driver_smoke(seed: int) -> None:
    """
    Start the manual driver with SDL's dummy backend and immediately quit it.
    """
    previous_video = os.environ.get("SDL_VIDEODRIVER")
    previous_audio = os.environ.get("SDL_AUDIODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.quit()
    try:
        pygame.init()
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        environment = manual_drive.build_environment(
            argparse.Namespace(seed=seed, track=None)
        )
        manual_drive.run_driver(environment)
        if pygame.display.get_surface() is not None:
            raise AssertionError("manual driver did not release its display.")
    finally:
        pygame.quit()
        _restore_environment("SDL_VIDEODRIVER", previous_video)
        _restore_environment("SDL_AUDIODRIVER", previous_audio)


def _restore_environment(name: str, previous: str | None) -> None:
    """
    Restore one environment variable after an SDL smoke test.
    """
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def run_acceptance(seed: int, repository_root: Path) -> None:
    """
    Run the complete Phase-1 acceptance pass using one explicit track seed.
    """
    run_automated_checks(repository_root)
    with TemporaryDirectory(prefix="phase1-acceptance-") as temporary:
        generate_save_reload_render(seed, Path(temporary))
    assert_deterministic_replay(seed)
    run_environment_checker(seed)
    manual_driver_smoke(seed)


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and run all Phase-1 acceptance checks.
    """
    arguments = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    run_acceptance(arguments.seed, repository_root)
    print(f"Phase-1 acceptance passed with seed {arguments.seed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
