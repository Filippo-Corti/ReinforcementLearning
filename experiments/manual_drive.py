"""Drive a generated or saved racing track with the keyboard."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from envs import RacingEnv


def build_parser() -> argparse.ArgumentParser:
    """
    Build the manual-driving command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=int, help="generate and drive this track seed")
    source.add_argument("--track", type=Path, help="load and drive this track JSON")
    return parser


def build_environment(arguments: argparse.Namespace) -> RacingEnv:
    """
    Construct the human-rendered environment selected by parsed arguments.
    """
    if arguments.track is not None:
        return RacingEnv(track_path=arguments.track, render_mode="human")
    return RacingEnv(track_seed=arguments.seed, render_mode="human")


def controls_from_keys(keys: Sequence[bool]) -> np.ndarray:
    """
    Translate the documented keyboard axes into a normalized environment action.
    """
    throttle = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
    steering = float(keys[pygame.K_a]) - float(keys[pygame.K_d])
    return np.asarray([throttle, steering], dtype=np.float32)


def diagnostic_caption(
    environment: RacingEnv,
    reward: float,
    info: dict[str, Any],
    terminal_reason: str | None,
) -> str:
    """
    Format the current driving diagnostics for the Pygame window title.
    """
    if environment.state is None:
        raise RuntimeError("environment state is not initialized.")
    reason = terminal_reason or "driving"
    return (
        "Manual drive | "
        f"speed: {environment.state.speed:.1f} | "
        f"progress: {info['episode_progress']:.1f} | "
        f"reward: {reward:.3f} | "
        f"status: {reason}"
    )


def terminal_reason(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> str | None:
    """
    Return the user-visible reason for a completed episode.
    """
    if not (terminated or truncated):
        return None
    if info["collision"]:
        return "crash"
    if info["lap_completed"]:
        return "lap completed"
    return "time limit"


def run_driver(environment: RacingEnv) -> None:
    """
    Run keyboard control until the user closes the window or presses Escape.
    """
    _, info = environment.reset()
    reward = 0.0
    completed_reason: str | None = None
    clock = pygame.time.Clock()
    running = True
    try:
        environment.render()
        pygame.display.set_caption(
            diagnostic_caption(environment, reward, info, completed_reason)
        )
        while running:
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    _, info = environment.reset()
                    reward = 0.0
                    completed_reason = None
                    reset_requested = True

            if not running:
                break
            if reset_requested:
                environment.render()
                pygame.display.set_caption(
                    diagnostic_caption(environment, reward, info, completed_reason)
                )
                clock.tick(round(1.0 / environment.config.simulation.agent_timestep))
                continue
            if completed_reason is None:
                _, reward, terminated, truncated, info = environment.step(
                    controls_from_keys(pygame.key.get_pressed())
                )
                completed_reason = terminal_reason(terminated, truncated, info)

            environment.render()
            pygame.display.set_caption(
                diagnostic_caption(environment, reward, info, completed_reason)
            )
            clock.tick(round(1.0 / environment.config.simulation.agent_timestep))
    finally:
        environment.close()


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and run the interactive manual driver.
    """
    arguments = build_parser().parse_args(argv)
    run_driver(build_environment(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
