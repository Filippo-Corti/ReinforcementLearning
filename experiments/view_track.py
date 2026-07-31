"""Open an interactive visualization of a generated or saved racing track."""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from envs import RacingEnv


def build_parser() -> argparse.ArgumentParser:
    """
    Build the track-viewer command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=int, help="generate and display this track seed")
    source.add_argument("--track", type=Path, help="load and display this track JSON")
    return parser


def build_environment(arguments: argparse.Namespace) -> RacingEnv:
    """
    Construct the rendering environment selected by parsed arguments.
    """
    if arguments.track is not None:
        return RacingEnv(track_path=arguments.track, render_mode="human")
    return RacingEnv(track_seed=arguments.seed, render_mode="human")


def run_viewer(environment: RacingEnv) -> None:
    """
    Display the reset pose until Escape or the window close button is pressed.
    """
    environment.reset()
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            environment.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            clock.tick(round(1.0 / environment.config.simulation.agent_timestep))
    finally:
        environment.close()


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and run the interactive track viewer.
    """
    arguments = build_parser().parse_args(argv)
    run_viewer(build_environment(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
