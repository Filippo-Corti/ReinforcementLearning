"""Generate and save a deterministic racing track."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import TrackGenerationConfig
from envs.tracks import generate_track_file


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """
    defaults = TrackGenerationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination JSON track file")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoints", type=int, default=defaults.n_checkpoints)
    parser.add_argument("--radius", type=float, default=defaults.base_radius)
    parser.add_argument(
        "--radial-jitter",
        type=float,
        default=defaults.radial_jitter,
    )
    parser.add_argument(
        "--angular-jitter",
        type=float,
        default=defaults.angular_jitter,
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=defaults.sample_spacing,
    )
    parser.add_argument("--width", type=float, default=defaults.width)
    parser.add_argument("--max-attempts", type=int, default=defaults.max_attempts)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Generate a track from command-line arguments.
    """
    arguments = build_parser().parse_args(argv)
    config = TrackGenerationConfig(
        n_checkpoints=arguments.checkpoints,
        base_radius=arguments.radius,
        radial_jitter=arguments.radial_jitter,
        angular_jitter=arguments.angular_jitter,
        sample_spacing=arguments.spacing,
        width=arguments.width,
        max_attempts=arguments.max_attempts,
    )
    track = generate_track_file(
        arguments.output,
        seed=arguments.seed,
        track_config=config,
    )
    print(
        f"saved {track.track_length:.1f} m track with "
        f"{track.s.size} samples to {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
