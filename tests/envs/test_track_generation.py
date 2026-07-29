"""Tests for deterministic procedural track generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configs import TrackGenerationConfig
from envs import (
    Track,
    TrackGenerationError,
    generate_track,
    generate_track_file,
    validate_track_geometry,
)
from experiments.generate_track import build_parser, main


def test_same_seed_and_configuration_produce_identical_track_data() -> None:
    first = generate_track(7)
    second = generate_track(7)

    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fixed_seeds_produce_valid_tracks(seed: int) -> None:
    track = generate_track(seed)

    geometry = validate_track_geometry(track)

    assert geometry.track is track
    assert 1_000.0 <= track.track_length_m <= 3_000.0
    assert track.sample_spacing_m == pytest.approx(0.5)
    assert track.s_m.dtype == np.float64


def test_different_seeds_produce_different_geometry() -> None:
    first = generate_track(10)
    second = generate_track(11)

    assert not np.array_equal(first.x_m, second.x_m)


def test_generation_does_not_change_numpy_global_random_state() -> None:
    np.random.seed(1234)
    expected = np.random.random(2)
    np.random.seed(1234)

    first = np.random.random()
    generate_track(3)
    second = np.random.random()

    np.testing.assert_array_equal([first, second], expected)


def test_retry_exhaustion_reports_attempt_count_and_constraint() -> None:
    config = TrackGenerationConfig(
        max_attempts=2,
        min_length_m=1_000.0,
        max_length_m=1_001.0,
    )

    with pytest.raises(
        TrackGenerationError,
        match=r"after 2 attempts: 2x track length",
    ):
        generate_track(0, track_config=config)


@pytest.mark.parametrize("seed", [-1, True])
def test_invalid_seed_is_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        generate_track(seed)


def test_generated_file_loads_through_public_loader(tmp_path: Path) -> None:
    destination = tmp_path / "generated.json"

    generated = generate_track_file(destination, seed=5)
    loaded = Track.load(destination)

    assert loaded.to_dict() == generated.to_dict()


def test_command_line_parser_requires_seed_and_output() -> None:
    arguments = build_parser().parse_args(["track.json", "--seed", "9"])

    assert arguments.output == Path("track.json")
    assert arguments.seed == 9


def test_command_line_entry_point_writes_track(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "cli-track.json"

    result = main([str(destination), "--seed", "12"])

    assert result == 0
    assert destination.is_file()
    assert "saved" in capsys.readouterr().out
    Track.load(destination)
