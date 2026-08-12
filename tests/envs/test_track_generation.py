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


def test_same_seed_produces_identical_track_data() -> None:
    first = generate_track(7)
    second = generate_track(7)

    assert first.to_dict() == second.to_dict()


def test_generated_track_is_valid_and_uniformly_sampled() -> None:
    track = generate_track(0)

    assert validate_track_geometry(track).track is track
    assert 300.0 <= track.track_length <= 700.0
    assert track.sample_spacing == pytest.approx(0.5)
    np.testing.assert_allclose(
        track.s,
        np.arange(track.s.size, dtype=np.float64) * track.sample_spacing,
    )


def test_circuit_alternates_between_straights_and_constant_radius_corners() -> None:
    """
    The property the generator exists for: a lap has somewhere to build speed.

    A spline through checkpoints cannot hold zero curvature for the length of a
    straight, so the circuit it produced curved almost everywhere and gave the
    car nothing to brake for.
    """
    config = TrackGenerationConfig()

    for seed in range(5):
        curvature = generate_track(seed).curvature
        straight = np.abs(curvature) <= 1e-12

        assert straight.mean() > 0.4
        # Curvature is piecewise constant, so a corner is a run of one value
        # rather than a continuously varying stretch.
        corner_radii = 1.0 / np.abs(curvature[~straight])
        # The radius band is applied before the circuit is rescaled to a whole
        # number of samples, and that rescale is bounded by half a sample
        # interval, so a radius may sit a fraction of a percent outside it.
        assert corner_radii.min() >= config.min_corner_radius * 0.999
        assert corner_radii.max() <= config.max_corner_radius * 1.001
        distinct = np.unique(np.round(curvature, 9))
        assert 3 <= distinct.size <= config.n_corners + 1


def test_start_line_sits_in_the_middle_of_a_straight() -> None:
    """
    A seam on a straight is both realistic and periodic in curvature for free.
    """
    for seed in range(5):
        track = generate_track(seed)

        assert track.start_index == 0
        assert track.curvature[0] == pytest.approx(0.0, abs=1e-12)
        assert track.curvature[-1] == pytest.approx(0.0, abs=1e-12)


def test_circuits_turn_in_both_directions() -> None:
    """
    A circuit whose corners all turn the same way is an oval, not a racetrack.
    """
    directions = set()
    for seed in range(5):
        curvature = generate_track(seed).curvature
        directions.update(np.unique(np.sign(curvature[np.abs(curvature) > 1e-12])))

    assert directions == {-1.0, 1.0}


def test_different_seeds_produce_different_geometry() -> None:
    first = generate_track(10)
    second = generate_track(11)

    assert not np.array_equal(first.x, second.x)


def test_generation_does_not_change_numpy_global_random_state() -> None:
    np.random.seed(1234)
    expected = np.random.random(2)
    np.random.seed(1234)

    first = np.random.random()
    generate_track(3)
    second = np.random.random()

    np.testing.assert_array_equal([first, second], expected)


def test_retry_exhaustion_reports_attempt_count() -> None:
    config = TrackGenerationConfig(
        max_attempts=2,
        min_length=1_000.0,
        max_length=1_001.0,
    )

    with pytest.raises(TrackGenerationError, match="after 2 attempts"):
        generate_track(0, track_config=config)


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
