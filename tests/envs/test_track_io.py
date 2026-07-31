"""Tests for sampled-track persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs import Track, TrackValidationError

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "tracks"
VALID_TRACK_PATH = FIXTURE_DIRECTORY / "valid_circle.json"


def test_valid_track_loads_with_expected_data() -> None:
    track = Track.load(VALID_TRACK_PATH)

    assert track.generation.seed == 0
    assert track.width == 12.0
    assert track.sample_spacing == 200.0
    assert track.track_length == 1_600.0
    assert track.start_index == 0
    assert track.s.dtype == np.float64
    np.testing.assert_array_equal(
        track.s,
        np.arange(8, dtype=np.float64) * 200.0,
    )


def test_save_load_round_trip_preserves_track_data(tmp_path: Path) -> None:
    original = Track.load(VALID_TRACK_PATH)
    destination = tmp_path / "track.json"

    original.save(destination)
    restored = Track.load(destination)

    assert restored.to_dict() == original.to_dict()


def test_serialization_is_byte_stable(tmp_path: Path) -> None:
    track = Track.load(VALID_TRACK_PATH)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    track.save(first)
    Track.load(first).save(second)

    assert first.read_bytes() == second.read_bytes()


def test_unknown_format_version_is_rejected() -> None:
    with pytest.raises(TrackValidationError, match="unsupported format_version"):
        Track.load(
            FIXTURE_DIRECTORY / "invalid_unknown_version.json",
            validate_geometry=False,
        )


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(TrackValidationError, match="is not valid JSON"):
        Track.load(FIXTURE_DIRECTORY / "invalid_json.json")


def test_missing_file_preserves_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Track.load(tmp_path / "missing.json")
