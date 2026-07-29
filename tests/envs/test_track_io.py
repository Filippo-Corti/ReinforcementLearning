"""Tests for sampled-track validation and JSON persistence."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from envs import (
    Track,
    TrackValidationError,
    UnsupportedTrackFormatError,
)


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "tracks"
VALID_TRACK_PATH = FIXTURE_DIRECTORY / "valid_circle.json"


def _valid_track_dict() -> dict[str, Any]:
    return json.loads(VALID_TRACK_PATH.read_text(encoding="utf-8"))


def test_valid_circle_fixture_loads_into_read_only_arrays() -> None:
    track = Track.load(VALID_TRACK_PATH)

    assert track.FORMAT_VERSION == 1
    assert track.units.to_dict() == {
        "length": "m",
        "angle": "rad",
        "curvature": "1/m",
    }
    assert track.generation.seed == 0
    assert track.width_m == 12.0
    assert track.sample_spacing_m == 200.0
    assert track.track_length_m == 1600.0
    assert track.start_index == 0
    assert track.s_m.dtype == np.float64
    assert track.s_m.shape == (8,)
    assert not track.s_m.flags.writeable
    np.testing.assert_allclose(
        track.s_m,
        np.arange(8, dtype=np.float64) * 200.0,
    )


def test_save_load_round_trip_preserves_runtime_data(tmp_path: Path) -> None:
    original = Track.load(VALID_TRACK_PATH)
    destination = tmp_path / "nested" / "circle.json"

    original.save(destination)
    restored = Track.load(destination)

    assert restored.to_dict() == original.to_dict()
    for field_name in (
        "s_m",
        "x_m",
        "y_m",
        "heading_rad",
        "curvature_per_m",
    ):
        np.testing.assert_array_equal(
            getattr(restored, field_name),
            getattr(original, field_name),
        )


def test_serialization_is_byte_stable(tmp_path: Path) -> None:
    first_track = Track.load(VALID_TRACK_PATH)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_track.save(first_path)
    Track.load(first_path).save(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")


def test_constructor_copies_input_arrays_and_makes_them_read_only() -> None:
    original = Track.load(VALID_TRACK_PATH)
    source = original.x_m.copy()

    copied = replace(original, x_m=source)
    source[0] = 99.0

    assert copied.x_m[0] == original.x_m[0]
    with pytest.raises(ValueError, match="read-only"):
        copied.x_m[0] = 99.0


def test_unknown_format_version_is_rejected() -> None:
    with pytest.raises(
        UnsupportedTrackFormatError,
        match="unsupported format_version 2",
    ):
        Track.load(FIXTURE_DIRECTORY / "invalid_unknown_version.json")


@pytest.mark.parametrize(
    ("fixture_name", "message"),
    [
        ("invalid_missing_samples.json", "missing keys: samples"),
        (
            "invalid_non_increasing_samples.json",
            "sample arc lengths must be strictly increasing",
        ),
        (
            "invalid_closing_segment.json",
            "track_length must include one sample_spacing closing segment",
        ),
        ("invalid_json.json", "is not valid JSON"),
    ],
)
def test_invalid_fixtures_are_rejected(
    fixture_name: str,
    message: str,
) -> None:
    with pytest.raises(TrackValidationError, match=message):
        Track.load(FIXTURE_DIRECTORY / fixture_name)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["units"].update(angle="deg"),
            "units must be length='m'",
        ),
        (
            lambda data: data["generation"].update(seed=-1),
            "generation.seed",
        ),
        (
            lambda data: data["generation"].update(n_checkpoints=True),
            "generation.n_checkpoints must be an integer",
        ),
        (
            lambda data: data.update(width=0.0),
            "width must be finite and positive",
        ),
        (
            lambda data: data.update(sample_spacing=float("inf")),
            "sample_spacing must be finite",
        ),
        (
            lambda data: data.update(start_index=8),
            "start_index must reference an existing sample",
        ),
        (
            lambda data: data["samples"][0].update(x=float("nan")),
            r"samples\[0\]\.x must be finite",
        ),
        (
            lambda data: data["samples"][1].update(s=1.5),
            r"sample arc lengths must equal index \* sample_spacing",
        ),
        (
            lambda data: data["samples"][-1].update(
                x=data["samples"][0]["x"],
                y=data["samples"][0]["y"],
            ),
            "final sample must not duplicate the first",
        ),
        (
            lambda data: data.update(unexpected=True),
            "unexpected keys: unexpected",
        ),
        (
            lambda data: data.update(samples="not-an-array"),
            "samples must be an array",
        ),
    ],
)
def test_invalid_decoded_data_is_rejected(
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    data = copy.deepcopy(_valid_track_dict())
    mutate(data)

    with pytest.raises(TrackValidationError, match=message):
        Track.from_dict(data)


def test_direct_constructor_rejects_mismatched_array_lengths() -> None:
    track = Track.load(VALID_TRACK_PATH)

    with pytest.raises(
        TrackValidationError,
        match="all sample arrays must have the same length",
    ):
        replace(track, curvature_per_m=track.curvature_per_m[:-1])


def test_direct_constructor_rejects_non_numeric_arrays() -> None:
    track = Track.load(VALID_TRACK_PATH)

    with pytest.raises(
        TrackValidationError,
        match="x_m must contain numeric values",
    ):
        replace(track, x_m=["not-a-number"])


def test_missing_file_preserves_file_not_found_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        Track.load(missing)
