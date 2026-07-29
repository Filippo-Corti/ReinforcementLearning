"""Tests for periodic track queries, projection, and geometric validation."""

from __future__ import annotations

from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest

from configs import TrackGenerationConfig
from envs import (
    SegmentIndex,
    Track,
    TrackGenerationMetadata,
    TrackGeometry,
    TrackValidationError,
    validate_track_geometry,
    wrap_angle,
)


FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "tracks" / "valid_circle.json"
)


def _circle() -> Track:
    return Track.load(FIXTURE_PATH)


def _polyline_track(points_m: list[tuple[float, float]]) -> Track:
    points = np.asarray(points_m, dtype=np.float64)
    count = len(points)
    spacing = 1_000.0 / count
    directions = np.roll(points, -1, axis=0) - points
    headings = np.arctan2(directions[:, 1], directions[:, 0])
    return Track(
        generation=TrackGenerationMetadata(
            seed=0,
            n_checkpoints=count,
            base_radius_m=100.0,
            radial_jitter_fraction=0.0,
            angular_jitter_sectors=0.0,
            max_attempts=1,
        ),
        width_m=12.0,
        sample_spacing_m=spacing,
        track_length_m=1_000.0,
        start_index=0,
        s_m=np.arange(count, dtype=np.float64) * spacing,
        x_m=points[:, 0],
        y_m=points[:, 1],
        heading_rad=headings,
        curvature_per_m=np.zeros(count, dtype=np.float64),
    )


def test_circle_queries_match_sampled_analytic_geometry() -> None:
    track = _circle()
    geometry = TrackGeometry(track)
    radius = 800.0 / pi

    np.testing.assert_allclose(
        geometry.position(0.0),
        [radius, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    assert geometry.heading(0.0) == pytest.approx(pi / 2.0)
    np.testing.assert_allclose(
        geometry.normal(0.0),
        [-1.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    assert geometry.curvature(0.0) == pytest.approx(1.0 / radius)


def test_queries_are_periodic_across_the_seam() -> None:
    track = _circle()
    geometry = validate_track_geometry(track)
    offsets = (0.0, 37.5, track.track_length_m - 0.25)

    for offset in offsets:
        np.testing.assert_allclose(
            geometry.position(offset),
            geometry.position(offset + track.track_length_m),
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            geometry.normal(offset),
            geometry.normal(offset + track.track_length_m),
            rtol=0.0,
            atol=1e-10,
        )
        assert geometry.heading(offset) == pytest.approx(
            geometry.heading(offset + track.track_length_m),
            abs=1e-10,
        )
        assert geometry.curvature(offset) == pytest.approx(
            geometry.curvature(offset + track.track_length_m),
            abs=1e-10,
        )


def test_boundary_convention_places_left_toward_circle_interior() -> None:
    track = _circle()
    geometry = TrackGeometry(track)
    radius = 800.0 / pi

    np.testing.assert_allclose(
        geometry.left_boundary_position(0.0),
        [radius - track.width_m / 2.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        geometry.right_boundary_position(0.0),
        [radius + track.width_m / 2.0, 0.0],
        atol=1e-12,
    )
    assert geometry.left_boundary_m.dtype == np.float64
    assert geometry.right_boundary_m.dtype == np.float64
    assert not geometry.left_boundary_m.flags.writeable
    assert not geometry.right_boundary_m.flags.writeable


def test_segment_index_finds_global_closest_projection() -> None:
    index = SegmentIndex(
        np.asarray(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
            dtype=np.float64,
        )
    )

    projection = index.project(np.asarray([6.0, 3.0]))

    assert projection.segment_index == 0
    assert projection.fraction == pytest.approx(0.6)
    assert projection.distance_m == pytest.approx(3.0)
    np.testing.assert_allclose(projection.point_m, [6.0, 0.0])
    assert index.segment_count == 4
    assert index.starts_m.dtype == np.float64
    assert not index.lengths_m.flags.writeable
    assert index.candidate_pairs(0.0).shape[1] == 2


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([[0.0, 0.0], [1.0, 1.0]], "at least 3"),
        (
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            "zero-length",
        ),
    ],
)
def test_segment_index_rejects_invalid_polylines(
    points: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SegmentIndex(np.asarray(points))


def test_segment_index_rejects_invalid_queries() -> None:
    index = SegmentIndex(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )

    with pytest.raises(ValueError, match="shape"):
        index.project(np.asarray([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        index.candidate_pairs(-1.0)


def test_wrap_angle_uses_half_open_principal_interval() -> None:
    assert wrap_angle(pi) == pytest.approx(-pi)
    assert wrap_angle(-3.0 * pi) == pytest.approx(-pi)
    assert wrap_angle(5.0 * pi / 2.0) == pytest.approx(pi / 2.0)
    with pytest.raises(ValueError, match="finite"):
        wrap_angle(float("nan"))


def test_geometry_queries_reject_non_finite_arc_length() -> None:
    geometry = TrackGeometry(_circle())

    with pytest.raises(ValueError, match="finite"):
        geometry.position(float("inf"))


def test_length_outside_configured_range_is_rejected() -> None:
    track = _polyline_track(
        [(-100.0, -100.0), (100.0, -100.0), (100.0, 100.0), (-100.0, 100.0)]
    )
    config = TrackGenerationConfig(min_length_m=1_100.0)

    with pytest.raises(TrackValidationError, match="generation range"):
        validate_track_geometry(track, track_config=config)


def test_curvature_beyond_vehicle_limit_is_rejected() -> None:
    track = replace(
        _circle(),
        curvature_per_m=np.full(8, 1.0, dtype=np.float64),
    )

    with pytest.raises(TrackValidationError, match="steering limit"):
        validate_track_geometry(track)


def test_self_intersecting_centerline_is_rejected() -> None:
    track = _polyline_track(
        [(-100.0, -100.0), (100.0, 100.0), (-100.0, 100.0), (100.0, -100.0)]
    )

    with pytest.raises(TrackValidationError, match="centerline segments"):
        validate_track_geometry(track)


def test_insufficient_nonlocal_separation_is_rejected() -> None:
    track = _polyline_track(
        [(-100.0, 0.0), (100.0, 0.0), (100.0, 10.0), (-100.0, 10.0)]
    )

    with pytest.raises(TrackValidationError, match="nonlocal centerline"):
        validate_track_geometry(track)


def test_intersecting_boundaries_are_rejected() -> None:
    track = _polyline_track(
        [(0.0, 0.0), (200.0, 0.0), (1.0, 20.0)]
    )

    with pytest.raises(
        TrackValidationError,
        match="left and right boundary segments",
    ):
        validate_track_geometry(track)


def test_load_and_save_validate_geometry_by_default(tmp_path: Path) -> None:
    invalid = replace(
        _circle(),
        curvature_per_m=np.full(8, 1.0, dtype=np.float64),
    )
    destination = tmp_path / "overcurved.json"

    with pytest.raises(TrackValidationError, match="steering limit"):
        invalid.save(destination)

    invalid.save(destination, validate_geometry=False)
    with pytest.raises(TrackValidationError, match="steering limit"):
        Track.load(destination)
    restored = Track.load(destination, validate_geometry=False)
    np.testing.assert_array_equal(
        restored.curvature_per_m,
        invalid.curvature_per_m,
    )
