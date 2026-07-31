"""Tests for Frenet projection, observations, preview, and progress."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from envs import (
    FrenetProjector,
    Track,
    TrackGenerationMetadata,
    TrackGeometry,
    signed_progress,
)


@pytest.fixture(scope="module")
def circle_geometry() -> TrackGeometry:
    sample_count = 3_200
    spacing = 0.5
    track_length = sample_count * spacing
    radius = track_length / (2.0 * pi)
    angles = np.arange(sample_count, dtype=np.float64) * 2.0 * pi / sample_count
    return TrackGeometry(
        Track(
            generation=TrackGenerationMetadata(
                seed=0,
                n_checkpoints=12,
                base_radius=radius,
                radial_jitter=0.0,
                angular_jitter=0.0,
                max_attempts=1,
            ),
            width=12.0,
            sample_spacing=spacing,
            track_length=track_length,
            start_index=0,
            s=np.arange(sample_count, dtype=np.float64) * spacing,
            x=radius * np.cos(angles),
            y=radius * np.sin(angles),
            heading=np.asarray(angles + pi / 2.0, dtype=np.float64),
            curvature=np.full(sample_count, 1.0 / radius, dtype=np.float64),
        )
    )


def _rectangle_geometry() -> TrackGeometry:
    points = np.asarray(
        [
            [-150.0, -100.0],
            [-50.0, -100.0],
            [50.0, -100.0],
            [150.0, -100.0],
            [150.0, 0.0],
            [150.0, 100.0],
            [50.0, 100.0],
            [-50.0, 100.0],
            [-150.0, 100.0],
            [-150.0, 0.0],
        ],
        dtype=np.float64,
    )
    headings = np.asarray(
        [0.0, 0.0, 0.0, pi / 2.0, pi / 2.0, pi, pi, pi, -pi / 2.0, -pi / 2.0],
        dtype=np.float64,
    )
    return TrackGeometry(
        Track(
            generation=TrackGenerationMetadata(
                seed=0,
                n_checkpoints=10,
                base_radius=150.0,
                radial_jitter=0.0,
                angular_jitter=0.0,
                max_attempts=1,
            ),
            width=12.0,
            sample_spacing=100.0,
            track_length=1_000.0,
            start_index=0,
            s=np.arange(10, dtype=np.float64) * 100.0,
            x=points[:, 0],
            y=points[:, 1],
            heading=headings,
            curvature=np.zeros(10, dtype=np.float64),
        )
    )


@pytest.mark.parametrize(
    ("s", "lateral_distance"),
    [(0.0, 0.0), (127.25, 2.0), (799.75, -2.5), (1_599.75, 1.5)],
)
def test_frenet_cartesian_round_trip_on_curve_and_seam(
    circle_geometry: TrackGeometry,
    s: float,
    lateral_distance: float,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(s, lateral_distance)

    result = projector.project(point)

    assert signed_progress(s, result.s, 1_600.0) == pytest.approx(0.0, abs=2e-3)
    assert result.lateral_distance == pytest.approx(lateral_distance, abs=2e-3)


def test_frenet_cartesian_round_trip_on_straight() -> None:
    projector = FrenetProjector(_rectangle_geometry())

    result = projector.project(projector.xy_from_frenet(100.0, 2.0))

    assert result.s == pytest.approx(100.0)
    assert result.lateral_distance == pytest.approx(2.0)


def test_lateral_distance_and_heading_error_signs(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)

    left = projector.project(projector.xy_from_frenet(0.0, 2.0))
    right = projector.project(projector.xy_from_frenet(0.0, -2.0))

    assert left.lateral_distance > 0
    assert right.lateral_distance < 0
    assert projector.heading_error(pi / 2.0 + 0.2, 0.0) == pytest.approx(0.2)
    assert projector.heading_error(pi / 2.0 - 0.2, 0.0) == pytest.approx(-0.2)


def test_temporally_coherent_projection_uses_local_window(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(2.0, 1.0)

    result = projector.project(point, previous_segment_index=0)

    assert not result.used_global_search
    assert result.segment_index <= projector.local_window


def test_implausible_local_projection_triggers_global_fallback(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(0.0, 0.0)
    opposite_segment = circle_geometry.centerline_projector.segment_count // 2

    result = projector.project(
        point,
        previous_segment_index=opposite_segment,
    )

    assert result.used_global_search
    assert result.segment_index in {
        0,
        circle_geometry.centerline_projector.segment_count - 1,
    }


@pytest.mark.parametrize("speed", [0.0, 20.0, 70.0])
def test_dynamic_preview_matches_constant_curvature(
    circle_geometry: TrackGeometry,
    speed: float,
) -> None:
    projector = FrenetProjector(circle_geometry)
    expected = 2.0 * pi / circle_geometry.track.track_length

    assert projector.curvature_preview(1_590.0, speed) == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_integrated_curvature_supports_seams_and_complete_laps(
    circle_geometry: TrackGeometry,
) -> None:
    expected = 2.0 * pi / circle_geometry.track.track_length
    distance = 2.5 * circle_geometry.track.track_length

    integral = circle_geometry.integrated_curvature(1_500.0, distance)

    assert integral == pytest.approx(expected * distance, rel=1e-12)


def test_observation_contains_documented_physical_values(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(50.0, -1.5)
    track_heading = circle_geometry.heading(50.0)

    observation, projection = projector.observation(
        point,
        vehicle_heading=track_heading + 0.1,
        speed=30.0,
    )

    assert observation.dtype == np.float64
    assert observation[0] == pytest.approx(-1.5, abs=2e-3)
    assert observation[1] == pytest.approx(0.1, abs=2e-3)
    assert observation[2] == pytest.approx(30.0)
    assert observation[3] == pytest.approx(2.0 * pi / 1_600.0)
    assert projection.used_global_search


@pytest.mark.parametrize(
    ("previous_s", "current_s", "expected"),
    [
        (1_599.0, 1.0, 2.0),
        (1.0, 1_599.0, -2.0),
        (100.0, 99.0, -1.0),
        (100.0, 101.0, 1.0),
    ],
)
def test_signed_progress_preserves_forward_and_backward_motion(
    previous_s: float,
    current_s: float,
    expected: float,
) -> None:
    assert signed_progress(previous_s, current_s, 1_600.0) == pytest.approx(expected)
