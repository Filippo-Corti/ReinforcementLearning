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
    spacing_m = 0.5
    track_length_m = sample_count * spacing_m
    radius_m = track_length_m / (2.0 * pi)
    angles = np.arange(sample_count, dtype=np.float64) * 2.0 * pi / sample_count
    track = Track(
        generation=TrackGenerationMetadata(
            seed=0,
            n_checkpoints=12,
            base_radius_m=radius_m,
            radial_jitter_fraction=0.0,
            angular_jitter_sectors=0.0,
            max_attempts=1,
        ),
        width_m=12.0,
        sample_spacing_m=spacing_m,
        track_length_m=track_length_m,
        start_index=0,
        s_m=np.arange(sample_count, dtype=np.float64) * spacing_m,
        x_m=radius_m * np.cos(angles),
        y_m=radius_m * np.sin(angles),
        heading_rad=np.asarray(angles + pi / 2.0, dtype=np.float64),
        curvature_per_m=np.full(
            sample_count,
            1.0 / radius_m,
            dtype=np.float64,
        ),
    )
    return TrackGeometry(track)


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
    track = Track(
        generation=TrackGenerationMetadata(
            seed=0,
            n_checkpoints=10,
            base_radius_m=150.0,
            radial_jitter_fraction=0.0,
            angular_jitter_sectors=0.0,
            max_attempts=1,
        ),
        width_m=12.0,
        sample_spacing_m=100.0,
        track_length_m=1_000.0,
        start_index=0,
        s_m=np.arange(10, dtype=np.float64) * 100.0,
        x_m=points[:, 0],
        y_m=points[:, 1],
        heading_rad=headings,
        curvature_per_m=np.zeros(10, dtype=np.float64),
    )
    return TrackGeometry(track)


@pytest.mark.parametrize(
    ("s_m", "lateral_distance_m"),
    [(0.0, 0.0), (127.25, 2.0), (799.75, -2.5), (1_599.75, 1.5)],
)
def test_frenet_cartesian_round_trip_on_curve_and_seam(
    circle_geometry: TrackGeometry,
    s_m: float,
    lateral_distance_m: float,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(s_m, lateral_distance_m)

    result = projector.project(point)

    assert signed_progress(s_m, result.s_m, 1_600.0) == pytest.approx(
        0.0,
        abs=2e-3,
    )
    assert result.lateral_distance_m == pytest.approx(
        lateral_distance_m,
        abs=2e-3,
    )


def test_frenet_cartesian_round_trip_on_straight() -> None:
    projector = FrenetProjector(_rectangle_geometry())
    point = projector.xy_from_frenet(100.0, 2.0)

    result = projector.project(point)

    assert result.s_m == pytest.approx(100.0)
    assert result.lateral_distance_m == pytest.approx(2.0)


def test_projection_onto_explicit_segment_subset() -> None:
    index = _rectangle_geometry().centerline_index

    result = index.project_candidates(np.asarray([-110.0, -97.0]), [0])

    assert result.segment_index == 0
    assert result.fraction == pytest.approx(0.4)
    assert result.distance_m == pytest.approx(3.0)


def test_lateral_distance_and_heading_error_signs(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    left_point = projector.xy_from_frenet(0.0, 2.0)
    right_point = projector.xy_from_frenet(0.0, -2.0)

    left = projector.project(left_point)
    right = projector.project(right_point)

    assert left.lateral_distance_m > 0
    assert right.lateral_distance_m < 0
    assert projector.heading_error(pi / 2.0 + 0.2, 0.0) == pytest.approx(0.2)
    assert projector.heading_error(pi / 2.0 - 0.2, 0.0) == pytest.approx(-0.2)


def test_temporally_coherent_projection_uses_local_window(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(2.0, 1.0)

    result = projector.project(point, previous_segment_index=0)

    assert not result.used_global_search
    assert result.segment_index <= projector.local_window_segments


def test_implausible_local_projection_triggers_global_fallback(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(0.0, 0.0)
    opposite_segment = circle_geometry.centerline_index.segment_count // 2

    result = projector.project(
        point,
        previous_segment_index=opposite_segment,
    )

    assert result.used_global_search
    assert result.segment_index in {
        0,
        circle_geometry.centerline_index.segment_count - 1,
    }


@pytest.mark.parametrize("speed_m_per_s", [0.0, 20.0, 70.0])
def test_dynamic_preview_matches_constant_curvature(
    circle_geometry: TrackGeometry,
    speed_m_per_s: float,
) -> None:
    projector = FrenetProjector(circle_geometry)
    expected = 2.0 * pi / circle_geometry.track.track_length_m

    assert projector.curvature_preview(
        1_590.0,
        speed_m_per_s,
    ) == pytest.approx(expected, rel=1e-12)


def test_integrated_curvature_supports_seams_and_complete_laps(
    circle_geometry: TrackGeometry,
) -> None:
    expected = 2.0 * pi / circle_geometry.track.track_length_m
    distance = 2.5 * circle_geometry.track.track_length_m

    integral = circle_geometry.integrated_curvature(1_500.0, distance)

    assert integral == pytest.approx(expected * distance, rel=1e-12)


def test_observation_contains_physical_values(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)
    point = projector.xy_from_frenet(50.0, -1.5)
    track_heading = circle_geometry.heading(50.0)

    observation, projection = projector.observation(
        point,
        vehicle_heading_rad=track_heading + 0.1,
        speed_m_per_s=30.0,
    )

    assert observation.dtype == np.float64
    assert observation[0] == pytest.approx(-1.5, abs=2e-3)
    assert observation[1] == pytest.approx(0.1, abs=2e-3)
    assert observation[2] == pytest.approx(30.0)
    assert observation[3] == pytest.approx(2.0 * pi / 1_600.0)
    assert projection.used_global_search


@pytest.mark.parametrize(
    ("previous_s_m", "current_s_m", "expected"),
    [
        (1_599.0, 1.0, 2.0),
        (1.0, 1_599.0, -2.0),
        (100.0, 99.0, -1.0),
        (100.0, 101.0, 1.0),
    ],
)
def test_signed_progress_preserves_forward_and_backward_motion(
    previous_s_m: float,
    current_s_m: float,
    expected: float,
) -> None:
    assert signed_progress(
        previous_s_m,
        current_s_m,
        1_600.0,
    ) == pytest.approx(expected)


def test_invalid_projection_and_observation_inputs_are_rejected(
    circle_geometry: TrackGeometry,
) -> None:
    projector = FrenetProjector(circle_geometry)

    with pytest.raises(ValueError, match="previous_segment_index"):
        projector.project(np.zeros(2), previous_segment_index=-1)
    with pytest.raises(ValueError, match="lateral_distance_m"):
        projector.xy_from_frenet(0.0, float("nan"))
    with pytest.raises(ValueError, match="speed_m_per_s"):
        projector.curvature_preview(0.0, -1.0)
    with pytest.raises(ValueError, match="speed_m_per_s"):
        projector.curvature_preview(0.0, 70.1)
    with pytest.raises(ValueError, match="distance_m"):
        circle_geometry.integrated_curvature(0.0, -1.0)
    with pytest.raises(ValueError, match="track_length_m positive"):
        signed_progress(0.0, 1.0, 0.0)


def test_segment_subset_validation(circle_geometry: TrackGeometry) -> None:
    index = circle_geometry.centerline_index

    with pytest.raises(ValueError, match="must not be empty"):
        index.project_candidates(np.zeros(2), [])
    with pytest.raises(ValueError, match="only integers"):
        index.project_candidates(np.zeros(2), [0.5])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="reference indexed"):
        index.project_candidates(np.zeros(2), [index.segment_count])
