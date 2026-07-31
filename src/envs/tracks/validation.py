"""Geometric validation for sampled racing tracks."""

from __future__ import annotations

from math import radians, tan

import numpy as np

from configs import CarConfig, TrackGenerationConfig

from .geometry import TrackGeometry, wrap_angle
from .model import Track, TrackValidationError
from .projection import FloatArray, PolylineProjector, project_to_segment


def validate_track_geometry(
    track: Track,
    *,
    vehicle_config: CarConfig | None = None,
    track_config: TrackGenerationConfig | None = None,
) -> TrackGeometry:
    """
    Validate track length, curvature, seams, intersections, and separation.
    """
    vehicle = vehicle_config or CarConfig()
    generation = track_config or TrackGenerationConfig()

    if not generation.min_length <= track.track_length <= generation.max_length:
        raise TrackValidationError(
            "track length must be within the configured generation range."
        )

    maximum_curvature = tan(radians(vehicle.max_steering_angle)) / vehicle.wheelbase
    if np.any(np.abs(track.curvature) > maximum_curvature + 1e-12):
        raise TrackValidationError(
            "track curvature exceeds the vehicle kinematic steering limit."
        )

    try:
        geometry = TrackGeometry(track)
    except ValueError as error:
        raise TrackValidationError(f"invalid track geometry: {error}") from error
    _validate_periodic_seam(geometry)
    _validate_simple_closed_polyline(geometry.centerline_projector, "centerline")

    required_separation = track.width + generation.nonlocal_centerline_margin
    _validate_nonlocal_centerline_separation(
        geometry.centerline_projector,
        sample_spacing=track.sample_spacing,
        required_separation=required_separation,
    )
    _validate_simple_closed_polyline(
        geometry.left_boundary_projector,
        "left boundary",
    )
    _validate_simple_closed_polyline(
        geometry.right_boundary_projector,
        "right boundary",
    )
    _validate_boundaries_do_not_intersect(
        geometry.left_boundary_projector,
        geometry.right_boundary_projector,
    )
    return geometry


def _validate_periodic_seam(geometry: TrackGeometry) -> None:
    length = geometry.track.track_length
    if not np.allclose(
        geometry.position(0.0),
        geometry.position(length),
        rtol=0.0,
        atol=1e-10,
    ):
        raise TrackValidationError("centerline position is not continuous at the seam.")
    if abs(wrap_angle(geometry.heading(length) - geometry.heading(0.0))) > 1e-10:
        raise TrackValidationError("centerline heading is not continuous at the seam.")
    if not np.isclose(
        geometry.curvature(0.0),
        geometry.curvature(length),
        rtol=0.0,
        atol=1e-10,
    ):
        raise TrackValidationError(
            "centerline curvature is not continuous at the seam."
        )


def _validate_simple_closed_polyline(
    index: PolylineProjector,
    name: str,
) -> None:
    pairs = index.candidate_pairs(0.0)
    for first, second in pairs:
        if _segments_are_adjacent(int(first), int(second), index.segment_count):
            continue
        if _segments_intersect(
            index.starts[first],
            index.ends[first],
            index.starts[second],
            index.ends[second],
        ):
            raise TrackValidationError(
                f"{name} segments {first} and {second} intersect."
            )


def _validate_nonlocal_centerline_separation(
    index: PolylineProjector,
    *,
    sample_spacing: float,
    required_separation: float,
) -> None:
    pairs = index.candidate_pairs(required_separation)
    for first, second in pairs:
        first_index = int(first)
        second_index = int(second)
        if _segments_are_adjacent(first_index, second_index, index.segment_count):
            continue
        index_delta = abs(first_index - second_index)
        cyclic_delta = min(index_delta, index.segment_count - index_delta)
        arc_separation = max(0.0, (cyclic_delta - 1) * sample_spacing)
        if arc_separation <= required_separation:
            continue
        distance = _segment_distance(
            index.starts[first],
            index.ends[first],
            index.starts[second],
            index.ends[second],
        )
        if distance <= required_separation:
            raise TrackValidationError(
                "nonlocal centerline segments "
                f"{first} and {second} are only {distance:.6g} m apart; "
                f"required separation is greater than {required_separation:.6g} m."
            )


def _validate_boundaries_do_not_intersect(
    left: PolylineProjector,
    right: PolylineProjector,
) -> None:
    candidate_lists = left.candidate_lists(right, 0.0)
    for left_index, right_indices in enumerate(candidate_lists):
        for right_index in right_indices:
            if _segments_intersect(
                left.starts[left_index],
                left.ends[left_index],
                right.starts[right_index],
                right.ends[right_index],
            ):
                raise TrackValidationError(
                    "left and right boundary segments "
                    f"{left_index} and {right_index} intersect."
                )


def _segments_are_adjacent(first: int, second: int, count: int) -> bool:
    difference = abs(first - second)
    return difference == 1 or difference == count - 1


def _segments_intersect(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
) -> bool:
    scale = max(
        1.0,
        float(np.max(np.abs(first_start))),
        float(np.max(np.abs(first_end))),
        float(np.max(np.abs(second_start))),
        float(np.max(np.abs(second_end))),
    )
    tolerance = np.finfo(np.float64).eps * scale * 64

    first_a = _cross(first_end - first_start, second_start - first_start)
    first_b = _cross(first_end - first_start, second_end - first_start)
    second_a = _cross(second_end - second_start, first_start - second_start)
    second_b = _cross(second_end - second_start, first_end - second_start)

    if first_a * first_b < -(tolerance**2) and second_a * second_b < -(tolerance**2):
        return True

    return (
        (
            abs(first_a) <= tolerance
            and _point_on_segment(second_start, first_start, first_end, tolerance)
        )
        or (
            abs(first_b) <= tolerance
            and _point_on_segment(second_end, first_start, first_end, tolerance)
        )
        or (
            abs(second_a) <= tolerance
            and _point_on_segment(first_start, second_start, second_end, tolerance)
        )
        or (
            abs(second_b) <= tolerance
            and _point_on_segment(first_end, second_start, second_end, tolerance)
        )
    )


def _segment_distance(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _point_segment_distance(
    point: FloatArray,
    start: FloatArray,
    end: FloatArray,
) -> float:
    _, _, distance = project_to_segment(point, start, end)
    return distance


def _point_on_segment(
    point: FloatArray,
    start: FloatArray,
    end: FloatArray,
    tolerance: float,
) -> bool:
    return bool(
        np.all(point >= np.minimum(start, end) - tolerance)
        and np.all(point <= np.maximum(start, end) + tolerance)
    )


def _cross(first: FloatArray, second: FloatArray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])
