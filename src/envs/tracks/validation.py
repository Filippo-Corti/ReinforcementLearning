"""Geometric validation for sampled racing tracks."""

from __future__ import annotations

from dataclasses import replace
from math import pi, radians, tan

import numpy as np

from configs import CarConfig, TrackGenerationConfig

from ..geometry import (
    PolylineProjector,
    segment_distance,
    segments_intersect,
    wrap_angle,
)
from .errors import TrackValidationError
from .track import Track, TrackWithGeometry


def validate_track_geometry(
    track: Track,
    *,
    vehicle_config: CarConfig | None = None,
    track_config: TrackGenerationConfig | None = None,
) -> TrackWithGeometry:
    """
    Validate track length, curvature, seams, intersections, and separation.
    """
    vehicle = vehicle_config or CarConfig()
    generation = track_config
    if generation is None:
        # Saved files retain their generation radius but not the acceptance
        # interval. Scale the default interval with that recorded radius so
        # both the current short task and earlier long circuits remain loadable.
        default = TrackGenerationConfig()
        scale = track.generation.base_radius / default.base_radius
        generation = replace(
            default,
            base_radius=track.generation.base_radius,
            min_length=default.min_length * scale,
            max_length=default.max_length * scale,
        )

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
        geometry = TrackWithGeometry(track)
    except ValueError as error:
        raise TrackValidationError(f"invalid track geometry: {error}") from error
    _validate_periodic_seam(geometry)
    _validate_simple_closed_polyline(geometry.centerline_projector, "centerline")

    required_separation = track.width + generation.nonlocal_centerline_margin
    _validate_nonlocal_centerline_separation(
        geometry.centerline_projector,
        sample_spacing=track.sample_spacing,
        required_separation=required_separation,
        # Half a turn at the tightest permitted radius: within that arc length a
        # corner cannot have come back around towards itself.
        local_arc_separation=max(
            required_separation, pi * generation.min_corner_radius
        ),
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


def _validate_periodic_seam(geometry: TrackWithGeometry) -> None:
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
        if segments_intersect(
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
    local_arc_separation: float,
) -> None:
    """
    Reject a centerline that runs close to a *different* part of itself.

    Two samples on one tight corner are legitimately close in space: the chord
    across an arc is shorter than the arc, and at the tightest permitted radius
    it falls below the required separation well before the corner has turned far
    enough to be considered another part of the circuit. Pairs closer than
    ``local_arc_separation`` along the track are therefore skipped, so this rule
    sees folding rather than cornering.
    """
    pairs = index.candidate_pairs(required_separation)
    for first, second in pairs:
        first_index = int(first)
        second_index = int(second)
        if _segments_are_adjacent(first_index, second_index, index.segment_count):
            continue
        index_delta = abs(first_index - second_index)
        cyclic_delta = min(index_delta, index.segment_count - index_delta)
        arc_separation = max(0.0, (cyclic_delta - 1) * sample_spacing)
        if arc_separation <= local_arc_separation:
            continue
        distance = segment_distance(
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
            if segments_intersect(
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
    """
    Return whether two segments are neighbours on a closed polyline.
    """
    difference = abs(first - second)
    return difference == 1 or difference == count - 1
