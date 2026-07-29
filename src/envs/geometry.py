"""Periodic track interpolation, segment search, and geometric validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, radians, tan

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from configs import TrackGenerationConfig, VehicleConfig

from .track import Track, TrackValidationError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SegmentProjection:
    """Closest-point result for one segment in a closed polyline."""

    segment_index: int
    fraction: float
    point_m: FloatArray = field(repr=False, compare=False)
    distance_m: float


class SegmentIndex:
    """Global exact nearest-segment search backed by a midpoint KD-tree."""

    def __init__(self, points_m: FloatArray) -> None:
        points = np.array(points_m, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_m must have shape (n, 2).")
        if points.shape[0] < 3:
            raise ValueError("points_m must contain at least 3 points.")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_m must contain only finite values.")

        ends = np.roll(points, shift=-1, axis=0)
        lengths = np.linalg.norm(ends - points, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("closed polylines cannot contain zero-length segments.")

        points.setflags(write=False)
        ends.setflags(write=False)
        lengths.setflags(write=False)
        midpoints = (points + ends) / 2.0
        midpoints.setflags(write=False)

        self._starts_m = points
        self._ends_m = ends
        self._lengths_m = lengths
        self._midpoints_m = midpoints
        self._max_half_length_m = float(np.max(lengths) / 2.0)
        self._tree = cKDTree(midpoints)

    @property
    def starts_m(self) -> FloatArray:
        """Read-only segment start points."""
        return self._starts_m

    @property
    def ends_m(self) -> FloatArray:
        """Read-only segment end points."""
        return self._ends_m

    @property
    def lengths_m(self) -> FloatArray:
        """Read-only Euclidean segment lengths."""
        return self._lengths_m

    @property
    def segment_count(self) -> int:
        """Number of segments in the closed polyline."""
        return self._starts_m.shape[0]

    def project(self, point_m: FloatArray) -> SegmentProjection:
        """Return the exact closest projection over all indexed segments."""
        point = _point_array(point_m, "point_m")
        _, nearest_midpoint = self._tree.query(point, k=1)
        initial_index = int(nearest_midpoint)
        _, _, initial_distance = _project_to_segment(
            point,
            self._starts_m[initial_index],
            self._ends_m[initial_index],
        )

        search_radius = initial_distance + self._max_half_length_m
        candidates = sorted(self._tree.query_ball_point(point, search_radius))
        best: SegmentProjection | None = None
        for index in candidates:
            projection, fraction, distance = _project_to_segment(
                point,
                self._starts_m[index],
                self._ends_m[index],
            )
            candidate = SegmentProjection(
                segment_index=index,
                fraction=fraction,
                point_m=projection,
                distance_m=distance,
            )
            if best is None or (distance, index) < (
                best.distance_m,
                best.segment_index,
            ):
                best = candidate

        if best is None:
            raise RuntimeError("segment index did not return any candidates.")
        return best

    def candidate_pairs(self, maximum_distance_m: float) -> NDArray[np.int64]:
        """Return segment-index pairs that may be within a target distance."""
        if not isfinite(maximum_distance_m) or maximum_distance_m < 0:
            raise ValueError("maximum_distance_m must be finite and non-negative.")
        search_radius = maximum_distance_m + 2.0 * self._max_half_length_m
        pairs = self._tree.query_pairs(search_radius, output_type="ndarray")
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64)


class TrackGeometry:
    """Periodic interpolators and derived geometry for sampled track data."""

    def __init__(self, track: Track) -> None:
        self.track = track
        extended_s = np.append(track.s_m, track.track_length_m)
        extended_x = np.append(track.x_m, track.x_m[0])
        extended_y = np.append(track.y_m, track.y_m[0])
        extended_curvature = np.append(
            track.curvature_per_m,
            track.curvature_per_m[0],
        )

        self._x_spline = CubicSpline(
            extended_s,
            extended_x,
            bc_type="periodic",
        )
        self._y_spline = CubicSpline(
            extended_s,
            extended_y,
            bc_type="periodic",
        )
        self._curvature_spline = CubicSpline(
            extended_s,
            extended_curvature,
            bc_type="periodic",
        )

        unwrapped_heading = np.unwrap(track.heading_rad)
        closing_turn = wrap_angle(float(track.heading_rad[0] - track.heading_rad[-1]))
        self._heading_s = extended_s
        self._heading_unwrapped = np.append(
            unwrapped_heading,
            unwrapped_heading[-1] + closing_turn,
        )

        normals = np.column_stack(
            (
                -np.sin(track.heading_rad),
                np.cos(track.heading_rad),
            )
        )
        centerline = np.column_stack((track.x_m, track.y_m))
        self._left_boundary_m = centerline + (track.width_m / 2.0) * normals
        self._right_boundary_m = centerline - (track.width_m / 2.0) * normals
        self._left_boundary_m.setflags(write=False)
        self._right_boundary_m.setflags(write=False)

        self.centerline_index = SegmentIndex(centerline)
        self.left_boundary_index = SegmentIndex(self._left_boundary_m)
        self.right_boundary_index = SegmentIndex(self._right_boundary_m)

    @property
    def left_boundary_m(self) -> FloatArray:
        """Read-only sampled left boundary array."""
        return self._left_boundary_m

    @property
    def right_boundary_m(self) -> FloatArray:
        """Read-only sampled right boundary array."""
        return self._right_boundary_m

    def position(self, s_m: float) -> FloatArray:
        """Interpolate centerline position periodically at arc length ``s_m``."""
        wrapped_s = self._wrapped_s(s_m)
        return np.array(
            [self._x_spline(wrapped_s), self._y_spline(wrapped_s)],
            dtype=np.float64,
        )

    def heading(self, s_m: float) -> float:
        """Interpolate wrapped centerline heading at arc length ``s_m``."""
        wrapped_s = self._wrapped_s(s_m)
        unwrapped = float(
            np.interp(
                wrapped_s,
                self._heading_s,
                self._heading_unwrapped,
            )
        )
        return wrap_angle(unwrapped)

    def normal(self, s_m: float) -> FloatArray:
        """Return the unit normal pointing left of the centerline tangent."""
        heading = self.heading(s_m)
        return np.array(
            [-np.sin(heading), np.cos(heading)],
            dtype=np.float64,
        )

    def curvature(self, s_m: float) -> float:
        """Interpolate local curvature periodically at arc length ``s_m``."""
        wrapped_s = self._wrapped_s(s_m)
        return float(self._curvature_spline(wrapped_s))

    def left_boundary_position(self, s_m: float) -> FloatArray:
        """Interpolate the left boundary at arc length ``s_m``."""
        return self.position(s_m) + (self.track.width_m / 2.0) * self.normal(s_m)

    def right_boundary_position(self, s_m: float) -> FloatArray:
        """Interpolate the right boundary at arc length ``s_m``."""
        return self.position(s_m) - (self.track.width_m / 2.0) * self.normal(s_m)

    def _wrapped_s(self, s_m: float) -> float:
        if not isfinite(s_m):
            raise ValueError("s_m must be finite.")
        return float(s_m % self.track.track_length_m)


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to the half-open interval ``[-pi, pi)``."""
    if not isfinite(angle_rad):
        raise ValueError("angle_rad must be finite.")
    return float((angle_rad + pi) % (2.0 * pi) - pi)


def validate_track_geometry(
    track: Track,
    *,
    vehicle_config: VehicleConfig | None = None,
    track_config: TrackGenerationConfig | None = None,
) -> TrackGeometry:
    """Validate geometric constraints and return prepared track geometry."""
    vehicle = vehicle_config or VehicleConfig()
    generation = track_config or TrackGenerationConfig()

    if not generation.min_length_m <= track.track_length_m <= generation.max_length_m:
        raise TrackValidationError(
            "track length must be within the configured generation range."
        )

    maximum_curvature = tan(radians(vehicle.max_steering_angle_deg)) / (
        vehicle.wheelbase_m
    )
    if np.any(np.abs(track.curvature_per_m) > maximum_curvature + 1e-12):
        raise TrackValidationError(
            "track curvature exceeds the vehicle kinematic steering limit."
        )

    try:
        geometry = TrackGeometry(track)
    except ValueError as error:
        raise TrackValidationError(f"invalid track geometry: {error}") from error
    _validate_periodic_seam(geometry)
    _validate_simple_closed_polyline(
        geometry.centerline_index,
        "centerline",
    )

    required_separation = track.width_m + generation.nonlocal_centerline_margin_m
    _validate_nonlocal_centerline_separation(
        geometry.centerline_index,
        sample_spacing_m=track.sample_spacing_m,
        required_separation_m=required_separation,
    )

    _validate_simple_closed_polyline(
        geometry.left_boundary_index,
        "left boundary",
    )
    _validate_simple_closed_polyline(
        geometry.right_boundary_index,
        "right boundary",
    )
    _validate_boundaries_do_not_intersect(
        geometry.left_boundary_index,
        geometry.right_boundary_index,
    )
    return geometry


def _validate_periodic_seam(geometry: TrackGeometry) -> None:
    length = geometry.track.track_length_m
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
    index: SegmentIndex,
    name: str,
) -> None:
    pairs = index.candidate_pairs(0.0)
    for first, second in pairs:
        if _segments_are_adjacent(
            int(first),
            int(second),
            index.segment_count,
        ):
            continue
        if _segments_intersect(
            index.starts_m[first],
            index.ends_m[first],
            index.starts_m[second],
            index.ends_m[second],
        ):
            raise TrackValidationError(
                f"{name} segments {first} and {second} intersect."
            )


def _validate_nonlocal_centerline_separation(
    index: SegmentIndex,
    *,
    sample_spacing_m: float,
    required_separation_m: float,
) -> None:
    pairs = index.candidate_pairs(required_separation_m)
    for first, second in pairs:
        first_index = int(first)
        second_index = int(second)
        if _segments_are_adjacent(
            first_index,
            second_index,
            index.segment_count,
        ):
            continue
        index_delta = abs(first_index - second_index)
        cyclic_delta = min(index_delta, index.segment_count - index_delta)
        arc_separation = max(0.0, (cyclic_delta - 1) * sample_spacing_m)
        if arc_separation <= required_separation_m:
            continue
        distance = _segment_distance(
            index.starts_m[first],
            index.ends_m[first],
            index.starts_m[second],
            index.ends_m[second],
        )
        if distance <= required_separation_m:
            raise TrackValidationError(
                "nonlocal centerline segments "
                f"{first} and {second} are only {distance:.6g} m apart; "
                f"required separation is greater than {required_separation_m:.6g} m."
            )


def _validate_boundaries_do_not_intersect(
    left: SegmentIndex,
    right: SegmentIndex,
) -> None:
    search_radius = float(np.max(left.lengths_m) / 2.0 + np.max(right.lengths_m) / 2.0)
    candidate_lists = left._tree.query_ball_tree(right._tree, search_radius)
    for left_index, right_indices in enumerate(candidate_lists):
        for right_index in right_indices:
            if _segments_intersect(
                left.starts_m[left_index],
                left.ends_m[left_index],
                right.starts_m[right_index],
                right.ends_m[right_index],
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
    if _segments_intersect(
        first_start,
        first_end,
        second_start,
        second_end,
    ):
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
    _, _, distance = _project_to_segment(point, start, end)
    return distance


def _project_to_segment(
    point: FloatArray,
    start: FloatArray,
    end: FloatArray,
) -> tuple[FloatArray, float, float]:
    direction = end - start
    squared_length = float(np.dot(direction, direction))
    fraction = float(np.dot(point - start, direction) / squared_length)
    fraction = min(1.0, max(0.0, fraction))
    projection = start + fraction * direction
    distance = float(np.linalg.norm(point - projection))
    return projection, fraction, distance


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


def _point_array(value: FloatArray, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite array with shape (2,).")
    return point
