"""Periodic track interpolation, segment search, and geometric validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite, pi, radians, tan

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from configs import CarConfig, TrackGenerationConfig

from .track import Track, TrackValidationError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SegmentProjection:
    """
    Result of projecting a point (x, y) onto a segment of a closed polyline.
    The closed polyline can either be the centerline or one of the boundaries of a track.

    Fields:
        * segment_index: The index of the relevant segment in the closed polyline.
        * fraction: The fraction along the segment where the projection occurs, in [0, 1].
        * point: The projected point in Cartesian coordinates, as a 2D array.
        * distance: The Euclidean distance from the original point to the projected point.
    """

    segment_index: int
    fraction: float
    point: FloatArray = field(repr=False, compare=False)
    distance: float


class PolylineProjector:
    """
    Projection processor for a given closed polyline (a sequence of 2D points with the last point implicitly connected to the first).
    It provides efficient functionalities for:
    - Projecting a point onto the nearest segment of the closed polyline.
    - Projecting a point onto a specified subset of segments.
    - Finding pairs of segments that may be within a specified distance of each other.

    Fields:
        * starts: The start points of the segments in the closed polyline, as a 2D array.
        * ends: The end points of the segments in the closed polyline, as a 2D array.
        * lengths: The lengths of the segments in the closed polyline, as a 1D array.
        * midpoints: The midpoints of the segments in the closed polyline, as a 2D array.
        * _max_half_length: The maximum half-length of the segments, used for search radius calculations.
        * _tree: A cKDTree built from the midpoints of the segments, used for efficient nearest-neighbor searches.
    """

    def __init__(self, points: FloatArray) -> None:
        points = np.array(points, dtype=np.float64, copy=True)
        if points.shape[0] < 3:
            raise ValueError("points must contain at least 3 points.")

        ends = np.roll(points, shift=-1, axis=0)
        lengths = np.linalg.norm(ends - points, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("closed polylines cannot contain zero-length segments.")

        midpoints = (points + ends) / 2.0

        self.starts = points
        self.ends = ends
        self.lengths = lengths
        self.midpoints = midpoints
        self._max_half_length = float(np.max(lengths) / 2.0)
        self._tree = cKDTree(midpoints)

    @property
    def segment_count(self) -> int:
        """
        Return the number of segments in the closed polyline.
        """
        return self.starts.shape[0]

    def project(self, point: FloatArray) -> SegmentProjection:
        """
        Return the exact closest projection of a point onto the closed polyline.
        """
        point = _point_array(point, "point")
        _, nearest_midpoint = self._tree.query(point, k=1)
        initial_index = int(nearest_midpoint)
        _, _, initial_distance = _project_to_segment(
            point,
            self.starts[initial_index],
            self.ends[initial_index],
        )

        search_radius = initial_distance + self._max_half_length
        candidates = sorted(self._tree.query_ball_point(point, search_radius))
        return self._project_candidates(point, candidates)

    def project_candidates(
        self,
        point: FloatArray,
        segment_indices: Iterable[int],
    ) -> SegmentProjection:
        """
        Return the closest projection among an explicit segment subset.
        It may be useful to restrict the search to a local window of segments,
        when the point is known to be near a certain portion of the closed polyline.
        """
        point = _point_array(point, "point")
        candidates = list(segment_indices)
        if not candidates:
            raise ValueError("segment_indices must not be empty.")
        if any(type(index) is not int for index in candidates):
            raise ValueError("segment_indices must contain only integers.")
        if any(not 0 <= index < self.segment_count for index in candidates):
            raise ValueError("segment_indices must reference indexed segments.")
        return self._project_candidates(point, candidates)

    def _project_candidates(
        self,
        point: FloatArray,
        candidates: Iterable[int],
    ) -> SegmentProjection:
        best: SegmentProjection | None = None
        for index in candidates:
            projection, fraction, distance = _project_to_segment(
                point,
                self.starts[index],
                self.ends[index],
            )
            candidate = SegmentProjection(
                segment_index=index,
                fraction=fraction,
                point=projection,
                distance=distance,
            )
            if best is None or (distance, index) < (
                best.distance,
                best.segment_index,
            ):
                best = candidate

        if best is None:
            raise RuntimeError("segment index did not return any candidates.")
        return best

    def candidate_pairs(self, maximum_distance: float) -> NDArray[np.int64]:
        """
        Return segment-index pairs that may be within a target distance.
        """
        search_radius = maximum_distance + 2.0 * self._max_half_length
        pairs = self._tree.query_pairs(search_radius, output_type="ndarray")
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64)


class TrackGeometry:
    """
    Geometry utilities for a given Track object.
    Provides periodic interpolators for centerline position, heading, and curvature,
    as well as left and right boundary positions, normals, and integrated curvature.
    All of these help with geometric validation and coordinate transformations.

    Fields:
        * track: The Track object for which this geometry is defined.
        * left_boundary: The sampled left boundary of the track, as a 2D array.
        * right_boundary: The sampled right boundary of the track, as a 2D array.
        * _x_spline: A periodic cubic spline interpolator for the x-coordinate of the centerline.
        * _y_spline: A periodic cubic spline interpolator for the y-coordinate of the centerline.
        * _curvature_spline: A periodic cubic spline interpolator for the curvature of the centerline.
        * _curvature_integral: The antiderivative of the curvature spline, used for integrated curvature calculations.
        * _heading_s: The arc lengths at which the centerline heading is sampled, used for interpolation.
        * _heading_unwrapped: The unwrapped centerline heading values at the sampled arc lengths, used for interpolation.
        * centerline_projector: A PolylineProjector object for the centerline, used for nearest-segment searches.
        * left_boundary_projector: A PolylineProjector object for the left boundary, used for nearest-segment searches.
        * right_boundary_projector: A PolylineProjector object for the right boundary, used for nearest-segment searches.
    """

    def __init__(self, track: Track) -> None:
        self.track = track
        extended_s = np.append(track.s, track.track_length)
        extended_x = np.append(track.x, track.x[0])
        extended_y = np.append(track.y, track.y[0])
        extended_curvature = np.append(
            track.curvature,
            track.curvature[0],
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
        self._curvature_integral = self._curvature_spline.antiderivative()

        unwrapped_heading = np.unwrap(track.heading)
        closing_turn = wrap_angle(float(track.heading[0] - track.heading[-1]))
        self._heading_s = extended_s
        self._heading_unwrapped = np.append(
            unwrapped_heading,
            unwrapped_heading[-1] + closing_turn,
        )

        normals = np.column_stack(
            (
                -np.sin(track.heading),
                np.cos(track.heading),
            )
        )
        centerline = np.column_stack((track.x, track.y))
        self.left_boundary = centerline + (track.width / 2.0) * normals
        self.right_boundary = centerline - (track.width / 2.0) * normals
        self.left_boundary.setflags(write=False)
        self.right_boundary.setflags(write=False)

        self.centerline_projector = PolylineProjector(centerline)
        self.left_boundary_projector = PolylineProjector(self.left_boundary)
        self.right_boundary_projector = PolylineProjector(self.right_boundary)

    def position(self, s: float) -> FloatArray:
        """
        Interpolate centerline position periodically at arc length ``s``.
        """
        wrapped = self._wrapped(s)
        return np.array(
            [self._x_spline(wrapped), self._y_spline(wrapped)],
            dtype=np.float64,
        )

    def heading(self, s: float) -> float:
        """
        Interpolate wrapped centerline heading at arc length ``s``.
        """
        wrapped = self._wrapped(s)
        unwrapped = float(
            np.interp(
                wrapped,
                self._heading_s,
                self._heading_unwrapped,
            )
        )
        return wrap_angle(unwrapped)

    def normal(self, s: float) -> FloatArray:
        """
        Return the unit normal pointing left of the centerline tangent.
        """
        heading = self.heading(s)
        return np.array(
            [-np.sin(heading), np.cos(heading)],
            dtype=np.float64,
        )

    def curvature(self, s: float) -> float:
        """
        Interpolate local curvature periodically at arc length ``s``.
        """
        wrapped = self._wrapped(s)
        return float(self._curvature_spline(wrapped))

    def integrated_curvature(self, start_s: float, distance: float) -> float:
        """
        Integrate periodic curvature forward over a non-negative distance.
        """
        start = self._wrapped(start_s)
        if not isfinite(distance) or distance < 0:
            raise ValueError("distance must be finite and non-negative.")
        length = self.track.track_length
        complete_laps, remainder = divmod(distance, length)
        lap_integral = float(
            self._curvature_integral(length) - self._curvature_integral(0.0)
        )
        total = complete_laps * lap_integral
        end = start + remainder
        if end <= length:
            total += float(
                self._curvature_integral(end) - self._curvature_integral(start)
            )
        else:
            total += float(
                self._curvature_integral(length)
                - self._curvature_integral(start)
                + self._curvature_integral(end - length)
            )
        return total

    def left_boundary_position(self, s: float) -> FloatArray:
        """
        Interpolate the left boundary at arc length ``s``.
        """
        return self.position(s) + (self.track.width / 2.0) * self.normal(s)

    def right_boundary_position(self, s: float) -> FloatArray:
        """
        Interpolate the right boundary at arc length ``s``.
        """
        return self.position(s) - (self.track.width / 2.0) * self.normal(s)

    def _wrapped(self, s: float) -> float:
        return float(s % self.track.track_length)


def wrap_angle(angle: float) -> float:
    """
    Wrap an angle to the half-open interval ``[-pi, pi)``.
    """
    if not isfinite(angle):
        raise ValueError("angle must be finite.")
    return float((angle + pi) % (2.0 * pi) - pi)


def validate_track_geometry(
    track: Track,
    *,
    vehicle_config: CarConfig | None = None,
    track_config: TrackGenerationConfig | None = None,
) -> TrackGeometry:
    """
    Validate geometric constraints and return prepared track geometry.

    This includes checking that:
    - The track length is within the configured generation range.
    - The track curvature does not exceed the vehicle's kinematic steering limit.
    - The centerline does not self-intersect.
    - The left and right boundaries do not self-intersect.
    - The left and right boundaries do not intersect each other.
    - Nonlocal centerline segments are separated by at least the required distance.
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
    _validate_simple_closed_polyline(
        geometry.centerline_projector,
        "centerline",
    )

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
        if _segments_are_adjacent(
            int(first),
            int(second),
            index.segment_count,
        ):
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
        if _segments_are_adjacent(
            first_index,
            second_index,
            index.segment_count,
        ):
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
    search_radius = float(np.max(left.lengths) / 2.0 + np.max(right.lengths) / 2.0)
    candidate_lists = left._tree.query_ball_tree(right._tree, search_radius)
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
