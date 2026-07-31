"""Closest-segment projection for closed track polylines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SegmentProjection:
    """
    Result of projecting a point onto a segment of a closed polyline.

    Fields:
        * segment_index: The index of the relevant segment in the closed polyline.
        * fraction: The fraction along the segment where the projection occurs, in [0, 1].
        * point: The projected Cartesian point.
        * distance: The Euclidean distance from the query point to the projection.
    """

    segment_index: int
    fraction: float
    point: FloatArray = field(repr=False, compare=False)
    distance: float


class PolylineProjector:
    """
    Project points and find nearby segment pairs on a closed polyline.

    Fields:
        * starts: The start points of the indexed segments.
        * ends: The end points of the indexed segments.
        * lengths: The segment lengths.
        * midpoints: The segment midpoint coordinates.
        * _max_half_length: The maximum half-length used for search bounds.
        * _tree: The spatial index over segment midpoints.
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
        Return the exact closest projection onto the closed polyline.
        """
        point = point_array(point, "point")
        _, nearest_midpoint = self._tree.query(point, k=1)
        initial_index = int(nearest_midpoint)
        _, _, initial_distance = project_to_segment(
            point,
            self.starts[initial_index],
            self.ends[initial_index],
        )

        search_radius = initial_distance + self._max_half_length
        candidates = sorted(
            set(self._tree.query_ball_point(point, search_radius)) | {initial_index}
        )
        return self._project_candidates(point, candidates)

    def project_candidates(
        self,
        point: FloatArray,
        segment_indices: Iterable[int],
    ) -> SegmentProjection:
        """
        Return the closest projection among an explicit segment subset.
        """
        point = point_array(point, "point")
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
            projection, fraction, distance = project_to_segment(
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
            if best is None or (distance, index) < (best.distance, best.segment_index):
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

    def candidate_lists(
        self,
        other: PolylineProjector,
        maximum_distance: float,
    ) -> list[list[int]]:
        """
        Return candidate segments from another indexed polyline for each segment.
        """
        search_radius = (
            maximum_distance + self._max_half_length + other._max_half_length
        )
        return self._tree.query_ball_tree(other._tree, search_radius)


def project_to_segment(
    point: FloatArray,
    start: FloatArray,
    end: FloatArray,
) -> tuple[FloatArray, float, float]:
    """
    Project a point onto one finite line segment.
    """
    direction = end - start
    squared_length = float(np.dot(direction, direction))
    fraction = float(np.dot(point - start, direction) / squared_length)
    fraction = min(1.0, max(0.0, fraction))
    projection = start + fraction * direction
    distance = float(np.linalg.norm(point - projection))
    return projection, fraction, distance


def point_array(value: FloatArray, name: str) -> FloatArray:
    """
    Convert a value to a finite two-dimensional float64 point.
    """
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite array with shape (2,).")
    return point
