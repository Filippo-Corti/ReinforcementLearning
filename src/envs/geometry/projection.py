"""Closest-segment projection for closed two-dimensional polylines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..types import FloatArray, point_array
from .segments import project_to_segment


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
        self._directions = ends - points
        self._squared_lengths = np.einsum(
            "ij,ij->i", self._directions, self._directions
        )
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
        point = point_array(point)
        _, nearest_midpoint = self._tree.query(point, k=1)
        initial_index = int(nearest_midpoint)
        _, _, initial_distance = project_to_segment(
            point,
            self.starts[initial_index],
            self.ends[initial_index],
        )

        search_radius = initial_distance + self._max_half_length
        candidates = np.union1d(
            np.asarray(
                self._tree.query_ball_point(point, search_radius), dtype=np.int64
            ),
            np.asarray([initial_index], dtype=np.int64),
        )
        return self.project_sorted_candidates(point, candidates)

    def project_candidates(
        self,
        point: FloatArray,
        segment_indices: Iterable[int],
    ) -> SegmentProjection:
        """
        Return the closest projection among an explicit segment subset.
        """
        point = point_array(point)
        candidates = np.asarray(list(segment_indices), dtype=np.int64)
        if candidates.size == 0:
            raise ValueError("segment_indices must not be empty.")
        if candidates.min() < 0 or candidates.max() >= self.segment_count:
            raise ValueError("segment_indices must reference indexed segments.")
        return self.project_sorted_candidates(point, np.sort(candidates))

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

    def project_sorted_candidates(
        self,
        point: FloatArray,
        indices: NDArray[np.int64],
    ) -> SegmentProjection:
        """
        Return the nearest projection among pre-validated ascending candidates.

        This is the per-physics-substep hot path, so it trusts its arguments and
        projects every candidate in one vectorized pass. Ascending order lets
        `argmin` reproduce the lowest-index winner among equidistant segments.
        """
        starts = self.starts[indices]
        directions = self._directions[indices]
        offsets = point - starts
        fractions = np.clip(
            np.einsum("ij,ij->i", offsets, directions) / self._squared_lengths[indices],
            0.0,
            1.0,
        )
        deltas = offsets - fractions[:, np.newaxis] * directions
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        best = int(np.argmin(squared_distances))
        return SegmentProjection(
            segment_index=int(indices[best]),
            fraction=float(fractions[best]),
            point=starts[best] + fractions[best] * directions[best],
            distance=float(np.sqrt(squared_distances[best])),
        )
