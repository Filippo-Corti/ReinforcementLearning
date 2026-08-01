"""Operations on finite two-dimensional line segments."""

from __future__ import annotations

import numpy as np

from ..types import FloatArray


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


def segments_intersect(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
) -> bool:
    """
    Return whether two closed line segments intersect within numeric tolerance.
    """
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


def segment_distance(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
) -> float:
    """
    Return the minimum Euclidean distance between two closed line segments.
    """
    if segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        project_to_segment(first_start, second_start, second_end)[2],
        project_to_segment(first_end, second_start, second_end)[2],
        project_to_segment(second_start, first_start, first_end)[2],
        project_to_segment(second_end, first_start, first_end)[2],
    )


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
