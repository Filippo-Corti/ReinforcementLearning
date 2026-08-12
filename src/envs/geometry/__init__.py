"""Reusable two-dimensional geometry and polyline projection utilities."""

from .angles import wrap_angle
from .interpolation import ScalarPiecewisePolynomial
from .projection import PolylineProjector, SegmentProjection
from .segments import (
    project_to_segment,
    segment_distance,
    segments_intersect,
)

__all__ = [
    "PolylineProjector",
    "ScalarPiecewisePolynomial",
    "SegmentProjection",
    "project_to_segment",
    "segment_distance",
    "segments_intersect",
    "wrap_angle",
]
