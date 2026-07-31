"""Periodic interpolation and derived geometry for sampled racing tracks."""

from __future__ import annotations

from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .model import Track
from .projection import PolylineProjector

FloatArray = NDArray[np.float64]


class TrackGeometry:
    """
    Provide periodic centerline queries and sampled track boundaries.

    Fields:
        * track: The sampled track that defines the geometry.
        * left_boundary: The sampled boundary on the left of forward travel.
        * right_boundary: The sampled boundary on the right of forward travel.
        * centerline_projector: The spatial index over centerline segments.
        * left_boundary_projector: The spatial index over left-boundary segments.
        * right_boundary_projector: The spatial index over right-boundary segments.
    """

    def __init__(self, track: Track) -> None:
        self.track = track
        extended_s = np.append(track.s, track.track_length)
        extended_x = np.append(track.x, track.x[0])
        extended_y = np.append(track.y, track.y[0])
        extended_curvature = np.append(track.curvature, track.curvature[0])

        self._x_spline = CubicSpline(extended_s, extended_x, bc_type="periodic")
        self._y_spline = CubicSpline(extended_s, extended_y, bc_type="periodic")
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

        normals = np.column_stack((-np.sin(track.heading), np.cos(track.heading)))
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
        Interpolate centerline position periodically at an arc length.
        """
        wrapped = self._wrapped(s)
        return np.array(
            [self._x_spline(wrapped), self._y_spline(wrapped)],
            dtype=np.float64,
        )

    def heading(self, s: float) -> float:
        """
        Interpolate wrapped centerline heading at an arc length.
        """
        wrapped = self._wrapped(s)
        unwrapped = float(np.interp(wrapped, self._heading_s, self._heading_unwrapped))
        return wrap_angle(unwrapped)

    def normal(self, s: float) -> FloatArray:
        """
        Return the unit normal pointing left of the centerline tangent.
        """
        heading = self.heading(s)
        return np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)

    def curvature(self, s: float) -> float:
        """
        Interpolate local curvature periodically at an arc length.
        """
        return float(self._curvature_spline(self._wrapped(s)))

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
        Interpolate the left boundary at an arc length.
        """
        return self.position(s) + (self.track.width / 2.0) * self.normal(s)

    def right_boundary_position(self, s: float) -> FloatArray:
        """
        Interpolate the right boundary at an arc length.
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
