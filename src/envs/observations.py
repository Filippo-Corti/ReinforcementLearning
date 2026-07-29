"""Cartesian-to-Frenet projection and observation geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite

import numpy as np
from numpy.typing import NDArray

from configs import (
    FrenetObservationConfig,
    SimulationConfig,
    CarConfig,
)

from .geometry import SegmentProjection, TrackGeometry, wrap_angle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FrenetProjection:
    """
    Projection of one Cartesian point onto the sampled centerline.
    
    Fields:
        * s: The distance along the centerline to the projected point, in meters.
        * lateral_distance: The signed distance from the projected point to the original point, in meters
        * segment_index: The index of the centerline segment containing the projected point.
        * segment_fraction: The fraction along the segment to the projected point.
        * projected_point: The Cartesian coordinates of the projected point, in meters.
        * used_global_search: Whether the projection was found using a global search (True) or a local search (False).
    """

    s: float
    lateral_distance: float
    segment_index: int
    segment_fraction: float
    projected_point: FloatArray = field(repr=False, compare=False)
    used_global_search: bool


class FrenetProjector:
    """State-aware centerline projector with a safe global fallback."""

    def __init__(
        self,
        geometry: TrackGeometry,
        *,
        simulation_config: SimulationConfig | None = None,
        vehicle_config: CarConfig | None = None,
    ) -> None:
        simulation = simulation_config or SimulationConfig()
        vehicle = vehicle_config or CarConfig()
        self.geometry = geometry
        self.maximum_speed_m_per_s = vehicle.max_speed
        maximum_physics_travel = (
            vehicle.max_speed * simulation.physics_timestep
        )
        spacing = geometry.track.sample_spacing
        self.local_window_segments = ceil(maximum_physics_travel / spacing) + 4
        self.maximum_local_distance = (
            geometry.track.width / 2.0 + maximum_physics_travel + 4.0 * spacing
        )

    def project(
        self,
        point: FloatArray,
        *,
        previous_segment_index: int | None = None,
    ) -> FrenetProjection:
        """Project a point locally when safe, otherwise search globally."""
        point = _point_array(point, "point")
        if previous_segment_index is None:
            projection = self.geometry.centerline_index.project(point)
            return self._frenet_projection(
                point,
                projection,
                used_global_search=True,
            )
        if (
            type(previous_segment_index) is not int
            or not 0
            <= previous_segment_index
            < self.geometry.centerline_index.segment_count
        ):
            raise ValueError(
                "previous_segment_index must reference a centerline segment."
            )

        segment_count = self.geometry.centerline_index.segment_count
        candidates = {
            (previous_segment_index + offset) % segment_count
            for offset in range(
                -self.local_window_segments,
                self.local_window_segments + 1,
            )
        }
        local = self.geometry.centerline_index.project_candidates(
            point,
            sorted(candidates),
        )
        if local.distance <= self.maximum_local_distance:
            return self._frenet_projection(
                point,
                local,
                used_global_search=False,
            )

        global_projection = self.geometry.centerline_index.project(point)
        return self._frenet_projection(
            point,
            global_projection,
            used_global_search=True,
        )

    def xy_from_frenet(self, s_m: float, lateral_distance_m: float) -> FloatArray:
        """Convert Frenet coordinates to a Cartesian point."""
        if not isfinite(lateral_distance_m):
            raise ValueError("lateral_distance_m must be finite.")
        return self.geometry.position(s_m) + (
            lateral_distance_m * self.geometry.normal(s_m)
        )

    def heading_error(self, vehicle_heading_rad: float, s_m: float) -> float:
        """Return wrapped vehicle heading minus centerline heading."""
        return wrap_angle(vehicle_heading_rad - self.geometry.heading(s_m))

    def curvature_preview(
        self,
        s_m: float,
        speed: float,
        *,
        config: FrenetObservationConfig | None = None,
    ) -> float:
        """Return average curvature over the velocity-dependent lookahead."""
        self._validate_speed(speed)
        observation = config or FrenetObservationConfig()
        lookahead = (
            observation.lookahead_base
            + observation.lookahead_speed_factor * speed
        )
        return self.geometry.integrated_curvature(s_m, lookahead) / lookahead

    def observation(
        self,
        point: FloatArray,
        *,
        vehicle_heading_rad: float,
        speed: float,
        previous_segment_index: int | None = None,
        config: FrenetObservationConfig | None = None,
    ) -> tuple[FloatArray, FrenetProjection]:
        """Build ``(d, heading error, speed, curvature preview)``."""
        self._validate_speed(speed)
        projection = self.project(
            point,
            previous_segment_index=previous_segment_index,
        )
        values = np.asarray(
            [
                projection.lateral_distance,
                self.heading_error(vehicle_heading_rad, projection.s),
                speed,
                self.curvature_preview(
                    projection.s,
                    speed,
                    config=config,
                ),
            ],
            dtype=np.float64,
        )
        return values, projection

    def _frenet_projection(
        self,
        point: FloatArray,
        projection: SegmentProjection,
        *,
        used_global_search: bool,
    ) -> FrenetProjection:
        track = self.geometry.track
        s = (
            (projection.segment_index + projection.fraction)
            * track.sample_spacing
            % track.track_length
        )
        displacement = projection.point
        lateral_distance = float(
            np.dot(
                point - displacement,
                self.geometry.normal(s),
            )
        )
        return FrenetProjection(
            s=s,
            lateral_distance=lateral_distance,
            segment_index=projection.segment_index,
            segment_fraction=projection.fraction,
            projected_point=projection.point,
            used_global_search=used_global_search,
        )

    def _validate_speed(self, speed: float) -> None:
        if (
            not isfinite(speed)
            or not 0 <= speed <= self.maximum_speed_m_per_s
        ):
            raise ValueError(
                "speed must be finite and within the vehicle speed range."
            )


def signed_progress(
    previous_s: float,
    current_s: float,
    track_length: float,
) -> float:
    """Return signed periodic progress in the half-open principal interval."""
    if (
        not isfinite(previous_s)
        or not isfinite(current_s)
        or not isfinite(track_length)
        or track_length <= 0
    ):
        raise ValueError(
            "progress positions must be finite and track_length positive."
        )
    difference = current_s - previous_s
    return float(
        (difference + track_length / 2.0) % track_length - track_length / 2.0
    )


def _point_array(value: FloatArray, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite array with shape (2,).")
    return point
