"""Cartesian-to-Frenet projection and observation geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite

import numpy as np
from numpy.typing import NDArray

from configs import (
    FrenetObservationConfig,
    SimulationConfig,
    VehicleConfig,
)

from .geometry import SegmentProjection, TrackGeometry, wrap_angle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FrenetProjection:
    """Projection of one Cartesian point onto the sampled centerline."""

    s_m: float
    lateral_distance_m: float
    segment_index: int
    segment_fraction: float
    projected_point_m: FloatArray = field(repr=False, compare=False)
    used_global_search: bool


class FrenetProjector:
    """State-aware centerline projector with a safe global fallback."""

    def __init__(
        self,
        geometry: TrackGeometry,
        *,
        simulation_config: SimulationConfig | None = None,
        vehicle_config: VehicleConfig | None = None,
    ) -> None:
        simulation = simulation_config or SimulationConfig()
        vehicle = vehicle_config or VehicleConfig()
        self.geometry = geometry
        self.maximum_speed_m_per_s = vehicle.max_speed_m_per_s
        maximum_physics_travel = (
            vehicle.max_speed_m_per_s * simulation.physics_timestep_s
        )
        spacing = geometry.track.sample_spacing_m
        self.local_window_segments = ceil(maximum_physics_travel / spacing) + 4
        self.maximum_local_distance_m = (
            geometry.track.width_m / 2.0 + maximum_physics_travel + 4.0 * spacing
        )

    def project(
        self,
        point_m: FloatArray,
        *,
        previous_segment_index: int | None = None,
    ) -> FrenetProjection:
        """Project a point locally when safe, otherwise search globally."""
        point = _point_array(point_m, "point_m")
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
        if local.distance_m <= self.maximum_local_distance_m:
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
        speed_m_per_s: float,
        *,
        config: FrenetObservationConfig | None = None,
    ) -> float:
        """Return average curvature over the velocity-dependent lookahead."""
        self._validate_speed(speed_m_per_s)
        observation = config or FrenetObservationConfig()
        lookahead = (
            observation.lookahead_base_m
            + observation.lookahead_speed_factor_s * speed_m_per_s
        )
        return self.geometry.integrated_curvature(s_m, lookahead) / lookahead

    def observation(
        self,
        point_m: FloatArray,
        *,
        vehicle_heading_rad: float,
        speed_m_per_s: float,
        previous_segment_index: int | None = None,
        config: FrenetObservationConfig | None = None,
    ) -> tuple[FloatArray, FrenetProjection]:
        """Build ``(d, heading error, speed, curvature preview)``."""
        self._validate_speed(speed_m_per_s)
        projection = self.project(
            point_m,
            previous_segment_index=previous_segment_index,
        )
        values = np.asarray(
            [
                projection.lateral_distance_m,
                self.heading_error(vehicle_heading_rad, projection.s_m),
                speed_m_per_s,
                self.curvature_preview(
                    projection.s_m,
                    speed_m_per_s,
                    config=config,
                ),
            ],
            dtype=np.float64,
        )
        return values, projection

    def _frenet_projection(
        self,
        point_m: FloatArray,
        projection: SegmentProjection,
        *,
        used_global_search: bool,
    ) -> FrenetProjection:
        track = self.geometry.track
        s_m = (
            (projection.segment_index + projection.fraction)
            * track.sample_spacing_m
            % track.track_length_m
        )
        displacement = projection.point_m
        lateral_distance = float(
            np.dot(
                point_m - displacement,
                self.geometry.normal(s_m),
            )
        )
        return FrenetProjection(
            s_m=s_m,
            lateral_distance_m=lateral_distance,
            segment_index=projection.segment_index,
            segment_fraction=projection.fraction,
            projected_point_m=projection.point_m,
            used_global_search=used_global_search,
        )

    def _validate_speed(self, speed_m_per_s: float) -> None:
        if (
            not isfinite(speed_m_per_s)
            or not 0 <= speed_m_per_s <= self.maximum_speed_m_per_s
        ):
            raise ValueError(
                "speed_m_per_s must be finite and within the vehicle speed range."
            )


def signed_progress(
    previous_s_m: float,
    current_s_m: float,
    track_length_m: float,
) -> float:
    """Return signed periodic progress in the half-open principal interval."""
    if (
        not isfinite(previous_s_m)
        or not isfinite(current_s_m)
        or not isfinite(track_length_m)
        or track_length_m <= 0
    ):
        raise ValueError(
            "progress positions must be finite and track_length_m positive."
        )
    difference = current_s_m - previous_s_m
    return float(
        (difference + track_length_m / 2.0) % track_length_m - track_length_m / 2.0
    )


def _point_array(value: FloatArray, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite array with shape (2,).")
    return point
