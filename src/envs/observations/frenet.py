"""Frenet observation construction for a vehicle on a sampled track."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import ClassVar

import numpy as np

from configs import CarConfig, FrenetObservationConfig, SimulationConfig

from ..geometry import SegmentProjection, wrap_angle
from ..tracks.track import TrackWithGeometry
from ..types import FloatArray, point_array
from ..vehicle.state import VehicleState


@dataclass(frozen=True, slots=True)
class FrenetObservation:
    """
    Store the environment observation expressed in track-relative Frenet values.

    The steering angle appears because it is rate limited and therefore part of
    the vehicle state: without it the agent could not tell how far its wheels
    already are from the angle it is about to request.

    Fields:
        * lateral_distance: The signed distance from the track centerline.
        * heading_error: The wrapped vehicle heading minus centerline heading.
        * speed: The vehicle's non-negative scalar speed.
        * steering_angle: The current front-wheel angle in radians.
        * curvature_preview: The average centerline curvature over the lookahead.
    """

    DIMENSIONS: ClassVar[int] = 5

    lateral_distance: float
    heading_error: float
    speed: float
    steering_angle: float
    curvature_preview: float

    def as_array(self) -> FloatArray:
        """
        Return values in the Gym observation order as a float64 array.
        """
        return np.asarray(
            [
                self.lateral_distance,
                self.heading_error,
                self.speed,
                self.steering_angle,
                self.curvature_preview,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class FrenetProjection:
    """
    Store the projection of one Cartesian point onto the track centerline.

    Fields:
        * s: The distance along the centerline to the projected point.
        * lateral_distance: The signed distance from the centerline.
        * segment_index: The centerline segment containing the projected point.
        * segment_fraction: The fraction along that segment.
        * projected_point: The Cartesian coordinates of the projected point.
        * used_global_search: Whether projection required a global search.
    """

    s: float
    lateral_distance: float
    segment_index: int
    segment_fraction: float
    projected_point: FloatArray = field(repr=False, compare=False)
    used_global_search: bool


class FrenetObserver:
    """
    Build Frenet observations and maintain spatially coherent track projections.

    Fields:
        * track: The sampled track and derived geometry used for observations.
        * max_speed: The maximum valid vehicle speed.
        * local_window: The centerline segments searched around a prior projection.
        * maximum_local_distance: The threshold that triggers global reprojection.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        simulation_config: SimulationConfig | None = None,
        vehicle_config: CarConfig | None = None,
    ) -> None:
        simulation = simulation_config or SimulationConfig()
        vehicle = vehicle_config or CarConfig()
        self.track = track
        self.max_speed = vehicle.max_speed
        maximum_physics_travel = vehicle.max_speed * simulation.physics_timestep
        spacing = track.track.sample_spacing
        self.local_window = ceil(maximum_physics_travel / spacing) + 4
        self.maximum_local_distance = (
            track.track.width / 2.0 + maximum_physics_travel + 4.0 * spacing
        )
        segment_count = track.centerline_projector.segment_count
        offsets = np.arange(-self.local_window, self.local_window + 1)
        self._local_candidates = np.sort(
            (np.arange(segment_count)[:, np.newaxis] + offsets) % segment_count,
            axis=1,
        )

    def observe(
        self,
        state: VehicleState,
        *,
        previous_segment_index: int | None = None,
        config: FrenetObservationConfig | None = None,
    ) -> tuple[FrenetObservation, FrenetProjection]:
        """
        Build an observation and return the centerline projection used for it.
        """
        self._validate_speed(state.speed)
        projection = self._project(
            state.position(),
            previous_segment_index=previous_segment_index,
        )
        observation = FrenetObservation(
            lateral_distance=projection.lateral_distance,
            heading_error=self.heading_error(state.heading, projection.s),
            speed=state.speed,
            steering_angle=state.steering_angle,
            curvature_preview=self.curvature_preview(
                projection.s,
                state.speed,
                config=config,
            ),
        )
        return observation, projection

    def xy_from_frenet(self, s: float, lateral_distance: float) -> FloatArray:
        """
        Convert Frenet coordinates to a Cartesian point.
        """
        if not isfinite(lateral_distance):
            raise ValueError("lateral_distance must be finite.")
        return self.track.position(s) + lateral_distance * self.track.normal(s)

    def heading_error(self, vehicle_heading: float, s: float) -> float:
        """
        Return wrapped vehicle heading minus centerline heading.
        """
        return wrap_angle(vehicle_heading - self.track.heading(s))

    def curvature_preview(
        self,
        s: float,
        speed: float,
        *,
        config: FrenetObservationConfig | None = None,
    ) -> float:
        """
        Return average curvature over the velocity-dependent lookahead.
        """
        self._validate_speed(speed)
        observation = config or FrenetObservationConfig()
        lookahead = (
            observation.lookahead_base + observation.lookahead_speed_factor * speed
        )
        return self.track.integrated_curvature(s, lookahead) / lookahead

    def _project(
        self,
        point: FloatArray,
        *,
        previous_segment_index: int | None = None,
    ) -> FrenetProjection:
        """
        Project a point locally when possible and globally when necessary.
        """
        point = point_array(point)
        projector = self.track.centerline_projector
        if previous_segment_index is None:
            projection = projector.project(point)
            return self._frenet_projection(
                point,
                projection,
                used_global_search=True,
            )
        if (
            type(previous_segment_index) is not int
            or not 0 <= previous_segment_index < projector.segment_count
        ):
            raise ValueError(
                "previous_segment_index must reference a centerline segment."
            )

        local = projector.project_sorted_candidates(
            point,
            self._local_candidates[previous_segment_index],
        )
        if local.distance <= self.maximum_local_distance:
            return self._frenet_projection(
                point,
                local,
                used_global_search=False,
            )

        global_projection = projector.project(point)
        return self._frenet_projection(
            point,
            global_projection,
            used_global_search=True,
        )

    def _frenet_projection(
        self,
        point: FloatArray,
        projection: SegmentProjection,
        *,
        used_global_search: bool,
    ) -> FrenetProjection:
        """
        Convert a polyline projection to periodic Frenet coordinates.
        """
        track = self.track.track
        s = (
            (projection.segment_index + projection.fraction)
            * track.sample_spacing
            % track.track_length
        )
        lateral_distance = float(np.dot(point - projection.point, self.track.normal(s)))
        return FrenetProjection(
            s=s,
            lateral_distance=lateral_distance,
            segment_index=projection.segment_index,
            segment_fraction=projection.fraction,
            projected_point=projection.point,
            used_global_search=used_global_search,
        )

    def _validate_speed(self, speed: float) -> None:
        """
        Validate speed against the configured vehicle range.
        """
        if not isfinite(speed) or not 0 <= speed <= self.max_speed:
            raise ValueError("speed must be finite and within the vehicle speed range.")


def signed_progress(
    previous_s: float,
    current_s: float,
    track_length: float,
) -> float:
    """
    Return signed periodic progress in the half-open principal interval.
    """
    if (
        not isfinite(previous_s)
        or not isfinite(current_s)
        or not isfinite(track_length)
        or track_length <= 0
    ):
        raise ValueError("progress positions must be finite and track_length positive.")
    difference = current_s - previous_s
    return float((difference + track_length / 2.0) % track_length - track_length / 2.0)
