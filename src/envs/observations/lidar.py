"""LiDAR observation construction by ray casting against the track boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, radians

import numpy as np

from configs import CarConfig, LidarObservationConfig

from ..tracks.track import TrackWithGeometry
from ..types import FloatArray
from ..vehicle.state import VehicleState


@dataclass(frozen=True, slots=True)
class LidarObservation:
    """
    Store the environment observation as vehicle state plus normalized ranges.

    The vehicle keeps its speed and steering angle because those are proprioception
    rather than perception: withholding them would compare two sensors while also
    changing what the car knows about itself.

    Fields:
        * speed: The vehicle's non-negative scalar speed.
        * steering_angle: The current front-wheel angle in radians.
        * normalized_ranges: Ray distances divided by the maximum range.
    """

    speed: float
    steering_angle: float
    normalized_ranges: FloatArray

    def as_array(self) -> FloatArray:
        """
        Return values in the Gym observation order as a float64 array.
        """
        return np.concatenate(
            (
                np.asarray([self.speed, self.steering_angle], dtype=np.float64),
                np.asarray(self.normalized_ranges, dtype=np.float64),
            )
        )


class LidarObserver:
    """
    Cast a fixed fan of rays from the vehicle pose to the nearest boundary.

    Rays are tested against *every* boundary segment rather than against a window
    around the vehicle's arc-length position. A ray can see a piece of track that
    is metres away in space but half a lap away along the centerline, and a
    window would report open road where there is a wall.

    Every ray is intersected with every segment in one vectorized pass. On a
    sampled circuit this is a few thousand candidate pairs, which is small enough
    that a spatial index would cost more bookkeeping than it saves.

    Fields:
        * track: The sampled track whose boundaries the rays hit.
        * max_range: The furthest distance a ray reports, in meters.
        * ray_angles: Ray offsets from the vehicle heading, in radians.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        observation_config: LidarObservationConfig | None = None,
        vehicle_config: CarConfig | None = None,
    ) -> None:
        observation = observation_config or LidarObservationConfig()
        vehicle = vehicle_config or CarConfig()
        if observation.ray_count < 2:
            raise ValueError("A LiDAR fan requires at least two rays.")
        if observation.max_range <= 0:
            raise ValueError("The LiDAR maximum range must be positive.")
        self.track = track
        self.max_speed = vehicle.max_speed
        self.max_range = observation.max_range
        half_field = radians(observation.field_of_view) / 2.0
        # Inclusive of both extremes, so the separation divides the field of view
        # by one less than the ray count.
        self.ray_angles = np.linspace(
            -half_field,
            half_field,
            observation.ray_count,
            dtype=np.float64,
        )

        starts = np.concatenate((track.left_boundary, track.right_boundary))
        # Each boundary is a closed polyline, so the last sample joins the first.
        ends = np.concatenate(
            (
                np.roll(track.left_boundary, -1, axis=0),
                np.roll(track.right_boundary, -1, axis=0),
            )
        )
        self._segment_starts = starts
        self._segment_edges = ends - starts
        self._segment_midpoints = starts + 0.5 * self._segment_edges
        self._segment_reach = 0.5 * np.linalg.norm(self._segment_edges, axis=1).max()

    @property
    def dimensions(self) -> int:
        """
        Return the length of the observation vector this observer produces.
        """
        return 2 + int(self.ray_angles.size)

    def observe(self, state: VehicleState) -> LidarObservation:
        """
        Build the observation for one vehicle pose.
        """
        if not isfinite(state.speed) or not 0 <= state.speed <= self.max_speed:
            raise ValueError("speed must be finite and within the vehicle speed range.")
        return LidarObservation(
            speed=state.speed,
            steering_angle=state.steering_angle,
            normalized_ranges=self.ranges(state) / self.max_range,
        )

    def ranges(self, state: VehicleState) -> FloatArray:
        """
        Return the distance from the vehicle to the first boundary hit by each ray.

        A ray that reaches nothing within its range reports the maximum range,
        which is indistinguishable from a hit exactly at that distance. Both mean
        the same thing to the policy: no boundary closer than this.
        """
        origin = state.position()
        directions = np.column_stack(
            (
                np.cos(state.heading + self.ray_angles),
                np.sin(state.heading + self.ray_angles),
            )
        )

        starts, edges = self._candidates(origin)
        if starts.shape[0] == 0:
            return np.full(self.ray_angles.shape, self.max_range, dtype=np.float64)

        offsets = starts - origin
        # Solving `origin + t d = start + u edge` for every (ray, segment) pair:
        # the ray parameter t is the hit distance, and the segment parameter u
        # says whether the crossing lies within the segment.
        denominator = (
            directions[:, 0, np.newaxis] * edges[np.newaxis, :, 1]
            - directions[:, 1, np.newaxis] * edges[np.newaxis, :, 0]
        )
        distance_numerator = offsets[:, 0] * edges[:, 1] - offsets[:, 1] * edges[:, 0]
        segment_numerator = (
            offsets[np.newaxis, :, 0] * directions[:, 1, np.newaxis]
            - offsets[np.newaxis, :, 1] * directions[:, 0, np.newaxis]
        )

        parallel = denominator == 0.0
        safe_denominator = np.where(parallel, 1.0, denominator)
        distance = distance_numerator[np.newaxis, :] / safe_denominator
        segment_fraction = segment_numerator / safe_denominator
        hit = (
            ~parallel
            & (distance >= 0.0)
            & (distance <= self.max_range)
            & (segment_fraction >= 0.0)
            & (segment_fraction <= 1.0)
        )
        return np.where(hit, distance, self.max_range).min(axis=1)

    def _candidates(self, origin: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Drop segments that cannot be reached, without assuming where they lie.

        The test is a distance from the ray origin, not an arc-length window, so
        a segment that is close in space is kept however far away it is along the
        lap. The half-segment margin keeps a segment whose midpoint is just out of
        range but whose near end is inside it.
        """
        reachable = (
            np.linalg.norm(self._segment_midpoints - origin, axis=1)
            <= self.max_range + self._segment_reach
        )
        return self._segment_starts[reachable], self._segment_edges[reachable]
