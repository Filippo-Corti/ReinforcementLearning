"""Tests for LiDAR ray casting against the track boundaries."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from configs import CarConfig, LidarObservationConfig
from envs import (
    LidarObserver,
    Track,
    TrackGenerationMetadata,
    TrackWithGeometry,
    VehicleState,
)

WIDTH = 12.0


@pytest.fixture(scope="module")
def circle_track() -> TrackWithGeometry:
    """
    A circle of known radius, whose boundaries are two concentric circles.
    """
    sample_count = 3_200
    spacing = 0.5
    track_length = sample_count * spacing
    radius = track_length / (2.0 * pi)
    angles = np.arange(sample_count, dtype=np.float64) * 2.0 * pi / sample_count
    return TrackWithGeometry(
        Track(
            generation=TrackGenerationMetadata(
                seed=0,
                n_corners=12,
                base_radius=radius,
                radial_jitter=0.0,
                angular_jitter=0.0,
                max_attempts=1,
            ),
            width=WIDTH,
            sample_spacing=spacing,
            track_length=track_length,
            start_index=0,
            s=np.arange(sample_count, dtype=np.float64) * spacing,
            x=radius * np.cos(angles),
            y=radius * np.sin(angles),
            heading=np.asarray(angles + pi / 2.0, dtype=np.float64),
            curvature=np.full(sample_count, 1.0 / radius, dtype=np.float64),
        )
    )


def _state_on_centerline(
    track: TrackWithGeometry, *, lateral_offset: float = 0.0
) -> VehicleState:
    position = track.position(0.0) + lateral_offset * track.normal(0.0)
    return VehicleState(
        x=float(position[0]),
        y=float(position[1]),
        heading=track.heading(0.0),
        speed=0.0,
    )


def _perpendicular_observer(track: TrackWithGeometry) -> LidarObserver:
    """
    A three-ray fan whose outer rays point exactly across the track.
    """
    return LidarObserver(
        track,
        observation_config=LidarObservationConfig(ray_count=3, field_of_view=180.0),
    )


def test_rays_across_the_track_measure_the_half_width(circle_track) -> None:
    observer = _perpendicular_observer(circle_track)

    ranges = observer.ranges(_state_on_centerline(circle_track))

    assert ranges[0] == pytest.approx(WIDTH / 2, abs=0.01)
    assert ranges[2] == pytest.approx(WIDTH / 2, abs=0.01)


def test_a_ray_reports_the_nearer_wall_when_the_car_is_off_centre(circle_track) -> None:
    """
    Moving toward one wall must move both sideways readings, not just one.
    """
    observer = _perpendicular_observer(circle_track)
    offset = 2.0

    ranges = observer.ranges(_state_on_centerline(circle_track, lateral_offset=offset))

    # A positive lateral offset moves along the normal, which points left.
    assert ranges[2] == pytest.approx(WIDTH / 2 - offset, abs=0.05)
    assert ranges[0] == pytest.approx(WIDTH / 2 + offset, abs=0.05)


def test_a_ray_down_an_open_lap_stops_at_the_configured_range(circle_track) -> None:
    """
    Nothing within range must read exactly the range, not an unbounded value.

    A circle curves away from a forward ray, so the ray ahead runs a long way
    before meeting the outer wall. Shortening the range must truncate it.
    """
    short = LidarObserver(
        circle_track,
        observation_config=LidarObservationConfig(
            ray_count=3, field_of_view=180.0, max_range=5.0
        ),
    )

    ranges = short.ranges(_state_on_centerline(circle_track))

    assert ranges[1] == pytest.approx(5.0)
    # The sideways walls are further than five metres away, so they vanish too.
    assert np.all(ranges == pytest.approx(5.0))


def test_the_fan_spans_the_field_of_view_inclusive_of_both_extremes(
    circle_track,
) -> None:
    """
    Sixteen rays over two hundred degrees are separated by 200/15, not 200/16.
    """
    angles = np.degrees(LidarObserver(circle_track).ray_angles)

    assert angles.size == 16
    assert angles[0] == pytest.approx(-100.0)
    assert angles[-1] == pytest.approx(100.0)
    np.testing.assert_allclose(np.diff(angles), 200.0 / 15.0)


def test_the_observation_carries_vehicle_state_then_normalized_ranges(
    circle_track,
) -> None:
    observer = LidarObserver(circle_track)
    state = _state_on_centerline(circle_track)

    observation = observer.observe(state).as_array()

    assert observation.shape == (observer.dimensions,)
    assert observation[0] == pytest.approx(state.speed)
    assert observation[1] == pytest.approx(state.steering_angle)
    np.testing.assert_allclose(
        observation[2:], observer.ranges(state) / observer.max_range
    )
    assert np.all(observation[2:] >= 0.0) and np.all(observation[2:] <= 1.0)


def test_a_far_boundary_is_visible_even_when_it_is_far_along_the_lap(
    circle_track,
) -> None:
    """
    Visibility is geometric, so an arc-length window must not decide it.

    Standing on the centerline and facing across the circle, the ray leaves
    through the near wall first. Removing the near wall from consideration is
    exactly the mistake an arc-length window would make, so check that the wall
    directly opposite is reachable when the near one is not in the way: a ray
    aimed along the tangent crosses the far side of the ring.
    """
    radius = circle_track.track.track_length / (2.0 * pi)
    observer = LidarObserver(
        circle_track,
        observation_config=LidarObservationConfig(
            ray_count=3, field_of_view=180.0, max_range=4.0 * radius
        ),
    )

    forward = observer.ranges(_state_on_centerline(circle_track))[1]

    # The forward ray meets the outer wall after crossing a chord of the ring,
    # far further than the half width and far along the lap from where it began.
    assert forward > WIDTH
    assert forward < 4.0 * radius


def test_a_lidar_observer_rejects_an_impossible_configuration(circle_track) -> None:
    with pytest.raises(ValueError, match="at least two rays"):
        LidarObserver(
            circle_track, observation_config=LidarObservationConfig(ray_count=1)
        )
    with pytest.raises(ValueError, match="maximum range must be positive"):
        LidarObserver(
            circle_track, observation_config=LidarObservationConfig(max_range=0.0)
        )
    with pytest.raises(ValueError, match="speed must be finite"):
        LidarObserver(circle_track).observe(
            VehicleState(x=0.0, y=0.0, heading=0.0, speed=CarConfig().max_speed + 1.0)
        )
