"""Tests for the pure kinematic vehicle transition."""

from __future__ import annotations

from math import pi, radians, tan

import pytest

from configs import CarConfig
from envs import (
    NormalizedAction,
    VehicleState,
    normalized_to_physical_controls,
    transition,
)


def test_zero_speed_zero_action_remains_stationary() -> None:
    """
    A stationary car with no controls does not move.
    """
    state = VehicleState(x=3.0, y=-2.0, heading=0.4, speed=0.0)

    result = transition(state, NormalizedAction(throttle=0.0, steering=0.0))

    assert result.state == state
    assert result.substep_states == (state, state, state, state)


def test_straight_acceleration_uses_explicit_euler_substeps() -> None:
    """
    Acceleration updates speed after each position update.
    """
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.0),
        NormalizedAction(throttle=1.0, steering=0.0),
    )

    acceleration = CarConfig().max_acceleration
    drag_free_speed = 4.0 * 0.01 * acceleration
    assert result.state.speed < drag_free_speed
    assert result.state.speed == pytest.approx(drag_free_speed, rel=1e-4)
    assert result.state.x == pytest.approx(
        0.01 * acceleration * 0.01 * (1 + 2 + 3), rel=1e-4
    )
    assert result.state.y == pytest.approx(0.0)
    assert result.state.heading == pytest.approx(0.0)


def test_braking_clamps_speed_at_zero_after_each_substep() -> None:
    """
    Braking cannot reverse the point car.
    """
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.3),
        NormalizedAction(throttle=-1.0, steering=0.0),
    )

    assert [state.speed for state in result.substep_states] == pytest.approx(
        [0.1, 0.0, 0.0, 0.0], abs=1e-5
    )
    assert result.state.x == pytest.approx(0.01 * (0.3 + 0.1), abs=1e-6)


def test_drag_makes_max_speed_the_terminal_speed() -> None:
    """
    Full throttle at the configured maximum produces no further acceleration.
    """
    vehicle = CarConfig()

    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=vehicle.max_speed),
        NormalizedAction(throttle=1.0, steering=0.0),
        vehicle_config=vehicle,
    )

    assert result.state.speed == pytest.approx(vehicle.max_speed)


def test_steering_angle_moves_toward_the_request_at_the_rate_limit() -> None:
    """
    One agent step cannot swing the wheels from centre to full lock.
    """
    vehicle = CarConfig()

    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0, steering_angle=0.0),
        NormalizedAction(throttle=0.0, steering=1.0),
        vehicle_config=vehicle,
    )

    reached = radians(vehicle.max_steering_rate) * 0.04
    assert result.state.steering_angle == pytest.approx(reached)
    assert reached < radians(vehicle.max_steering_angle)


def test_steering_angle_stops_at_the_requested_value() -> None:
    """
    The wheels do not overshoot a request they can reach within one substep.
    """
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0, steering_angle=0.0),
        NormalizedAction(throttle=0.0, steering=0.01),
    )

    assert result.state.steering_angle == pytest.approx(radians(30.0) * 0.01)


def test_cornering_never_exceeds_the_friction_budget() -> None:
    """
    Full lock at speed understeers instead of pulling impossible lateral force.
    """
    vehicle = CarConfig()
    speed = 40.0
    state = VehicleState(
        x=0.0,
        y=0.0,
        heading=0.0,
        speed=speed,
        steering_angle=radians(vehicle.max_steering_angle),
    )

    result = transition(
        state,
        NormalizedAction(throttle=0.0, steering=1.0),
        vehicle_config=vehicle,
    )

    yaw_rate = result.substep_states[0].heading / 0.01
    assert speed * yaw_rate == pytest.approx(vehicle.max_lateral_acceleration)
    assert speed**2 * tan(state.steering_angle) / vehicle.wheelbase > 10.0 * (
        vehicle.max_lateral_acceleration
    )


def test_braking_at_the_limit_leaves_no_grip_for_cornering() -> None:
    """
    Longitudinal and lateral tyre demand share one friction budget.
    """
    vehicle = CarConfig()
    state = VehicleState(
        x=0.0,
        y=0.0,
        heading=0.0,
        speed=30.0,
        steering_angle=radians(vehicle.max_steering_angle),
    )

    braking = transition(
        state,
        NormalizedAction(throttle=-1.0, steering=1.0),
        vehicle_config=vehicle,
    )
    coasting = transition(
        state,
        NormalizedAction(throttle=0.0, steering=1.0),
        vehicle_config=vehicle,
    )

    assert braking.state.heading == pytest.approx(0.0)
    assert coasting.state.heading > 0.0


@pytest.mark.parametrize(
    ("steering", "expected_sign"),
    [(1.0, 1), (-1.0, -1)],
)
def test_steering_turns_in_the_documented_direction(
    steering: float,
    expected_sign: int,
) -> None:
    """
    Positive steering turns left and negative steering turns right.
    """
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0),
        NormalizedAction(throttle=0.0, steering=steering),
    )

    assert expected_sign * result.state.heading > 0


def test_physical_control_mapping_uses_documented_limits() -> None:
    """
    Normalized extrema map to the configured acceleration and steering limits.
    """
    braking = normalized_to_physical_controls(
        NormalizedAction(throttle=-1.0, steering=1.0)
    )
    accelerating = normalized_to_physical_controls(
        NormalizedAction(throttle=1.0, steering=1.0)
    )

    assert braking.acceleration == pytest.approx(-CarConfig().max_braking)
    assert accelerating.acceleration == pytest.approx(CarConfig().max_acceleration)
    assert braking.steering_angle == pytest.approx(pi / 6.0)


def test_speed_never_leaves_configured_bounds() -> None:
    """
    Repeated full throttle clamps speed at the configured maximum.
    """
    vehicle = CarConfig(max_speed=0.1)
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.1),
        NormalizedAction(throttle=1.0, steering=0.0),
        vehicle_config=vehicle,
    )

    assert all(
        0.0 <= state.speed <= vehicle.max_speed for state in result.substep_states
    )
    assert result.state.speed == vehicle.max_speed


def test_transition_returns_one_pose_per_physics_substep() -> None:
    """
    The transition exposes all four states needed by future lifecycle checks.
    """
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=1.0),
        NormalizedAction(throttle=0.0, steering=0.0),
    )

    assert len(result.substep_states) == 4
    assert result.substep_states[-1] == result.state


def test_vehicle_state_exposes_float64_position() -> None:
    """
    Position conversion is owned by the state rather than lifecycle callers.
    """
    state = VehicleState(x=3.0, y=-2.0, heading=0.0, speed=0.0)

    position = state.position()

    assert position.dtype.name == "float64"
    assert position.tolist() == [3.0, -2.0]
