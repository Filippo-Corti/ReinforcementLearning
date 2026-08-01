"""Tests for the pure kinematic vehicle transition."""

from __future__ import annotations

from math import pi

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
    assert result.state.speed == pytest.approx(4.0 * 0.01 * acceleration)
    assert result.state.x == pytest.approx(0.01 * acceleration * 0.01 * (1 + 2 + 3))
    assert result.state.y == pytest.approx(0.0)
    assert result.state.heading == pytest.approx(0.0)


def test_braking_clamps_speed_at_zero_after_each_substep() -> None:
    """
    Braking cannot reverse the point car.
    """
    result = transition(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.1),
        NormalizedAction(throttle=-1.0, steering=0.0),
    )

    assert [state.speed for state in result.substep_states] == pytest.approx(
        [0.0074, 0.0, 0.0, 0.0]
    )
    assert result.state.x == pytest.approx(0.001074)


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
    controls = normalized_to_physical_controls(
        NormalizedAction(throttle=-1.0, steering=1.0)
    )

    assert controls.acceleration == pytest.approx(-CarConfig().max_acceleration)
    assert controls.steering_angle == pytest.approx(pi / 6.0)


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
