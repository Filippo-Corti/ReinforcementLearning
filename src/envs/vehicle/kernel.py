"""Deterministic kinematic bicycle-model transition kernel."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, copysign, cos, radians, sin, sqrt, tan

from configs import CarConfig, SimulationConfig

from .controls import (
    NormalizedAction,
    PhysicalControls,
    normalized_to_physical_controls,
)
from .state import VehicleState


@dataclass(frozen=True, slots=True)
class KinematicTransition:
    """
    Store the complete vehicle result of applying one agent action.

    Fields:
        * state: The final state after all physics substeps.
        * substep_states: The states after each physics substep in temporal order.
        * controls: The physical controls held constant over the substeps.
    """

    state: VehicleState
    substep_states: tuple[VehicleState, ...]
    controls: PhysicalControls


def transition(
    state: VehicleState,
    action: NormalizedAction,
    *,
    simulation_config: SimulationConfig | None = None,
    vehicle_config: CarConfig | None = None,
) -> KinematicTransition:
    """
    Apply one action through explicit-Euler physics substeps.

    Each substep moves the front wheels toward the requested angle at the
    steering rate limit, spends what longitudinal grip the action demands and
    corners on whatever friction budget remains, then integrates aerodynamic
    drag alongside the requested acceleration.

    Args:
        state: The current vehicle state.
        action: The normalized action held over every physics substep.
        simulation_config: Simulation timing and substep configuration.
        vehicle_config: Vehicle geometry and physical control limits.

    Returns:
        The final state, intermediate substep states, and applied controls.
    """
    simulation = simulation_config or SimulationConfig()
    vehicle = vehicle_config or CarConfig()
    if state.speed > vehicle.max_speed:
        raise ValueError("vehicle state speed must not exceed the configured maximum.")

    controls = normalized_to_physical_controls(action, vehicle_config=vehicle)
    timestep = simulation.physics_timestep
    steering_step = radians(vehicle.max_steering_rate) * timestep
    drag_coefficient = vehicle.drag_coefficient

    current = state
    substep_states: list[VehicleState] = []
    for _ in range(simulation.physics_substeps):
        steering_angle = current.steering_angle + copysign(
            min(abs(controls.steering_angle - current.steering_angle), steering_step),
            controls.steering_angle - current.steering_angle,
        )
        yaw_rate = (
            current.speed
            / vehicle.wheelbase
            * tan(
                _grip_limited_steering(
                    steering_angle,
                    current.speed,
                    controls.acceleration,
                    vehicle=vehicle,
                )
            )
        )
        acceleration = (
            controls.acceleration - drag_coefficient * current.speed * current.speed
        )
        next_state = VehicleState(
            x=current.x + timestep * current.speed * cos(current.heading),
            y=current.y + timestep * current.speed * sin(current.heading),
            heading=current.heading + timestep * yaw_rate,
            speed=min(
                vehicle.max_speed,
                max(0.0, current.speed + timestep * acceleration),
            ),
            steering_angle=steering_angle,
        )
        substep_states.append(next_state)
        current = next_state

    return KinematicTransition(
        state=current,
        substep_states=tuple(substep_states),
        controls=controls,
    )


def _grip_limited_steering(
    steering_angle: float,
    speed: float,
    longitudinal_acceleration: float,
    *,
    vehicle: CarConfig,
) -> float:
    """
    Return the steering angle the tyres can actually deliver at this speed.

    Longitudinal and lateral tyre demand share one friction budget, so what is
    spent accelerating or braking is unavailable for cornering. Asking for more
    lateral acceleration than the remaining budget makes the car understeer: the
    wheels stay where the driver put them but the car turns less than requested
    and runs wide.
    """
    budget = vehicle.max_lateral_acceleration
    longitudinal = min(abs(longitudinal_acceleration), budget)
    remaining = budget * budget - longitudinal * longitudinal
    if remaining <= 0.0:
        return 0.0
    lateral_demand = speed * speed * tan(steering_angle) / vehicle.wheelbase
    available = sqrt(remaining)
    if abs(lateral_demand) <= available:
        return steering_angle
    if speed <= 0.0:
        return steering_angle
    return copysign(
        atan(available * vehicle.wheelbase / (speed * speed)), steering_angle
    )
