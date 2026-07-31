"""Deterministic kinematic bicycle-model transitions for the racing car."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, tan

from configs import CarConfig, SimulationConfig


@dataclass(frozen=True, slots=True)
class VehicleState:
    """
    Cartesian pose and scalar speed of the point-car model.

    Fields:
        * x: The car's Cartesian horizontal position.
        * y: The car's Cartesian vertical position.
        * heading: The car heading in radians.
        * speed: The non-negative scalar speed.
    """

    x: float
    y: float
    heading: float
    speed: float

    def __post_init__(self) -> None:
        """
        Validate the physical state values.
        """
        if not all(isfinite(value) for value in (self.x, self.y, self.heading)):
            raise ValueError("vehicle pose values must be finite.")
        if not isfinite(self.speed) or self.speed < 0:
            raise ValueError("vehicle speed must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class NormalizedAction:
    """
    Normalized throttle/brake and steering controls for one agent action.

    Fields:
        * throttle: The normalized acceleration command, from braking (-1) to throttle (+1).
        * steering: The normalized steering-angle command, from right (-1) to left (+1).
    """

    throttle: float
    steering: float

    def __post_init__(self) -> None:
        """
        Validate the normalized action bounds.
        """
        if not all(isfinite(value) and -1.0 <= value <= 1.0 for value in self):
            raise ValueError(
                "normalized action values must be finite and within [-1, 1]."
            )

    def __iter__(self):
        """
        Iterate over the controls in throttle, steering order.
        """
        yield self.throttle
        yield self.steering


@dataclass(frozen=True, slots=True)
class PhysicalControls:
    """
    Physical controls derived from one normalized action.

    Fields:
        * acceleration: The longitudinal acceleration command.
        * steering_angle: The front-wheel steering angle in radians.
    """

    acceleration: float
    steering_angle: float


@dataclass(frozen=True, slots=True)
class DynamicsTransition:
    """
    Complete result of applying one agent action to the vehicle model.

    Fields:
        * state: The final state after all physics substeps.
        * substep_states: States after each physics substep, in temporal order.
        * controls: The constant physical controls held over the substeps.
    """

    state: VehicleState
    substep_states: tuple[VehicleState, ...]
    controls: PhysicalControls


def map_action(
    action: NormalizedAction,
    *,
    vehicle_config: CarConfig | None = None,
) -> PhysicalControls:
    """
    Convert a normalized action into the documented physical controls.
    """
    vehicle = vehicle_config or CarConfig()
    return PhysicalControls(
        acceleration=vehicle.max_acceleration * action.throttle,
        steering_angle=radians(vehicle.max_steering_angle) * action.steering,
    )


def transition(
    state: VehicleState,
    action: NormalizedAction,
    *,
    simulation_config: SimulationConfig | None = None,
    vehicle_config: CarConfig | None = None,
) -> DynamicsTransition:
    """
    Apply one action as explicit-Euler physics substeps and return every pose.
    """
    simulation = simulation_config or SimulationConfig()
    vehicle = vehicle_config or CarConfig()
    if state.speed > vehicle.max_speed:
        raise ValueError("vehicle state speed must not exceed the configured maximum.")

    controls = map_action(action, vehicle_config=vehicle)
    current = state
    substep_states: list[VehicleState] = []
    for _ in range(simulation.physics_substeps):
        next_speed = min(
            vehicle.max_speed,
            max(
                0.0, current.speed + simulation.physics_timestep * controls.acceleration
            ),
        )
        next_state = VehicleState(
            x=current.x
            + simulation.physics_timestep * current.speed * cos(current.heading),
            y=current.y
            + simulation.physics_timestep * current.speed * sin(current.heading),
            heading=current.heading
            + simulation.physics_timestep
            * current.speed
            / vehicle.wheelbase
            * tan(controls.steering_angle),
            speed=next_speed,
        )
        substep_states.append(next_state)
        current = next_state

    return DynamicsTransition(
        state=current,
        substep_states=tuple(substep_states),
        controls=controls,
    )
