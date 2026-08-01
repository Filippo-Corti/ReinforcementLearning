"""Deterministic kinematic bicycle-model transition kernel."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tan

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
    current = state
    substep_states: list[VehicleState] = []
    for _ in range(simulation.physics_substeps):
        next_speed = min(
            vehicle.max_speed,
            max(
                0.0,
                current.speed + simulation.physics_timestep * controls.acceleration,
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

    return KinematicTransition(
        state=current,
        substep_states=tuple(substep_states),
        controls=controls,
    )
