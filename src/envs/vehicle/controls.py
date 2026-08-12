"""Normalized agent actions and physical vehicle controls."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, radians

from configs import CarConfig


@dataclass(frozen=True, slots=True)
class NormalizedAction:
    """
    Store normalized throttle/brake and steering for one agent action.

    Fields:
        * throttle: The command from braking (-1) to throttle (+1).
        * steering: The command from right (-1) to left (+1).
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
    Store the controls an agent action requests for one agent step.

    These are demands, not achieved values. The transition kernel still applies
    the steering rate limit, aerodynamic drag and the tyre friction budget, so
    the car may reach neither the requested acceleration nor the requested
    steering angle.

    Fields:
        * acceleration: The requested longitudinal tyre acceleration.
        * steering_angle: The requested front-wheel steering angle in radians.
    """

    acceleration: float
    steering_angle: float


def normalized_to_physical_controls(
    action: NormalizedAction,
    *,
    vehicle_config: CarConfig | None = None,
) -> PhysicalControls:
    """
    Convert a normalized action into the requested physical controls.

    Braking uses its own larger limit because a car sheds speed through its
    brakes far harder than its engine can add it.
    """
    vehicle = vehicle_config or CarConfig()
    longitudinal_limit = (
        vehicle.max_acceleration if action.throttle >= 0.0 else vehicle.max_braking
    )
    return PhysicalControls(
        acceleration=longitudinal_limit * action.throttle,
        steering_angle=radians(vehicle.max_steering_angle) * action.steering,
    )
