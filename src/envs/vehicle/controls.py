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
    Store physical controls held constant during one agent action.

    Fields:
        * acceleration: The longitudinal acceleration command.
        * steering_angle: The front-wheel steering angle in radians.
    """

    acceleration: float
    steering_angle: float


def normalized_to_physical_controls(
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
