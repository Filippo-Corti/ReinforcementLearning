"""Vehicle state for the kinematic racing model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..types import FloatArray


@dataclass(frozen=True, slots=True)
class VehicleState:
    """
    Store the point-car Cartesian pose and non-negative scalar speed.

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

    def position(self) -> FloatArray:
        """
        Return the Cartesian position as a float64 vector.
        """
        return np.asarray([self.x, self.y], dtype=np.float64)
