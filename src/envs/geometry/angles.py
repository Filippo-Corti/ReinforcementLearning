"""Angular geometry operations."""

from __future__ import annotations

from math import isfinite, pi


def wrap_angle(angle: float) -> float:
    """
    Wrap an angle to the half-open interval ``[-pi, pi)``.
    """
    if not isfinite(angle):
        raise ValueError("angle must be finite.")
    return float((angle + pi) % (2.0 * pi) - pi)
