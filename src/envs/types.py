"""Shared numerical types and conversions for environment components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def point_array(value: FloatArray, name: str = "point") -> FloatArray:
    """
    Convert a value to a finite two-dimensional float64 point.
    """
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite array with shape (2,).")
    return point
