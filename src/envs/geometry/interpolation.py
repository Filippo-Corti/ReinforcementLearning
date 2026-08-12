"""Scalar evaluation of piecewise polynomials built by SciPy."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

from ..types import FloatArray


class ScalarPiecewisePolynomial:
    """
    Evaluate one SciPy piecewise polynomial at a single argument.

    SciPy builds the coefficients, but the environment evaluates them at one
    arc length per physics substep, where SciPy's array-dispatch machinery costs
    far more than the polynomial arithmetic itself. This class keeps the
    identical coefficients and applies Horner's rule directly, so results match
    the original interpolant while removing the per-call overhead.

    Fields:
        * breakpoints: Ascending interval boundaries of the interpolant.
        * coefficients: Coefficients per interval, highest power first.
    """

    def __init__(self, breakpoints: FloatArray, coefficients: FloatArray) -> None:
        self.breakpoints = np.ascontiguousarray(breakpoints, dtype=np.float64)
        self.coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
        self._last_interval = self.coefficients.shape[1] - 1

    @classmethod
    def periodic_spline(
        cls,
        breakpoints: FloatArray,
        values: FloatArray,
    ) -> ScalarPiecewisePolynomial:
        """
        Build a periodic cubic spline and keep only its scalar evaluation.
        """
        return cls.of(CubicSpline(breakpoints, values, bc_type="periodic"))

    @classmethod
    def of(cls, polynomial: PPoly) -> ScalarPiecewisePolynomial:
        """
        Adopt the coefficients of an existing SciPy piecewise polynomial.
        """
        return cls(polynomial.x, polynomial.c)

    def __call__(self, argument: float) -> float | FloatArray:
        """
        Return the interpolated value, scalar or vector, at one argument.
        """
        interval = int(np.searchsorted(self.breakpoints, argument, side="right")) - 1
        interval = min(max(interval, 0), self._last_interval)
        offset = argument - self.breakpoints[interval]
        coefficients = self.coefficients[:, interval]
        value = coefficients[0]
        for power in range(1, coefficients.shape[0]):
            value = value * offset + coefficients[power]
        return value if value.ndim else float(value)
