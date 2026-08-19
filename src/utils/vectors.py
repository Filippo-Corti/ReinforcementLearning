"""Small conversions shared at NumPy and PyTorch boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@overload
def to_vector(
    value: object,
    *,
    name: str,
    dimensions: int | None = None,
    readonly: bool = False,
    dtype: Literal["float32"] = "float32",
) -> NDArray[np.float32]: ...


@overload
def to_vector(
    value: object,
    *,
    name: str,
    dimensions: int | None = None,
    readonly: bool = False,
    dtype: Literal["float64"],
) -> NDArray[np.float64]: ...


def to_vector(
    value: object,
    *,
    name: str,
    dimensions: int | None = None,
    readonly: bool = False,
    dtype: Literal["float32", "float64"] = "float32",
) -> NDArray[np.float32] | NDArray[np.float64]:
    """
    Convert a one-dimensional value to an independent floating NumPy array.
    """
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.dtype(dtype))
    expected_shape = None if dimensions is None else (dimensions,)
    if vector.ndim != 1 or (
        expected_shape is not None and vector.shape != expected_shape
    ):
        expected = "a vector" if expected_shape is None else f"shape {expected_shape}"
        raise ValueError(f"{name} must have {expected}; received {vector.shape}.")
    vector = vector.copy()
    if readonly:
        vector.setflags(write=False)
    return vector


def to_tensor(
    value: NDArray[np.float32],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    """
    Convert a NumPy array to a tensor of the given dtype and device.
    """
    return torch.as_tensor(value, dtype=dtype, device=device)


def optional_scalar(value: float | Tensor | None) -> float | None:
    """
    Convert an optional scalar tensor or number to a detached Python float.
    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        return float(value.detach().item())
    return float(value)


def optional_tensor(
    values: Sequence[float | Tensor | None],
    device: torch.device | str | None = None,
) -> Tensor | None:
    """
    Convert an all-present optional scalar sequence to one float32 tensor.

    A field that is absent from every row becomes `None`; mixed presence is
    rejected because it cannot be represented by a dense rollout tensor.
    """
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("A tensor field must be present for every transition or none.")
    return torch.tensor(
        [float(value) for value in values if value is not None],
        dtype=torch.float32,
        device=device,
    )
