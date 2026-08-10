"""Parameter diagnostics shared by project-owned learning agents."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def parameter_norm(parameters: Iterable[Tensor]) -> float:
    """
    Return the global Euclidean norm of one network's parameters.
    """
    squared_norm = torch.zeros(())
    for parameter in parameters:
        component = parameter.detach().square().sum().cpu()
        squared_norm += component
    return float(torch.sqrt(squared_norm).item())


def parameter_update_norm(
    parameters: Iterable[Tensor], before: tuple[Tensor, ...]
) -> float:
    """
    Return the global Euclidean parameter change made by one optimizer step.
    """
    squared_norm = torch.zeros(())
    for parameter, prior in zip(parameters, before, strict=True):
        component = (parameter.detach() - prior).square().sum().cpu()
        squared_norm += component
    return float(torch.sqrt(squared_norm).item())
