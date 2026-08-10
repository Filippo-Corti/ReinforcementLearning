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


def standardize(values: Tensor, epsilon: float) -> Tensor:
    """
    Return detached population-standardized values with a numerical safeguard.
    """
    return ((values - values.mean()) / (values.std(unbiased=False) + epsilon)).detach()


def explained_variance(value_targets: Tensor, predictions: Tensor) -> float:
    """
    Return one minus residual population variance divided by target variance.
    """
    target_variance = value_targets.var(unbiased=False)
    if target_variance.item() == 0.0:
        return 0.0
    residual_variance = (value_targets - predictions.detach()).var(unbiased=False)
    return float((1.0 - residual_variance / target_variance).item())
