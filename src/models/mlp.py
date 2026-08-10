"""Reusable multilayer perceptron construction for learning models."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


def make_mlp(
    input_dimensions: int,
    output_dimensions: int,
    hidden_sizes: Sequence[int],
    activation: str,
    hidden_initialization_gain: float,
    output_initialization_gain: float,
    initialization_generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> nn.Sequential:
    """
    Build an orthogonally initialized fully connected network.

    Layers are first created on PyTorch's meta device so their default
    initialization cannot consume the process-wide random stream. Every real
    parameter is then initialized from ``initialization_generator``.
    """
    if activation != "tanh":
        raise ValueError(f"Unsupported MLP activation: {activation!r}.")

    layer_dimensions = (input_dimensions, *hidden_sizes, output_dimensions)
    modules: list[nn.Module] = []
    for index, (features_in, features_out) in enumerate(pairwise(layer_dimensions)):
        layer = nn.Linear(
            features_in,
            features_out,
            device="meta",
            dtype=dtype,
        ).to_empty(device=device)
        gain = (
            output_initialization_gain
            if index == len(layer_dimensions) - 2
            else hidden_initialization_gain
        )
        nn.init.orthogonal_(
            layer.weight,
            gain=gain,
            generator=initialization_generator,
        )
        nn.init.zeros_(layer.bias)
        modules.append(layer)
        if index != len(layer_dimensions) - 2:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)
