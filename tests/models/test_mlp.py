from __future__ import annotations

import pytest
import torch
from torch import nn

from models import make_mlp


def test_mlp_has_requested_topology_and_initialization() -> None:
    generator = torch.Generator().manual_seed(7)

    model = make_mlp(
        input_dimensions=4,
        output_dimensions=2,
        hidden_sizes=(32, 32),
        activation="tanh",
        hidden_initialization_gain=2**0.5,
        output_initialization_gain=0.01,
        initialization_generator=generator,
    )

    assert [type(module) for module in model] == [
        nn.Linear,
        nn.Tanh,
        nn.Linear,
        nn.Tanh,
        nn.Linear,
    ]
    layers = [module for module in model if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (4, 32),
        (32, 32),
        (32, 2),
    ]
    assert all(
        torch.equal(layer.bias, torch.zeros_like(layer.bias)) for layer in layers
    )
    assert torch.allclose(
        layers[-1].weight @ layers[-1].weight.T, 0.0001 * torch.eye(2)
    )


def test_mlp_rejects_undocumented_activation() -> None:
    with pytest.raises(ValueError, match="Unsupported MLP activation"):
        make_mlp(
            input_dimensions=4,
            output_dimensions=2,
            hidden_sizes=(32, 32),
            activation="relu",
            hidden_initialization_gain=2**0.5,
            output_initialization_gain=0.01,
            initialization_generator=torch.Generator().manual_seed(1),
        )
