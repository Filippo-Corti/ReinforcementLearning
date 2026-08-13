"""Tests for the shared learning diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import torch

from agents.diagnostics import gradient_dispersion

# Features share a constant offset, so a weighting that agrees across samples
# produces the same gradient direction from every sub-batch, while one that
# changes sign sample to sample mostly cancels. That is the contrast the measure
# exists to detect: a return-weighted estimator against a baselined one.
_FEATURES = torch.randn(2048, 3, generator=torch.Generator().manual_seed(0)) + 3.0


def _subbatch_loss(
    parameter: torch.Tensor, weights: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    def build(selected: torch.Tensor) -> torch.Tensor:
        return -(_FEATURES[selected] @ parameter * weights[selected]).mean()

    return build


def test_dispersion_separates_agreeing_samples_from_noisy_ones() -> None:
    """
    A gradient norm cannot rank estimators by noise; this measure must.

    The norm describes one averaged estimate rather than how far an equally
    sized second sample would have landed from it.
    """
    parameter = torch.zeros(3, requires_grad=True)
    noise = torch.randn(2048, generator=torch.Generator().manual_seed(1)).sign()

    agreeing = gradient_dispersion(
        (parameter,), _subbatch_loss(parameter, torch.ones(2048)), 2048, 256
    )
    noisy = gradient_dispersion(
        (parameter,), _subbatch_loss(parameter, noise), 2048, 256
    )

    def measured(row: dict[str, float | int | None], key: str) -> float:
        value = row[key]
        assert value is not None
        return float(value)

    assert measured(agreeing, "gradient_signal_to_noise") > measured(
        noisy, "gradient_signal_to_noise"
    )
    assert measured(agreeing, "gradient_cosine_similarity") > measured(
        noisy, "gradient_cosine_similarity"
    )
    assert measured(agreeing, "gradient_cosine_similarity") > 0.9
    assert agreeing["gradient_subbatch_count"] == 8
    assert agreeing["gradient_subbatch_size"] == 256


def test_dispersion_reports_absent_values_when_it_cannot_measure() -> None:
    parameter = torch.zeros(3, requires_grad=True)
    loss = _subbatch_loss(parameter, torch.ones(2048))

    disabled = gradient_dispersion((parameter,), loss, 2048, None)
    too_small = gradient_dispersion((parameter,), loss, 300, 256)

    assert all(value is None for value in disabled.values())
    assert all(value is None for value in too_small.values())


def test_dispersion_leaves_accumulated_gradients_untouched() -> None:
    """
    The probe runs inside an update, so it must not disturb the optimizer.
    """
    parameter = torch.zeros(3, requires_grad=True)
    parameter.grad = torch.full((3,), 7.0)

    gradient_dispersion(
        (parameter,), _subbatch_loss(parameter, torch.ones(2048)), 2048, 256
    )

    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.full((3,), 7.0))
