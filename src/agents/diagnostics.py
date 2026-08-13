"""Parameter diagnostics shared by project-owned learning agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

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


def gradient_dispersion(
    parameters: Sequence[Tensor],
    subbatch_loss: Callable[[Tensor], Tensor],
    transition_count: int,
    subbatch_size: int | None,
    maximum_subbatches: int = 8,
) -> dict[str, float | int | None]:
    """
    Measure how much the policy-gradient estimate varies between equal samples.

    The variance of the gradient estimator is the quantity that separates these
    algorithms in theory: a return-weighted score function is unbiased but noisy,
    and subtracting a learned baseline is supposed to reduce that noise. A
    gradient norm cannot show it, because it describes one averaged estimate
    rather than how far another sample would have landed from it.

    The batch is split into equal disjoint sub-batches, each estimating the same
    gradient, and their spread is summarized by a signal-to-noise ratio
    `|mean|^2 / mean |deviation|^2` and by the mean pairwise cosine similarity.
    Both are scale free, so they compare across algorithms whose losses carry
    different constant factors.

    The sub-batch size is fixed rather than proportional, because estimator
    variance falls as one over the sample count and equal-sized samples are the
    only fair comparison. Sub-batches are strided rather than contiguous so each
    one spans the whole rollout instead of a correlated stretch of it, and
    striding needs no random draw, so the probe consumes no seeded stream.
    Gradients are taken through `torch.autograd.grad`, which leaves the
    accumulated `.grad` buffers the optimizer uses untouched.
    """
    absent: dict[str, float | int | None] = {
        "gradient_signal_to_noise": None,
        "gradient_cosine_similarity": None,
        "gradient_subbatch_count": None,
        "gradient_subbatch_size": None,
    }
    if subbatch_size is None or transition_count < 2 * subbatch_size:
        return absent

    count = min(maximum_subbatches, transition_count // subbatch_size)
    indices = torch.arange(transition_count)
    vectors: list[Tensor] = []
    for offset in range(count):
        selected = indices[offset::count][:subbatch_size]
        gradients = torch.autograd.grad(
            subbatch_loss(selected), parameters, allow_unused=True
        )
        vectors.append(
            torch.cat(
                [
                    (
                        torch.zeros_like(parameter)
                        if gradient is None
                        else gradient.detach()
                    ).reshape(-1)
                    for gradient, parameter in zip(gradients, parameters, strict=True)
                ]
            )
        )

    stacked = torch.stack(vectors)
    mean = stacked.mean(dim=0)
    noise = (stacked - mean).square().sum(dim=1).mean()
    signal = mean.square().sum()
    normalized = torch.nn.functional.normalize(stacked, dim=1)
    similarity = normalized @ normalized.T
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    return {
        "gradient_signal_to_noise": (
            float("inf") if noise.item() == 0.0 else float((signal / noise).item())
        ),
        "gradient_cosine_similarity": float(similarity[off_diagonal].mean().item()),
        "gradient_subbatch_count": count,
        "gradient_subbatch_size": subbatch_size,
    }
