from __future__ import annotations

import torch

from configs import FIXED_CRITIC_CONFIG, SMALL_ACTOR_CONFIG
from models import GaussianPolicy, ValueNetwork, agent_parameter_counts


def _critic(seed: int = 21) -> ValueNetwork:
    return ValueNetwork(
        observation_dimensions=4,
        config=FIXED_CRITIC_CONFIG,
        initialization_generator=torch.Generator().manual_seed(seed),
    )


def test_value_network_returns_scalar_for_single_and_batched_observations() -> None:
    critic = _critic()

    assert critic(torch.zeros(4)).shape == ()
    assert critic(torch.zeros(3, 4)).shape == (3,)


def test_critic_count_matches_documented_formula_and_gradients_reach_every_parameter() -> (
    None
):
    critic = _critic()
    actor = GaussianPolicy(
        observation_dimensions=4,
        config=SMALL_ACTOR_CONFIG,
        initialization_generator=torch.Generator().manual_seed(22),
    )

    counts = agent_parameter_counts(actor, critic)
    assert counts.critic == (4 + 1) * 64 + (64 + 1) * 64 + (64 + 1)
    assert counts.total == counts.actor + counts.critic

    critic(torch.randn(5, 4)).square().mean().backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in critic.parameters()
    )
