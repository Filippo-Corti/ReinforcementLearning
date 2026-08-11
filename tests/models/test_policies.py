from __future__ import annotations

from math import exp, log

import torch

from configs import LARGE_ACTOR_CONFIG, MEDIUM_ACTOR_CONFIG, SMALL_ACTOR_CONFIG
from models import ActorNetwork, agent_parameter_counts


def _actor(config=SMALL_ACTOR_CONFIG, seed: int = 10) -> ActorNetwork:
    return ActorNetwork(
        observation_dimensions=4,
        config=config,
        initialization_generator=torch.Generator().manual_seed(seed),
    )


def test_actor_sizes_match_documented_parameter_formula() -> None:
    for config in (SMALL_ACTOR_CONFIG, MEDIUM_ACTOR_CONFIG, LARGE_ACTOR_CONFIG):
        first_hidden, second_hidden = config.hidden_sizes
        expected = (4 + 1) * first_hidden + (first_hidden + 1) * second_hidden
        expected += (second_hidden + 1) * 2 + 2

        assert agent_parameter_counts(_actor(config)).actor == expected


def test_sample_and_recomputed_probability_support_single_and_batched_inputs() -> None:
    actor = _actor()
    single = actor.sample(torch.zeros(4), torch.Generator().manual_seed(11))
    batch = actor.sample(torch.zeros(3, 4), torch.Generator().manual_seed(11))

    assert single.env_action.shape == (2,)
    assert single.raw_action.shape == (2,)
    assert single.log_probability.shape == ()
    assert batch.env_action.shape == (3, 2)
    assert batch.raw_action.shape == (3, 2)
    assert batch.log_probability.shape == (3,)
    assert torch.all(single.env_action > -1.0)
    assert torch.all(single.env_action < 1.0)
    assert torch.all(batch.env_action > -1.0)
    assert torch.all(batch.env_action < 1.0)
    assert not single.env_action.requires_grad
    assert not single.raw_action.requires_grad
    assert not single.log_probability.requires_grad
    assert torch.allclose(
        actor.log_probability(torch.zeros(4), single.raw_action),
        single.log_probability,
    )


def test_vector_sample_batches_forward_pass_but_keeps_row_generators_independent() -> (
    None
):
    actor = _actor()
    observations = torch.asarray(((0.1, 0.2, 0.3, 0.4), (1.0, 2.0, 3.0, 4.0)))
    batched = actor.sample_with_generators(
        observations,
        (torch.Generator().manual_seed(21), torch.Generator().manual_seed(22)),
    )
    expected = tuple(
        actor.sample(observation, torch.Generator().manual_seed(seed))
        for observation, seed in zip(observations, (21, 22), strict=True)
    )

    torch.testing.assert_close(
        batched.raw_action,
        torch.stack([sample.raw_action for sample in expected]),
    )
    torch.testing.assert_close(
        batched.log_probability,
        torch.stack([sample.log_probability for sample in expected]),
    )


def test_log_probability_matches_independent_squashed_gaussian_calculation() -> None:
    actor = _actor()
    observations = torch.tensor([[0.3, -0.7, 1.2, 0.1], [-0.2, 0.4, 0.0, 1.1]])
    pre_squash_actions = torch.tensor([[0.2, -0.5], [0.7, 0.1]])

    mean = actor.policy.mean(observations)
    standard_deviation = actor.policy.standard_deviation
    gaussian = -0.5 * (
        ((pre_squash_actions - mean) / standard_deviation).square()
        + 2.0 * torch.log(standard_deviation)
        + log(2.0 * torch.pi)
    )
    jacobian = torch.log(1.0 - torch.tanh(pre_squash_actions).square())
    expected = (gaussian - jacobian).sum(dim=-1)

    assert torch.allclose(
        actor.log_probability(observations, pre_squash_actions), expected
    )


def test_log_probability_reaches_mean_and_dispersion_parameters() -> None:
    actor = _actor()
    observations = torch.randn(5, 4)
    pre_squash_actions = torch.randn(5, 2)

    loss = -actor.log_probability(observations, pre_squash_actions).mean()
    loss.backward()

    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in actor.parameters()
    )


def test_dispersion_bounds_and_extreme_actions_have_finite_log_probability() -> None:
    actor = _actor()
    with torch.no_grad():
        actor.policy.log_standard_deviation.fill_(100.0)
    assert torch.allclose(actor.policy.standard_deviation, torch.full((2,), exp(2.0)))

    with torch.no_grad():
        actor.policy.log_standard_deviation.fill_(-100.0)
    assert torch.allclose(actor.policy.standard_deviation, torch.full((2,), exp(-5.0)))

    log_probability = actor.log_probability(
        torch.zeros(2, 4),
        torch.tensor([[50.0, -50.0], [-25.0, 25.0]]),
    )
    assert torch.isfinite(log_probability).all()


def test_initialization_and_sampling_are_seed_reproducible_without_global_rng_change() -> (
    None
):
    torch.manual_seed(991)
    expected_global_draw = torch.rand(5)
    torch.manual_seed(991)
    first = _actor(seed=12)
    observed_global_draw = torch.rand(5)
    second = _actor(seed=12)
    third = _actor(seed=13)

    assert torch.equal(observed_global_draw, expected_global_draw)
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters()
        )
    )
    assert any(
        not torch.equal(first_parameter, third_parameter)
        for first_parameter, third_parameter in zip(
            first.parameters(), third.parameters()
        )
    )

    observations = torch.ones(2, 4)
    first_sample = first.sample(observations, torch.Generator().manual_seed(14))
    second_sample = second.sample(observations, torch.Generator().manual_seed(14))
    assert torch.equal(first_sample.raw_action, second_sample.raw_action)
    assert torch.equal(first_sample.env_action, second_sample.env_action)


def test_deterministic_action_does_not_consume_sampling_generator() -> None:
    actor = _actor()
    generator = torch.Generator().manual_seed(15)
    expected_generator = torch.Generator().manual_seed(15)

    actor.deterministic_action(torch.ones(4))

    assert torch.equal(
        torch.rand(4, generator=generator), torch.rand(4, generator=expected_generator)
    )
