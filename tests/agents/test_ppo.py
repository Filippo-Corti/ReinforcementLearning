from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agents import AgentUpdateInput, CollectionMode, PPOAgent
from configs import ActorConfig, CriticConfig, PPOConfig
from tests.fixtures.envs.continuous_control import PositiveThrottleEnvironment
from training import TrainingTransition
from training.buffers import OnPolicyRollout


def _agent(
    seed: int,
    *,
    actor_learning_rate: float = 0.02,
    critic_learning_rate: float = 0.02,
    optimization_epochs: int = 3,
    minibatch_size: int = 3,
) -> PPOAgent:
    return PPOAgent(
        observation_dimensions=1,
        actor_config=ActorConfig(name="small", hidden_sizes=(4, 4)),
        critic_config=CriticConfig(hidden_sizes=(4, 4)),
        config=PPOConfig(
            discount=0.9,
            gae_lambda=0.95,
            transitions_per_rollout=8,
            optimization_epochs=optimization_epochs,
            minibatch_size=minibatch_size,
        ),
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        actor_initialization_generator=torch.Generator().manual_seed(seed),
        critic_initialization_generator=torch.Generator().manual_seed(seed + 50),
        sampling_generator=torch.Generator().manual_seed(seed + 100),
        optimization_generator=torch.Generator().manual_seed(seed + 150),
    )


def _transition(
    agent: PPOAgent,
    *,
    reward: float,
    value: float,
    next_value: float,
    identity: int,
    step: int,
    terminated: bool = False,
    pre_squash_action: tuple[float, float] = (0.1, -0.2),
    behaviour_log_probability: float | None = None,
) -> TrainingTransition:
    observation = np.asarray((1.0,), dtype=np.float32)
    pre_squash = np.asarray(pre_squash_action, dtype=np.float32)
    if behaviour_log_probability is None:
        with torch.inference_mode():
            behaviour_log_probability = float(
                agent.actor.log_probability(
                    torch.as_tensor(observation).unsqueeze(0),
                    torch.as_tensor(pre_squash).unsqueeze(0),
                )[0].item()
            )
    return TrainingTransition(
        normalized_observation=observation,
        raw_action=pre_squash,
        env_action=np.tanh(pre_squash).astype(np.float32),
        reward=reward,
        behaviour_log_probability=behaviour_log_probability,
        current_value=value,
        next_value=next_value,
        terminated=terminated,
        truncated=False,
        next_normalized_observation=observation,
        episode_identity=identity,
        episode_step_index=step,
        circuit_identity="controlled",
    )


def _rollout(agent: PPOAgent) -> OnPolicyRollout:
    environment = PositiveThrottleEnvironment()
    rows = []
    for identity in range(agent.collection_size):
        observation = environment.reset()
        decision = agent.collect_action(observation)
        _, reward = environment.step(decision.env_action)
        rows.append(
            TrainingTransition(
                normalized_observation=observation,
                raw_action=decision.raw_action,
                env_action=decision.env_action,
                reward=reward,
                behaviour_log_probability=decision.behaviour_log_probability,
                current_value=decision.current_value,
                next_value=0.0,
                terminated=True,
                truncated=False,
                next_normalized_observation=observation,
                episode_identity=identity,
                episode_step_index=0,
                circuit_identity="controlled",
            )
        )
    return OnPolicyRollout(tuple(rows))


def test_ppo_clipped_surrogate_matches_positive_and_negative_advantages() -> None:
    agent = _agent(4)
    observations = torch.ones((2, 1))
    pre_squash_actions = torch.tensor(((0.1, -0.2), (0.2, 0.3)))
    with torch.inference_mode():
        current_log_probabilities = agent.actor.log_probability(
            observations, pre_squash_actions
        )
    ratios = torch.tensor((1.4, 0.6))
    old_log_probabilities = current_log_probabilities - ratios.log()
    advantages = torch.tensor((2.0, -3.0))

    loss, actual_ratios, _ = agent._actor_loss(
        observations,
        pre_squash_actions,
        old_log_probabilities,
        advantages,
    )

    expected_loss = -torch.minimum(
        ratios * advantages,
        ratios.clamp(0.8, 1.2) * advantages,
    ).mean()
    torch.testing.assert_close(actual_ratios, ratios)
    torch.testing.assert_close(loss, expected_loss)


def test_ppo_unchanged_policy_has_unit_ratios_and_zero_approximate_kl() -> None:
    agent = _agent(5)
    observations = torch.ones((3, 1))
    pre_squash_actions = torch.tensor(((0.1, -0.2), (0.2, 0.3), (-0.4, 0.5)))
    with torch.inference_mode():
        old_log_probabilities = agent.actor.log_probability(
            observations, pre_squash_actions
        )

    _, ratios, log_ratios = agent._actor_loss(
        observations,
        pre_squash_actions,
        old_log_probabilities,
        torch.tensor((1.0, -1.0, 0.5)),
    )

    torch.testing.assert_close(ratios, torch.ones_like(ratios))
    assert float(((ratios - 1.0) - log_ratios).mean().item()) == pytest.approx(0.0)


def test_ppo_minibatches_cover_every_rollout_row_once_per_epoch() -> None:
    agent = _agent(6, optimization_epochs=4, minibatch_size=3)
    agent.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(agent))
    )

    assert len(agent.last_minibatch_indices) == 4
    for epoch in agent.last_minibatch_indices:
        assert sorted(index for batch in epoch for index in batch) == list(range(8))


def test_ppo_keeps_collection_log_probabilities_fixed_for_all_epochs() -> None:
    agent = _agent(
        7,
        actor_learning_rate=0.0,
        critic_learning_rate=0.0,
        optimization_epochs=3,
        minibatch_size=8,
    )
    log_ratio = math.log(1.3)
    rollout = OnPolicyRollout(
        tuple(
            _transition(
                agent,
                reward=1.0,
                value=0.0,
                next_value=0.0,
                identity=index,
                step=0,
                terminated=True,
                pre_squash_action=(0.1 * (index + 1), -0.2),
                behaviour_log_probability=(
                    float(
                        agent.actor.log_probability(
                            torch.ones((1, 1)),
                            torch.tensor(((0.1 * (index + 1), -0.2),)),
                        )[0].item()
                    )
                    - log_ratio
                ),
            )
            for index in range(8)
        )
    )

    output = agent.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=rollout)
    )

    assert output.diagnostics["ratio_mean"] == pytest.approx(1.3)
    assert output.diagnostics["approximate_kl"] == pytest.approx(
        1.3 - 1.0 - log_ratio, abs=1e-7
    )


def test_controlled_problem_improves_for_all_validation_seeds() -> None:
    observation = np.asarray((1.0,), dtype=np.float32)
    for seed in range(5):
        agent = _agent(seed, optimization_epochs=10, minibatch_size=8)
        before = float(agent.deterministic_action(observation)[0])
        for _ in range(40):
            agent.update(
                AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(agent))
            )
        after = float(agent.deterministic_action(observation)[0])

        assert after > before + 0.2


def test_same_seed_reproduces_minibatch_order_diagnostics_and_parameters() -> None:
    first = _agent(9)
    second = _agent(9)
    first_output = first.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(first))
    )
    second_output = second.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(second))
    )

    assert first.last_minibatch_indices == second.last_minibatch_indices
    assert first_output == second_output
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.actor.parameters(), second.actor.parameters(), strict=True
        )
    )
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.critic.parameters(), second.critic.parameters(), strict=True
        )
    )


def test_ppo_rejects_disabled_optional_objectives() -> None:
    with pytest.raises(ValueError, match="value clipping"):
        PPOAgent(
            observation_dimensions=1,
            actor_config=ActorConfig(name="small", hidden_sizes=(4, 4)),
            critic_config=CriticConfig(hidden_sizes=(4, 4)),
            config=PPOConfig(value_clipping_enabled=True),
            actor_learning_rate=0.01,
            critic_learning_rate=0.01,
            actor_initialization_generator=torch.Generator().manual_seed(1),
            critic_initialization_generator=torch.Generator().manual_seed(2),
            sampling_generator=torch.Generator().manual_seed(3),
            optimization_generator=torch.Generator().manual_seed(4),
        )
