from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import A2CAgent, AgentUpdateInput, CollectedAction, CollectionMode
from configs import A2CConfig, ActorConfig, CriticConfig
from tests.fixtures.envs.continuous_control import PositiveThrottleEnvironment
from training import TrainingTransition
from training.buffers import OnPolicyRollout, compute_gae_targets


def _agent(
    seed: int,
    *,
    learning_rate: float = 0.02,
    gae_lambda: float = 0.95,
) -> A2CAgent:
    return A2CAgent(
        observation_dimensions=1,
        actor_config=ActorConfig(name="small", hidden_sizes=(4, 4)),
        critic_config=CriticConfig(hidden_sizes=(4, 4)),
        config=A2CConfig(
            discount=0.9,
            gae_lambda=gae_lambda,
            transitions_per_rollout=8,
        ),
        actor_learning_rate=learning_rate,
        critic_learning_rate=learning_rate,
        actor_initialization_generator=torch.Generator().manual_seed(seed),
        critic_initialization_generator=torch.Generator().manual_seed(seed + 50),
        sampling_generator=torch.Generator().manual_seed(seed + 100),
    )


def _transition(
    agent: A2CAgent,
    *,
    reward: float,
    value: float,
    next_value: float,
    identity: int,
    step: int,
    terminated: bool = False,
    pre_squash_action: tuple[float, float] | None = None,
) -> TrainingTransition:
    observation = np.asarray((1.0,), dtype=np.float32)
    if pre_squash_action is None:
        choice = agent.collect_action(observation)
    else:
        pre_squash = np.asarray(pre_squash_action, dtype=np.float32)
        choice = CollectedAction(
            raw_action=pre_squash,
            env_action=np.tanh(pre_squash).astype(np.float32),
            behaviour_log_probability=None,
            current_value=value,
        )
    return TrainingTransition(
        normalized_observation=observation,
        raw_action=choice.raw_action,
        env_action=choice.env_action,
        reward=reward,
        behaviour_log_probability=choice.behaviour_log_probability,
        current_value=value,
        next_value=next_value,
        terminated=terminated,
        truncated=False,
        next_normalized_observation=observation,
        episode_identity=identity,
        episode_step_index=step,
        circuit_identity="controlled",
    )


def _rollout(agent: A2CAgent) -> OnPolicyRollout:
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


def test_a2c_losses_match_the_documented_mean_reductions() -> None:
    agent = _agent(4)
    rollout = OnPolicyRollout(
        (
            _transition(
                agent,
                reward=1.0,
                value=0.5,
                next_value=0.6,
                identity=0,
                step=0,
                pre_squash_action=(-0.5, 0.2),
            ),
            _transition(
                agent,
                reward=2.0,
                value=0.6,
                next_value=0.0,
                identity=0,
                step=1,
                terminated=True,
                pre_squash_action=(0.8, -0.1),
            ),
        )
    )
    targets = compute_gae_targets(rollout, discount=0.9, gae_lambda=0.95)
    advantages = agent._standardize_advantages(targets.raw_advantages)
    tensors = rollout.tensors()
    log_probabilities = agent.actor.log_probability(
        tensors.observations, tensors.raw_actions
    )
    expected_actor_loss = -(log_probabilities * advantages).mean()
    predictions = agent.critic(tensors.observations)
    expected_critic_loss = 0.5 * (predictions - targets.value_targets).square().mean()

    output = agent.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=rollout)
    )

    assert output.diagnostics["actor_loss"] == pytest.approx(
        float(expected_actor_loss.item())
    )
    assert output.diagnostics["critic_loss"] == pytest.approx(
        float(expected_critic_loss.item())
    )


def test_a2c_lambda_zero_uses_one_step_advantages_and_targets() -> None:
    agent = _agent(5, gae_lambda=0.0)
    rollout = OnPolicyRollout(
        (
            _transition(
                agent,
                reward=1.0,
                value=0.5,
                next_value=0.6,
                identity=0,
                step=0,
                pre_squash_action=(-0.5, 0.2),
            ),
            _transition(
                agent,
                reward=2.0,
                value=0.6,
                next_value=0.0,
                identity=0,
                step=1,
                terminated=True,
                pre_squash_action=(0.8, -0.1),
            ),
        )
    )

    targets = compute_gae_targets(rollout, discount=0.9, gae_lambda=0.0)

    torch.testing.assert_close(
        targets.raw_advantages, targets.temporal_difference_errors
    )
    torch.testing.assert_close(targets.value_targets, torch.tensor([1.54, 2.0]))


def test_a2c_actor_and_critic_gradients_are_isolated() -> None:
    agent = _agent(6)
    rollout = _rollout(agent)
    targets = compute_gae_targets(rollout, discount=0.9, gae_lambda=0.95)
    advantages = agent._standardize_advantages(targets.raw_advantages)

    agent.actor_optimizer.zero_grad(set_to_none=True)
    agent.critic_optimizer.zero_grad(set_to_none=True)
    agent._actor_loss(rollout, advantages).backward()

    assert all(parameter.grad is not None for parameter in agent.actor.parameters())
    assert all(parameter.grad is None for parameter in agent.critic.parameters())

    agent.actor_optimizer.zero_grad(set_to_none=True)
    agent.critic_optimizer.zero_grad(set_to_none=True)
    tensors = rollout.tensors()
    critic_loss, _ = agent._critic_loss(tensors.observations, targets.value_targets)
    critic_loss.backward()

    assert all(parameter.grad is None for parameter in agent.actor.parameters())
    assert all(parameter.grad is not None for parameter in agent.critic.parameters())


def test_controlled_problem_improves_for_all_validation_seeds() -> None:
    observation = np.asarray((1.0,), dtype=np.float32)
    for seed in range(5):
        agent = _agent(seed)
        before = float(agent.deterministic_action(observation)[0])
        for _ in range(40):
            agent.update(
                AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(agent))
            )
        after = float(agent.deterministic_action(observation)[0])

        assert after > before + 0.2


def test_same_seed_reproduces_actions_diagnostics_and_parameters() -> None:
    first = _agent(9)
    second = _agent(9)
    first_output = first.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(first))
    )
    second_output = second.update(
        AgentUpdateInput(CollectionMode.FIXED_ROLLOUT, rollout=_rollout(second))
    )

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
