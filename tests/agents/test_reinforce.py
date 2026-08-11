from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import AgentUpdateInput, CollectedAction, CollectionMode, ReinforceAgent
from configs import ActorConfig, ReinforceConfig
from tests.fixtures.envs.continuous_control import PositiveThrottleEnvironment
from training import TrainingTransition
from training.buffers import OnPolicyRollout


def _agent(
    seed: int,
    *,
    learning_rate: float = 0.02,
    actor_weight_decay: float = 0.0,
) -> ReinforceAgent:
    return ReinforceAgent(
        observation_dimensions=1,
        actor_config=ActorConfig(
            name="small", hidden_sizes=(4, 4), learning_rate=learning_rate
        ),
        config=ReinforceConfig(
            discount=0.9,
            actor_weight_decay=actor_weight_decay,
        ),
        initialization_generator=torch.Generator().manual_seed(seed),
        sampling_generator=torch.Generator().manual_seed(seed + 100),
    )


def _episode(
    agent: ReinforceAgent,
    *,
    reward: float,
    pre_squash_action: tuple[float, float] | None = None,
    identity: int,
) -> OnPolicyRollout:
    observation = np.asarray((1.0,), dtype=np.float32)
    if pre_squash_action is None:
        choice = agent.collect_action(observation)
    else:
        pre_squash = np.asarray(pre_squash_action, dtype=np.float32)
        choice = CollectedAction(
            raw_action=pre_squash,
            env_action=np.tanh(pre_squash).astype(np.float32),
            behaviour_log_probability=None,
            current_value=None,
        )
    return OnPolicyRollout(
        (
            TrainingTransition(
                normalized_observation=observation,
                raw_action=choice.raw_action,
                env_action=choice.env_action,
                reward=reward,
                behaviour_log_probability=choice.behaviour_log_probability,
                current_value=None,
                next_value=None,
                terminated=True,
                truncated=False,
                next_normalized_observation=observation,
                episode_identity=identity,
                episode_step_index=0,
                circuit_identity="controlled",
            ),
        )
    )


def _batch(agent: ReinforceAgent) -> AgentUpdateInput:
    environment = PositiveThrottleEnvironment()
    episodes = []
    for identity in range(agent.collection_size):
        observation = environment.reset()
        decision = agent.collect_action(observation)
        _, reward = environment.step(decision.env_action)
        episodes.append(
            OnPolicyRollout(
                (
                    TrainingTransition(
                        normalized_observation=observation,
                        raw_action=decision.raw_action,
                        env_action=decision.env_action,
                        reward=reward,
                        behaviour_log_probability=decision.behaviour_log_probability,
                        current_value=None,
                        next_value=None,
                        terminated=True,
                        truncated=False,
                        next_normalized_observation=observation,
                        episode_identity=identity,
                        episode_step_index=0,
                        circuit_identity="controlled",
                    ),
                )
            )
        )
    return AgentUpdateInput(CollectionMode.COMPLETE_EPISODES, tuple(episodes))


def test_reinforce_requires_an_explicit_actor_learning_rate() -> None:
    with pytest.raises(ValueError, match="explicit learning rate"):
        ReinforceAgent(
            observation_dimensions=1,
            actor_config=ActorConfig(name="small", hidden_sizes=(4, 4)),
            config=ReinforceConfig(discount=0.9),
            initialization_generator=torch.Generator().manual_seed(1),
            sampling_generator=torch.Generator().manual_seed(2),
        )


def test_reinforce_weight_decay_applies_only_to_mlp_weights() -> None:
    weight_decay = 1e-4
    agent = _agent(3, actor_weight_decay=weight_decay)
    regularized = next(
        group
        for group in agent.optimizer.param_groups
        if group["weight_decay"] == weight_decay
    )
    unregularized = next(
        group for group in agent.optimizer.param_groups if group["weight_decay"] == 0.0
    )
    named_parameters = dict(agent.actor.named_parameters())

    assert {id(parameter) for parameter in regularized["params"]} == {
        id(parameter)
        for name, parameter in named_parameters.items()
        if name.endswith(".weight")
    }
    assert {id(parameter) for parameter in unregularized["params"]} == {
        id(parameter)
        for name, parameter in named_parameters.items()
        if not name.endswith(".weight")
    }
    assert id(agent.actor.policy.log_standard_deviation) in {
        id(parameter) for parameter in unregularized["params"]
    }


def test_reinforce_rejects_negative_actor_weight_decay() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        _agent(3, actor_weight_decay=-1e-4)


def test_reinforce_loss_matches_the_documented_per_trajectory_reduction() -> None:
    agent = _agent(4)
    pre_squash_actions = (-1.5, -0.5, -0.2, 0.1, 0.4, 0.8, 1.1, 1.8)
    episodes = tuple(
        _episode(
            agent,
            reward=reward,
            pre_squash_action=(pre_squash, 0.2),
            identity=identity,
        )
        for identity, (pre_squash, reward) in enumerate(
            zip(pre_squash_actions, range(1, 9), strict=True)
        )
    )
    returns = torch.arange(1, 9, dtype=torch.float32)
    standardized = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    log_probabilities = torch.cat(
        tuple(agent._log_probabilities(episode) for episode in episodes)
    )
    expected = -(log_probabilities * standardized).sum() / len(episodes)

    result = agent.update(AgentUpdateInput(CollectionMode.COMPLETE_EPISODES, episodes))

    assert result.diagnostics["actor_loss"] == pytest.approx(float(expected.item()))


def test_one_controlled_update_increases_the_probability_of_positive_throttle() -> None:
    agent = _agent(5)
    pre_squash_actions = (-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0)
    episodes = tuple(
        _episode(
            agent,
            reward=float(np.tanh(pre_squash)),
            pre_squash_action=(pre_squash, 0.0),
            identity=identity,
        )
        for identity, pre_squash in enumerate(pre_squash_actions)
    )
    observation = np.asarray((1.0,), dtype=np.float32)
    before = float(agent.deterministic_action(observation)[0])

    agent.update(AgentUpdateInput(CollectionMode.COMPLETE_EPISODES, episodes))

    assert float(agent.deterministic_action(observation)[0]) > before


def test_controlled_problem_improves_for_all_validation_seeds() -> None:
    observation = np.asarray((1.0,), dtype=np.float32)
    for seed in range(5):
        agent = _agent(seed)
        before = float(agent.deterministic_action(observation)[0])
        for _ in range(40):
            agent.update(_batch(agent))
        after = float(agent.deterministic_action(observation)[0])

        assert after > before + 0.2


def test_same_seed_reproduces_actions_diagnostics_and_parameters() -> None:
    first = _agent(9)
    second = _agent(9)
    first_output = first.update(_batch(first))
    second_output = second.update(_batch(second))

    assert first_output == second_output
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.actor.parameters(), second.actor.parameters(), strict=True
        )
    )
