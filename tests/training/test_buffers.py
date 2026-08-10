from __future__ import annotations

import numpy as np
import pytest
import torch

from recording import TrainingTransition
from training.buffers import (
    FixedRolloutBuffer,
    OnPolicyRollout,
    ReinforceEpisodeBuffer,
    compute_gae_targets,
    monte_carlo_return_to_go,
)


def _transition(
    step: int,
    *,
    episode: int = 0,
    circuit: str = "track-a",
    reward: float = 1.0,
    value: float | None = 0.0,
    next_value: float | None = 0.0,
    terminated: bool = False,
    truncated: bool = False,
) -> TrainingTransition:
    return TrainingTransition(
        normalized_observation=np.array([step, step + 0.5], dtype=np.float32),
        pre_squash_action=np.array([step + 0.25, -step - 0.25], dtype=np.float32),
        action=np.array([0.5, -0.5], dtype=np.float32),
        reward=reward,
        behaviour_log_probability=torch.tensor(-0.25, requires_grad=True),
        current_value=(
            None if value is None else torch.tensor(value, requires_grad=True)
        ),
        next_value=(
            None if next_value is None else torch.tensor(next_value, requires_grad=True)
        ),
        terminated=terminated,
        truncated=truncated,
        next_normalized_observation=np.array(
            [step + 1.0, step + 1.5], dtype=np.float32
        ),
        episode_identity=episode,
        episode_step_index=step,
        circuit_identity=circuit,
    )


def test_transition_preserves_vector_fields_as_detached_float32_data() -> None:
    transition = _transition(0)

    assert transition.normalized_observation.shape == (2,)
    assert transition.pre_squash_action.shape == (2,)
    assert transition.action.shape == (2,)
    assert transition.next_normalized_observation.shape == (2,)
    assert transition.normalized_observation.dtype == np.float32
    assert transition.behaviour_log_probability == -0.25
    assert transition.current_value == 0.0
    assert transition.next_value == 0.0


def test_transition_detaches_tensor_vector_fields() -> None:
    transition = TrainingTransition(
        normalized_observation=torch.tensor([1.0, 2.0], requires_grad=True),
        pre_squash_action=torch.tensor([0.1, 0.2], requires_grad=True),
        action=torch.tensor([0.1, 0.2], requires_grad=True),
        reward=1.0,
        behaviour_log_probability=0.0,
        current_value=0.0,
        next_value=0.0,
        terminated=False,
        truncated=False,
        next_normalized_observation=torch.tensor([3.0, 4.0], requires_grad=True),
        episode_identity=0,
        episode_step_index=0,
        circuit_identity="track-a",
    )

    np.testing.assert_array_equal(
        transition.normalized_observation, np.array([1.0, 2.0], dtype=np.float32)
    )


def test_rollout_tensors_have_framework_ready_vector_shapes_and_dtypes() -> None:
    rollout = OnPolicyRollout((_transition(0, next_value=0.2),))

    tensors = rollout.tensors()

    assert tensors.observations.shape == (1, 2)
    assert tensors.pre_squash_actions.shape == (1, 2)
    assert tensors.actions.shape == (1, 2)
    assert tensors.next_observations.shape == (1, 2)
    assert tensors.observations.dtype == torch.float32
    assert tensors.terminated.dtype == torch.bool
    assert tensors.current_values is not None
    assert not tensors.current_values.requires_grad


def test_monte_carlo_returns_match_complete_time_limited_episode() -> None:
    episode = OnPolicyRollout(
        (
            _transition(0, reward=1.0),
            _transition(1, reward=2.0, truncated=True),
        )
    )

    returns = monte_carlo_return_to_go(episode, discount=0.5)

    torch.testing.assert_close(returns, torch.tensor([2.0, 2.0]))


def test_monte_carlo_returns_reject_incomplete_episode() -> None:
    episode = OnPolicyRollout((_transition(0),))

    with pytest.raises(ValueError, match="complete episode"):
        monte_carlo_return_to_go(episode, discount=0.9)


def test_gae_handles_ordinary_transition_and_rollout_cut() -> None:
    rollout = OnPolicyRollout(
        (
            _transition(0, reward=1.0, value=0.5, next_value=0.6),
            _transition(1, reward=2.0, value=0.6, next_value=0.7),
        )
    )

    targets = compute_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

    torch.testing.assert_close(
        targets.temporal_difference_errors, torch.tensor([0.8, 1.75])
    )
    torch.testing.assert_close(targets.raw_advantages, torch.tensor([1.2375, 1.75]))
    torch.testing.assert_close(targets.value_targets, torch.tensor([1.7375, 2.35]))
    assert not targets.value_targets.requires_grad


def test_gae_removes_bootstrap_after_true_termination() -> None:
    rollout = OnPolicyRollout(
        (_transition(0, reward=2.0, value=0.5, next_value=0.0, terminated=True),)
    )

    targets = compute_gae_targets(rollout, discount=0.5, gae_lambda=0.95)

    torch.testing.assert_close(targets.temporal_difference_errors, torch.tensor([1.5]))
    torch.testing.assert_close(targets.raw_advantages, torch.tensor([1.5]))
    torch.testing.assert_close(targets.value_targets, torch.tensor([2.0]))


def test_transition_rejects_nonzero_bootstrap_after_true_termination() -> None:
    with pytest.raises(ValueError, match="zero bootstrap"):
        _transition(0, next_value=1.0, terminated=True)


def test_gae_bootstraps_once_but_does_not_recurse_after_truncation() -> None:
    rollout = OnPolicyRollout(
        (
            _transition(0, reward=1.0, value=0.0, next_value=2.0, truncated=True),
            _transition(0, episode=1, reward=8.0, value=0.0, next_value=0.0),
        )
    )

    targets = compute_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

    torch.testing.assert_close(targets.raw_advantages, torch.tensor([2.0, 8.0]))


def test_gae_lambda_zero_equals_one_step_td_errors() -> None:
    rollout = OnPolicyRollout(
        (
            _transition(0, reward=1.0, value=0.5, next_value=0.6),
            _transition(1, reward=2.0, value=0.6, next_value=0.0, terminated=True),
        )
    )

    targets = compute_gae_targets(rollout, discount=0.5, gae_lambda=0.0)

    torch.testing.assert_close(
        targets.raw_advantages, targets.temporal_difference_errors
    )


def test_gae_combines_truncation_termination_and_final_rollout_cut() -> None:
    rollout = OnPolicyRollout(
        (
            _transition(0, reward=1.0, value=0.5, next_value=1.0),
            _transition(
                1,
                reward=2.0,
                value=1.0,
                next_value=4.0,
                truncated=True,
            ),
            _transition(0, episode=1, reward=3.0, value=2.0, next_value=0.0),
            _transition(
                1,
                episode=1,
                reward=4.0,
                value=3.0,
                next_value=0.0,
                terminated=True,
            ),
            _transition(0, episode=2, reward=5.0, value=1.0, next_value=6.0),
        )
    )

    targets = compute_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

    torch.testing.assert_close(
        targets.temporal_difference_errors,
        torch.tensor([1.0, 3.0, 1.0, 1.0, 7.0]),
    )
    torch.testing.assert_close(
        targets.raw_advantages,
        torch.tensor([1.75, 3.0, 1.25, 1.0, 7.0]),
    )
    torch.testing.assert_close(
        targets.value_targets,
        torch.tensor([2.25, 4.0, 3.25, 4.0, 8.0]),
    )


def test_reinforce_buffer_excludes_incomplete_episodes_and_preserves_rows() -> None:
    buffer = ReinforceEpisodeBuffer()
    first = _transition(0, reward=1.0)
    second = _transition(1, reward=2.0, terminated=True)
    buffer.append(first)
    buffer.append(second)

    completed = buffer.finalize_episode()

    assert completed.transitions == (first, second)
    assert buffer.take_completed_batch(2) is None
    assert buffer.take_completed_batch(1) == (completed,)
    assert buffer.active_episode == ()


def test_reinforce_buffer_rejects_incomplete_or_cross_episode_finalization() -> None:
    buffer = ReinforceEpisodeBuffer()
    buffer.append(_transition(0))

    with pytest.raises(ValueError, match="terminated or truncated"):
        buffer.finalize_episode()
    with pytest.raises(ValueError, match="cannot mix episode identities"):
        buffer.append(_transition(1, episode=1))


def test_fixed_rollout_preserves_multi_episode_rows_and_resets_after_finalize() -> None:
    buffer = FixedRolloutBuffer(capacity=3)
    first = _transition(3, episode=4, reward=1.0)
    second = _transition(4, episode=4, reward=2.0, truncated=True)
    third = _transition(0, episode=5, circuit="track-b", reward=3.0)
    buffer.append(first)
    buffer.append(second)
    buffer.append(third)

    rollout = buffer.finalize()

    assert rollout.transitions == (first, second, third)
    assert buffer.transitions == ()
    buffer.append(_transition(1, episode=5, circuit="track-b"))
    assert buffer.transitions[0].episode_step_index == 1


def test_fixed_rollout_rejects_duplicate_row_after_a_rollout_cut() -> None:
    buffer = FixedRolloutBuffer(capacity=2)
    buffer.append(_transition(0))
    buffer.finalize()

    with pytest.raises(ValueError, match="consecutive"):
        buffer.append(_transition(0))


def test_fixed_rollout_rejects_cross_episode_without_boundary_and_overflow() -> None:
    buffer = FixedRolloutBuffer(capacity=1)
    buffer.append(_transition(0))

    with pytest.raises(ValueError, match="Finalize the full rollout"):
        buffer.append(_transition(1))

    buffer = FixedRolloutBuffer(capacity=3)
    buffer.append(_transition(0))
    with pytest.raises(ValueError, match="environment boundary"):
        buffer.append(_transition(0, episode=1))
