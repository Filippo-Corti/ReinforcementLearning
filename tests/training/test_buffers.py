from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.targets import compute_vector_gae_targets, monte_carlo_return_to_go
from training import TrainingTransition
from training.buffers import Trajectory
from training.multienvs import VectorRollout
from utils.vectors import stack_vectors


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
    environment: int = 0,
) -> TrainingTransition:
    return TrainingTransition(
        normalized_observation=np.array([step, step + 0.5], dtype=np.float32),
        raw_action=np.array([step + 0.25, -step - 0.25], dtype=np.float32),
        env_action=np.array([0.5, -0.5], dtype=np.float32),
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
        environment_index=environment,
    )


def test_transition_preserves_vector_fields_as_detached_float32_data() -> None:
    transition = _transition(0)

    assert transition.normalized_observation.shape == (2,)
    assert transition.raw_action.shape == (2,)
    assert transition.env_action.shape == (2,)
    assert transition.next_normalized_observation.shape == (2,)
    assert transition.normalized_observation.dtype == np.float32
    assert transition.behaviour_log_probability == -0.25
    assert transition.current_value == 0.0
    assert transition.next_value == 0.0


def test_transition_detaches_tensor_vector_fields() -> None:
    transition = TrainingTransition(
        normalized_observation=torch.tensor([1.0, 2.0], requires_grad=True),
        raw_action=torch.tensor([0.1, 0.2], requires_grad=True),
        env_action=torch.tensor([0.1, 0.2], requires_grad=True),
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


def test_stacked_transition_fields_have_framework_ready_shapes_and_dtypes() -> None:
    trajectory = Trajectory((_transition(0, next_value=0.2),))

    observations = stack_vectors(
        (transition.normalized_observation for transition in trajectory),
        name="normalized_observation",
    )

    assert observations.shape == (1, 2)
    assert observations.dtype == torch.float32
    assert not observations.requires_grad


def test_monte_carlo_returns_match_complete_time_limited_episode() -> None:
    episode = Trajectory(
        (
            _transition(0, reward=1.0),
            _transition(1, reward=2.0, truncated=True),
        )
    )

    returns = monte_carlo_return_to_go(episode.transitions, discount=0.5)

    torch.testing.assert_close(returns, torch.tensor([2.0, 2.0]))


def test_monte_carlo_returns_reject_incomplete_episode() -> None:
    episode = Trajectory((_transition(0),))

    with pytest.raises(ValueError, match="complete episode"):
        monte_carlo_return_to_go(episode.transitions, discount=0.9)


def _single_column(*transitions: TrainingTransition) -> VectorRollout:
    """
    Wrap one worker's transitions as a one-column vector rollout.

    A single column is the simplest rollout there is, which makes it the right
    shape for pinning down the boundary rules the recursion must obey before a
    second column is introduced.
    """
    rollout = VectorRollout(capacity=len(transitions), environment_count=1)
    for transition in transitions:
        rollout.append_step((transition,))
    return rollout


def test_gae_handles_ordinary_transition_and_rollout_cut() -> None:
    rollout = _single_column(
        _transition(0, reward=1.0, value=0.5, next_value=0.6),
        _transition(1, reward=2.0, value=0.6, next_value=0.7),
    )

    targets = compute_vector_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

    torch.testing.assert_close(
        targets.temporal_difference_errors, torch.tensor([0.8, 1.75])
    )
    torch.testing.assert_close(targets.raw_advantages, torch.tensor([1.2375, 1.75]))
    torch.testing.assert_close(targets.value_targets, torch.tensor([1.7375, 2.35]))
    assert not targets.value_targets.requires_grad


def test_gae_removes_bootstrap_after_true_termination() -> None:
    rollout = _single_column(
        _transition(0, reward=2.0, value=0.5, next_value=0.0, terminated=True)
    )

    targets = compute_vector_gae_targets(rollout, discount=0.5, gae_lambda=0.95)

    torch.testing.assert_close(targets.temporal_difference_errors, torch.tensor([1.5]))
    torch.testing.assert_close(targets.raw_advantages, torch.tensor([1.5]))
    torch.testing.assert_close(targets.value_targets, torch.tensor([2.0]))


def test_transition_rejects_nonzero_bootstrap_after_true_termination() -> None:
    with pytest.raises(ValueError, match="zero bootstrap"):
        _transition(0, next_value=1.0, terminated=True)


def test_gae_bootstraps_once_but_does_not_recurse_after_truncation() -> None:
    rollout = _single_column(
        _transition(0, reward=1.0, value=0.0, next_value=2.0, truncated=True),
        _transition(0, episode=1, reward=8.0, value=0.0, next_value=0.0),
    )

    targets = compute_vector_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

    torch.testing.assert_close(targets.raw_advantages, torch.tensor([2.0, 8.0]))


def test_gae_lambda_zero_equals_one_step_td_errors() -> None:
    rollout = _single_column(
        _transition(0, reward=1.0, value=0.5, next_value=0.6),
        _transition(1, reward=2.0, value=0.6, next_value=0.0, terminated=True),
    )

    targets = compute_vector_gae_targets(rollout, discount=0.5, gae_lambda=0.0)

    torch.testing.assert_close(
        targets.raw_advantages, targets.temporal_difference_errors
    )


def test_gae_combines_truncation_termination_and_final_rollout_cut() -> None:
    rollout = _single_column(
        _transition(0, reward=1.0, value=0.5, next_value=1.0),
        _transition(1, reward=2.0, value=1.0, next_value=4.0, truncated=True),
        _transition(0, episode=1, reward=3.0, value=2.0, next_value=0.0),
        _transition(
            1, episode=1, reward=4.0, value=3.0, next_value=0.0, terminated=True
        ),
        _transition(0, episode=2, reward=5.0, value=1.0, next_value=6.0),
    )

    targets = compute_vector_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

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


def test_vector_rollout_rejects_a_gap_in_one_column() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        _single_column(_transition(0), _transition(2))


def test_vector_rollout_rejects_a_new_episode_without_a_boundary() -> None:
    with pytest.raises(ValueError, match="environment boundary"):
        _single_column(_transition(0), _transition(0, episode=1))


def test_vector_rollout_flattens_time_major_and_survives_being_cleared() -> None:
    rollout = VectorRollout(capacity=3, environment_count=2)
    first = _transition(0, environment=0)
    second = _transition(0, episode=1, environment=1)
    third = _transition(1, environment=0, terminated=True)
    rollout.append_step((first, second))
    rollout.append_step((third, None))

    assert rollout.transitions == (first, second, third)
    assert rollout.transition_count == 3
    assert rollout.remaining_capacity == 0
    assert rollout.transition_steps == ((first, second), (third, None))

    rollout.clear()

    # Emptying starts the next rollout but keeps each column's last transition,
    # so continuity is still checked across the boundary between two rollouts.
    assert rollout.transitions == ()
    assert rollout.previous_transitions == (third, second)


def test_vector_gae_recurses_independently_down_environment_columns() -> None:
    rollout = VectorRollout(capacity=4, environment_count=2)
    rollout.append_step(
        (
            _transition(0, reward=1.0, value=0.5, next_value=0.6, environment=0),
            _transition(
                0, episode=1, reward=10.0, value=1.0, next_value=2.0, environment=1
            ),
        )
    )
    rollout.append_step(
        (
            _transition(
                1,
                reward=2.0,
                value=0.6,
                next_value=0.0,
                terminated=True,
                environment=0,
            ),
            _transition(
                1,
                episode=1,
                reward=20.0,
                value=2.0,
                next_value=4.0,
                truncated=True,
                environment=1,
            ),
        )
    )

    targets = compute_vector_gae_targets(rollout, discount=0.5, gae_lambda=0.5)

    # Flat order is time-major, so the columns interleave: worker 0 then
    # worker 1 at each tick. Neither column borrows from the other.
    torch.testing.assert_close(
        targets.temporal_difference_errors,
        torch.tensor([0.8, 10.0, 1.4, 20.0]),
    )
    torch.testing.assert_close(
        targets.raw_advantages,
        torch.tensor([1.15, 15.0, 1.4, 20.0]),
    )
