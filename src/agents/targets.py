"""What each algorithm asks the collected experience to predict.

The learning target is the difference between these algorithms. REINFORCE
waits for the real return; the actor-critic pair bootstraps and blends the
horizons with GAE. That is a statement about the estimator, not about how the
transitions were gathered, so it lives with the agents rather than with the
buffers they read.

Both functions return values in the same flat order as the transitions they
were given, which is the order the losses consume.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from training.buffers import TrainingTransition
    from training.multienvs import VectorRollout


@dataclass(frozen=True, slots=True)
class GAETargets:
    """
    Store the detached quantities one actor-critic update is fitted to.

    Fields:
        * temporal_difference_errors: Detached one-step TD errors.
        * raw_advantages: Detached GAE values before actor-only standardization.
        * value_targets: Detached critic targets, advantage plus the old value.
    """

    temporal_difference_errors: Tensor
    raw_advantages: Tensor
    value_targets: Tensor


def monte_carlo_return_to_go(
    transitions: Sequence[TrainingTransition],
    discount: float,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Compute the detached return from each state to the end of the episode.

    Requires transitions that actually reach an environment boundary: without
    one the later returns are unknown, and treating the last collected step as
    the end would quietly report a truncated sum as a complete return.
    """
    if not transitions or not transitions[-1].ends_episode:
        raise ValueError("Monte Carlo return-to-go requires a complete episode.")
    returns = torch.empty(len(transitions), dtype=torch.float32, device=device)
    running_return = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        running_return = transitions[index].reward + discount * running_return
        returns[index] = running_return
    return returns.detach()


def compute_vector_gae_targets(
    rollout: VectorRollout,
    discount: float,
    gae_lambda: float,
    *,
    device: torch.device | str | None = None,
) -> GAETargets:
    """
    Blend every horizon of TD error, separately within each worker's own history.

    The recursion is what forces the per-worker grouping: an advantage borrows
    from the step that followed it, and on a pooled rollout the next *row* is a
    different car. Transitions carry the worker they came from, so the columns
    are recovered from the flat order rather than from the rollout's layout.

    It also stops at every episode boundary, because the state after a crash
    explains nothing about the state before it.
    """
    transitions = rollout.transitions
    if any(
        transition.current_value is None or transition.next_value is None
        for transition in transitions
    ):
        raise ValueError(
            "GAE requires a current and bootstrap value on every rollout transition."
        )

    values = torch.tensor(
        [float(transition.current_value or 0.0) for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    # A true termination has no future to bootstrap from; a truncation does,
    # which is the whole reason the two flags are kept apart.
    bootstrap_values = torch.tensor(
        [
            0.0 if transition.terminated else float(transition.next_value or 0.0)
            for transition in transitions
        ],
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    temporal_difference_errors = (
        rewards + discount * bootstrap_values - values
    ).detach()

    columns: dict[int, list[int]] = defaultdict(list)
    for position, transition in enumerate(transitions):
        columns[transition.environment_index].append(position)

    raw_advantages = torch.zeros_like(temporal_difference_errors)
    for positions in columns.values():
        next_advantage = torch.zeros((), dtype=torch.float32, device=device)
        for index in range(len(positions) - 1, -1, -1):
            position = positions[index]
            transition = transitions[position]
            recursion_ends = (
                index == len(positions) - 1
                or transition.ends_episode
                or transitions[positions[index + 1]].episode_identity
                != transition.episode_identity
            )
            if recursion_ends:
                next_advantage = temporal_difference_errors[position]
            else:
                next_advantage = (
                    temporal_difference_errors[position]
                    + discount * gae_lambda * next_advantage
                )
            raw_advantages[position] = next_advantage

    return GAETargets(
        temporal_difference_errors=temporal_difference_errors,
        raw_advantages=raw_advantages.detach(),
        value_targets=(raw_advantages + values).detach(),
    )
