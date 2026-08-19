"""Trainable parameter counts, reported per model role.

Kept apart from the networks themselves because it is about describing a
trained agent for the run log, not about computing anything the agent needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .actor import ActorNetwork
from .critic import CriticNetwork


@dataclass(frozen=True, slots=True)
class TrainableParameterCounts:
    """
    Report actor, critic, and combined trainable parameter counts.

    Fields:
        * actor: Trainable policy parameters, including learned dispersion.
        * critic: Trainable value-network parameters, when an agent has a critic.
    """

    actor: int
    critic: int | None

    @property
    def total(self) -> int:
        """
        Return all trainable parameters owned by the agent.
        """
        return self.actor + (self.critic or 0)


def trainable_parameter_count(model: nn.Module) -> int:
    """
    Return the number of parameters optimized for one model role.
    """
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def agent_parameter_counts(
    actor: ActorNetwork, critic: CriticNetwork | None = None
) -> TrainableParameterCounts:
    """
    Return role-separated counts for an actor-only or actor-critic agent.
    """
    return TrainableParameterCounts(
        actor=trainable_parameter_count(actor),
        critic=trainable_parameter_count(critic) if critic is not None else None,
    )
