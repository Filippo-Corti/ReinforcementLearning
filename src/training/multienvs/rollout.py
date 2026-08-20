"""A rollout collected across parallel workers, laid out time by worker.

Kept here rather than beside `Trajectory` because the layout only means
anything when several environments are stepped together: a row is one
synchronous tick of the whole pool, and a column is one car's own history.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..buffers import TrainingTransition

# One synchronous tick across the worker pool: one cell per worker, and `None`
# where that worker was parked and therefore produced nothing.
MultiEnvTrainingTransition = tuple[TrainingTransition | None, ...]


class VectorRollout:
    """
    Accumulate a fixed number of transitions in time-by-worker steps.

    This is what A2C and PPO learn from. It is collected tick by tick and read
    back as flat rows, and it is one object for both: appending is how it grows
    and `clear` is how it starts the next one, so a partly filled rollout is
    never a different type from a full one.

    The `(time, worker)` layout exists for one reason: the advantage recursion
    runs backwards down a single worker's column. Flattening the pool together
    would let one car's future explain another car's present. Every consumer
    outside this class reads `transitions`, which is flat.

    Fields:
        * capacity: Valid transitions collected before the rollout is full.
        * environment_count: Number of persistent worker columns.
    """

    def __init__(self, capacity: int, environment_count: int) -> None:
        """
        Open an empty rollout sized to the worker pool that will fill it.
        """
        if capacity <= 0:
            raise ValueError("Rollout capacity must be positive.")
        if environment_count <= 0:
            raise ValueError("Vector rollouts require a positive environment count.")
        self.capacity = capacity
        self.environment_count = environment_count
        self._steps: list[MultiEnvTrainingTransition] = []
        self._previous: list[TrainingTransition | None] = [None] * environment_count
        self._transition_count = 0

    def append_step(self, transitions: Iterable[TrainingTransition | None]) -> None:
        """
        Append one synchronous worker tick without exceeding capacity.
        """
        step: MultiEnvTrainingTransition = tuple(transitions)
        if len(step) != self.environment_count:
            raise ValueError("A vector step must contain one cell per environment.")
        valid_count = sum(transition is not None for transition in step)
        if valid_count == 0:
            raise ValueError("A vector rollout step must contain a valid transition.")
        if self._transition_count + valid_count > self.capacity:
            raise ValueError(
                "Clear the full vector rollout before appending more transitions."
            )
        for environment_index, transition in enumerate(step):
            if transition is None:
                continue
            if transition.environment_index != environment_index:
                raise ValueError(
                    "A transition environment index must match its vector column."
                )
            transition.validate_follows(self._previous[environment_index])
            self._previous[environment_index] = transition
        self._steps.append(step)
        self._transition_count += valid_count

    def clear(self) -> None:
        """
        Drop the collected steps and start the next rollout.

        The last transition of each column is deliberately kept: the next
        rollout continues the same episodes, and that is what lets the
        continuity check span the boundary between two rollouts.
        """
        self._steps = []
        self._transition_count = 0

    @property
    def transitions(self) -> tuple[TrainingTransition, ...]:
        """
        Return the valid transitions in time-major, worker-minor order.

        This is the order every learning target and loss is expressed in, so it
        is also the order this rollout is flattened to everywhere else.
        """
        return tuple(
            transition
            for step in self._steps
            for transition in step
            if transition is not None
        )

    @property
    def transition_count(self) -> int:
        """
        Return how many valid transitions are currently collected.
        """
        return self._transition_count

    @property
    def remaining_capacity(self) -> int:
        """
        Return how many further valid transitions fit before the rollout is full.
        """
        return self.capacity - self._transition_count

    @property
    def transition_steps(self) -> tuple[MultiEnvTrainingTransition, ...]:
        """
        Return the stored ticks, for checkpointing rather than for learning.
        """
        return tuple(self._steps)

    @property
    def previous_transitions(self) -> MultiEnvTrainingTransition:
        """
        Return the last valid transition retained for each worker column.
        """
        return tuple(self._previous)

    def restore(
        self,
        transition_steps: Sequence[Iterable[TrainingTransition | None]],
        previous_transitions: Sequence[TrainingTransition | None],
    ) -> None:
        """
        Restore checkpointed ticks and each column's continuity context.
        """
        if len(previous_transitions) != self.environment_count:
            raise ValueError("Vector checkpoint column count does not match.")
        self._steps = []
        self._previous = [None] * self.environment_count
        self._transition_count = 0
        for step in transition_steps:
            self.append_step(step)
        restored_previous = tuple(previous_transitions)
        for environment_index, previous in enumerate(restored_previous):
            current = self._previous[environment_index]
            if current is None:
                continue
            # Compared by continuity key rather than by whole transition: the
            # records hold NumPy arrays, and `==` on those is elementwise.
            restored_key = None if previous is None else previous.continuity_key
            if restored_key != current.continuity_key:
                raise ValueError("Vector checkpoint continuity state is inconsistent.")
        self._previous = list(restored_previous)

    def __len__(self) -> int:
        """
        Return how many valid transitions are currently collected.
        """
        return self._transition_count
