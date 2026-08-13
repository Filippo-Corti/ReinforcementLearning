"""Training loop for A2C with GAE over persistent parallel racing workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.types import AgentUpdateInput, CollectionMode

from ..buffers import TrainingTransition, VectorRolloutBuffer
from .base import TrainingEngine, TrainingRunState, transition_to_dict


class A2CTrainingEngine(TrainingEngine):
    """
    Fill a fixed rollout, then take exactly one gradient step from it.

    A2C bootstraps rather than waiting for an episode to finish, so it collects a
    fixed number of transitions that may begin and end mid-lap. The rollout is
    kept in `(time, worker)` shape because the advantage estimate recurses
    backwards down each worker's own column, and a column is one car's history.

    The whole rollout becomes a single update. That is the difference from PPO,
    which reuses the same transitions over several epochs of minibatches: A2C
    looks at each transition once, so nothing it collects is ever off-policy and
    no importance ratio is needed.

    Fields:
        * rollout_buffer: Fixed-size store of transitions in time-by-worker shape.
    """

    def _setup(self) -> None:
        """
        Create the fixed rollout this engine fills before every update.
        """
        self.rollout_buffer = VectorRolloutBuffer(
            self.agent.collection_size, self.worker_count
        )

    def train(
        self, interaction_budget: int, *, finalize: bool = True
    ) -> TrainingRunState:
        """
        Collect and optimize until the exact requested interaction budget is reached.
        """
        if interaction_budget < self.training_interactions:
            raise ValueError("Training budget cannot be below consumed interactions.")
        while self.training_interactions < interaction_budget:
            # Three separate limits decide how far this step may go: the budget,
            # the room left in the rollout, and the distance to the next
            # evaluation. Stopping at the nearest keeps every boundary exact.
            rows = interaction_budget - self.training_interactions
            rows = min(rows, self.rollout_buffer.remaining_capacity)
            if self.schedule.interval is not None:
                rows = min(
                    rows,
                    self.schedule.interval
                    - (self.training_interactions % self.schedule.interval),
                )
            active_indices = np.flatnonzero(~self.parked)[:rows]
            active = np.zeros(self.worker_count, dtype=np.bool_)
            active[active_indices] = True

            self._collect_step(active)
            self._update_if_rollout_ready(final=False)
            self.evaluate_if_due()
        self._update_if_rollout_ready(final=finalize)
        return self.state()

    def _collect_step(self, active: np.ndarray) -> None:
        """
        Advance one step and append it to the rollout, resetting finished workers.
        """
        with self.timer.collecting():
            step = self.collector.step(
                active,
                training_interactions=self.training_interactions,
                evaluation_interactions=self.schedule.evaluation_interactions,
            )
            self.training_interactions += step.interactions
            self.rollout_buffer.append_step(step.transitions)
            if step.finished:
                # A finished episode does not end the rollout: collection
                # continues on a fresh episode until the rollout is full.
                reset_mask = np.zeros(self.worker_count, dtype=np.bool_)
                reset_mask[list(step.finished)] = True
                self.collector.reset_workers(reset_mask)

    def _update_if_rollout_ready(self, *, final: bool) -> None:
        """
        Take the single actor-critic step once the rollout is full.
        """
        filled = self.rollout_buffer.transition_count
        if filled != self.agent.collection_size and not (final and filled):
            return
        update_input = AgentUpdateInput(
            mode=CollectionMode.FIXED_ROLLOUT,
            rollout=self.rollout_buffer.finalize(),
        )
        with self.timer.optimizing() as elapsed:
            output = self.agent.update(update_input)
        self.record_update(output, elapsed.seconds)

    def collector_state(self) -> dict[str, Any]:
        """
        Return the partially filled rollout for a checkpoint.
        """
        return {
            "mode": CollectionMode.FIXED_ROLLOUT.value,
            "steps": [
                [None if row is None else transition_to_dict(row) for row in step]
                for step in self.rollout_buffer.transition_steps
            ],
            "previous": [
                None if row is None else transition_to_dict(row)
                for row in self.rollout_buffer.previous_transitions
            ],
        }

    def restore_collector(self, state: dict[str, Any]) -> None:
        """
        Restore a partially filled rollout, so resume continues the same one.
        """
        if CollectionMode(state["mode"]) is not CollectionMode.FIXED_ROLLOUT:
            raise ValueError("checkpoint collection mode does not match the agent.")
        self.rollout_buffer.restore(
            [
                [None if row is None else TrainingTransition(**row) for row in step]
                for step in state["steps"]
            ],
            [
                None if row is None else TrainingTransition(**row)
                for row in state["previous"]
            ],
        )
