"""Training loop for clipped PPO over persistent parallel racing workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.types import AgentUpdateInput, CollectionMode

from ..buffers import TrainingTransition, VectorRolloutBuffer
from .base import TrainingEngine, TrainingRunState, transition_to_dict


class PPOTrainingEngine(TrainingEngine):
    """
    Fill a fixed rollout, then reuse it for several clipped epochs.

    PPO collects a fixed number of transitions rather than complete episodes, so
    a rollout may begin and end mid-lap and may span several episodes on the same
    worker. It is kept in `(time, worker)` shape because the advantage estimate
    recurses backwards down each worker's own column: mixing workers there would
    let one car's future explain another car's present.

    Collection also stores the log probability of each action *under the policy
    that chose it*. That is what makes the update off-policy-correctable: every
    later epoch reuses these transitions while the policy has already moved, and
    the importance ratio against this stored value is what the clipping bounds.

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
        self.progress.start(self.training_interactions, interaction_budget)
        while self.training_interactions < interaction_budget:
            # Three separate limits decide how far this step may go: the budget,
            # the room left in the rollout, and the distance to the next
            # evaluation. Stopping at the nearest keeps every boundary exact.
            rows = interaction_budget - self.training_interactions
            rows = min(rows, self.rollout_buffer.remaining_capacity)
            if self.evaluation_scheduler.interval is not None:
                rows = min(
                    rows,
                    self.evaluation_scheduler.interval
                    - (self.training_interactions % self.evaluation_scheduler.interval),
                )
            active_indices = np.flatnonzero(~self.parked_mask)[:rows]
            active_mask = np.zeros(self.worker_count, dtype=np.bool_)
            active_mask[active_indices] = True

            self._collect_step(active_mask)
            self.progress.advance(self.training_interactions)
            self._update_if_rollout_ready(final=False)
            self.evaluate()
        self._update_if_rollout_ready(final=finalize)
        self.progress.close()
        return self.state()

    def _collect_step(self, active_mask: np.ndarray) -> None:
        """
        Advance one step and append it to the rollout, resetting finished workers.
        """
        with self.timer.collecting():
            step = self.envs_manager.step(
                active_mask,
                training_interactions=self.training_interactions,
                evaluation_interactions=self.evaluation_scheduler.evaluation_interactions,
            )
            self.training_interactions += step.interactions
            self.rollout_buffer.append_step(step.transitions)
            if step.finished:
                # A finished episode does not end the rollout: PPO keeps
                # collecting on a fresh episode until the rollout is full.
                reset_mask = np.zeros(self.worker_count, dtype=np.bool_)
                reset_mask[list(step.finished)] = True
                self.envs_manager.reset_workers(reset_mask)

    def _update_if_rollout_ready(self, *, final: bool) -> None:
        """
        Run the clipped update once the rollout is full, or on a final remainder.
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
