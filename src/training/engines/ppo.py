"""Training loop for clipped PPO over persistent parallel racing workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.types import FixedRolloutInput

from ..buffers import TrainingTransition
from ..multienvs import VectorRollout
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

    No worker is ever parked here, unlike REINFORCE. A finished episode is
    replaced by a fresh one and collection carries straight on, because a
    rollout is allowed to span an episode boundary.

    Fields:
        * rollout: Fixed-size store of transitions in time-by-worker shape.
    """

    def _setup(self) -> None:
        """
        Create the fixed rollout this engine fills before every update.
        """
        self.rollout = VectorRollout(self.agent.collection_size, self.worker_count)

    def interaction_allowance(self, interaction_budget: int) -> int:
        """
        Narrow the shared limits with the room left in the rollout.

        The budget and the next evaluation checkpoint are limits every engine
        shares, so the base class owns them. The rollout capacity is this
        algorithm's own limit and belongs here, and the nearest of the three
        still wins.
        """
        return min(
            super().interaction_allowance(interaction_budget),
            self.rollout.remaining_capacity,
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

        # As long as we have budget to spend:
        while self.training_interactions < interaction_budget:
            # Determine how many interactions we can run (before hitting
            # budget/rollout/evaluation limits), and, consequently, how many
            # workers to activate
            allowance = self.interaction_allowance(interaction_budget)
            active_mask = self.active_worker_mask(allowance)

            # Advance the environments one step each and collect the transitions
            self._collect_step(active_mask)
            self.progress.advance(self.training_interactions)

            # Check if it's time to update the policy or to run evaluation
            self.try_update(final=False)
            self.try_evaluate()

        # Run the final update (always possible for PPO)
        self.try_update(final=finalize)
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
            self.rollout.append_step(step.transitions)
            if step.finished:
                # A finished episode does not end the rollout: PPO keeps
                # collecting on a fresh episode until the rollout is full.
                reset_mask = np.zeros(self.worker_count, dtype=np.bool_)
                reset_mask[list(step.finished)] = True
                self.envs_manager.reset_workers(reset_mask)

    def try_update(self, *, final: bool) -> None:
        """
        Run the clipped update once the rollout is full, or on a final remainder.

        Unlike a Monte Carlo batch, a short rollout is still usable: every
        transition in it already carries its own bootstrap value and the
        behaviour log probability its clipping needs, so a partly filled buffer
        is a smaller but valid estimate rather than an incomplete one. `final`
        therefore permits learning from the remainder at the end of a training
        call instead of discarding it.
        """
        filled = self.rollout.transition_count
        if filled != self.agent.collection_size and not (final and filled):
            return
        update_input = FixedRolloutInput(rollout=self.rollout)
        with self.timer.optimizing() as elapsed:
            output = self.agent.update(update_input)
        self.record_update(output, elapsed.seconds)
        # The rollout is one object across its whole life, so the update does
        # not consume it: emptying it here is what starts the next one.
        self.rollout.clear()

    def _collection_payload(self) -> dict[str, Any]:
        """
        Return the partially filled rollout for a checkpoint.
        """
        return {
            "steps": [
                [
                    None if transition is None else transition_to_dict(transition)
                    for transition in step
                ]
                for step in self.rollout.transition_steps
            ],
            "previous": [
                None if transition is None else transition_to_dict(transition)
                for transition in self.rollout.previous_transitions
            ],
        }

    def _restore_collection_payload(self, state: dict[str, Any]) -> None:
        """
        Restore a partially filled rollout, so resume continues the same one.
        """
        self.rollout.restore(
            [
                tuple(
                    None if transition is None else TrainingTransition(**transition)
                    for transition in step
                )
                for step in state["steps"]
            ],
            tuple(
                None if transition is None else TrainingTransition(**transition)
                for transition in state["previous"]
            ),
        )
