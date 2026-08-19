"""Training loop for REINFORCE over persistent parallel racing workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.types import CompleteEpisodesInput

from ..buffers import TrainingTransition, Trajectory
from .base import TrainingEngine, TrainingRunState, transition_to_dict


class ReinforceTrainingEngine(TrainingEngine):
    """
    Collect whole episodes, then take one step from a batch of them.

    REINFORCE has no critic, so its target is the actual return from each state
    to the end of the episode. Nothing can be learned from a half-finished
    episode: the return is not known until the car crashes or crosses the line.
    Collection therefore runs until episodes end rather than until a fixed
    number of transitions have been gathered.

    Spreading collection over parallel workers introduces two aspects:
        * A worker is **parked** when the episode it was racing has just ended. It
        stops stepping and waits, holding its finished trajectory, instead of
        immediately starting another one.
        Restarting it at once would open an episode under the current policy
        that would still be running after the next optimizer step, so
        the batch would end up mixing trajectories from two different policies.

        * A **wave** is one group of workers released to race at the same time, sized
        to however many trajectories the batch still needs. Eight workers and a
        batch of eight is a single wave. Eight workers and a batch of twenty is
        three waves of eight, eight and four. No optimizer step happens between the
        waves of one batch, so every trajectory in it still comes from a single
        policy; the next batch's first wave is the one that starts under the updated
        policy.

    Fields:
        * active_trajectories: Transitions gathered so far, per worker.
        * batch: Completed trajectories waiting to become one update.
    """

    def _setup(self) -> None:
        """
        Open the per-worker trajectories and park the workers a wave excludes.
        """
        self.active_trajectories: list[Trajectory] = [
            Trajectory() for _ in range(self.worker_count)
        ]
        self.batch: list[Trajectory] = []
        self.parked_mask = ~self._build_wave_mask()

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

            # Determine how many interactions we can run (before hitting budget/batch/evaluation limits),
            # and, consequently, how many workers to activate
            allowance = self.interaction_allowance(interaction_budget)
            active_mask = self.active_worker_mask(allowance)

            if not active_mask.any():
                # If there are no active workers, we should try to update the policy.
                # If the batch is full, we update and start a new wave.
                # if the batch is not full, we start a new wave and continue collecting.
                self.try_update(final=False)
                self._start_wave()
                continue

            # Advance the environments one step each and collect the transitions
            self._collect_step(active_mask)
            self.progress.advance(self.training_interactions)

            # Check if it's time to run evaluation
            self.try_evaluate()

        # Run the final update, if possible
        self.try_update(final=finalize)
        self.progress.close()
        return self.state()

    def _collect_step(self, active_mask: np.ndarray) -> None:
        """
        Advance one step, extending each worker's trajectory and parking finishers.
        """
        with self.timer.collecting():
            step = self.envs_manager.step(
                active_mask,
                training_interactions=self.training_interactions,
                evaluation_interactions=self.evaluation_scheduler.evaluation_interactions,
            )
            self.training_interactions += step.interactions
            for worker_index, transition in enumerate(step.transitions):
                if transition is not None:
                    self.active_trajectories[worker_index].append(transition)
            for worker_index in step.finished:
                self._close_trajectory(worker_index)

    def _close_trajectory(self, worker_index: int) -> None:
        """
        Move a finished episode into the batch and park the worker that ran it.
        """
        self.batch.append(self.active_trajectories[worker_index])
        self.active_trajectories[worker_index] = Trajectory()
        self.parked_mask[worker_index] = True

    def _build_wave_mask(self) -> np.ndarray:
        """
        Select only the workers the current batch still has room for.

        The worker count is an execution choice shared with A2C and PPO, while
        the batch size belongs to REINFORCE, so the two need not agree.
        """
        wave_size = min(self.agent.collection_size - len(self.batch), self.worker_count)
        wave_mask = np.zeros(self.worker_count, dtype=np.bool_)
        wave_mask[:wave_size] = True
        return wave_mask

    def _start_wave(self) -> None:
        """
        Reset and unpark the workers that collect the next group of trajectories.
        """
        wave_mask = self._build_wave_mask()
        self.parked_mask = ~wave_mask
        self.envs_manager.reset_workers(wave_mask)

    def try_update(self, *, final: bool) -> None:
        """
        Take the policy-gradient step once the batch holds its full episode count.

        A partial batch is never learned from, `final` or not: its returns are
        complete but its size is not, and the batch size is the denominator the
        estimator averages over. Discarding the remainder at the end of a
        training call is the honest option, since resuming continues the same
        batch from the checkpoint anyway.
        """
        del final
        if len(self.batch) != self.agent.collection_size:
            return
        update_input = CompleteEpisodesInput(episodes=tuple(self.batch))
        with self.timer.optimizing() as elapsed:
            output = self.agent.update(update_input)
        self.record_update(output, elapsed.seconds)
        self.batch = []

    def _collection_payload(self) -> dict[str, Any]:
        """
        Return the partial trajectories and the completed batch for a checkpoint.
        """
        return {
            "active": [
                [transition_to_dict(transition) for transition in episode]
                for episode in self.active_trajectories
            ],
            "completed": [
                [transition_to_dict(transition) for transition in episode.transitions]
                for episode in self.batch
            ],
        }

    def _restore_collection_payload(self, state: dict[str, Any]) -> None:
        """
        Restore partial trajectories, so resume continues the episodes in flight.
        """
        self.active_trajectories = [
            Trajectory(TrainingTransition(**transition) for transition in episode)
            for episode in state["active"]
        ]
        self.batch = [
            Trajectory(TrainingTransition(**transition) for transition in episode)
            for episode in state["completed"]
        ]
