"""Training loop for REINFORCE over persistent parallel racing workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.types import AgentUpdateInput, CollectionMode

from ..buffers import OnPolicyRollout, TrainingTransition
from .base import TrainingEngine, TrainingRunState, transition_to_dict


class ReinforceTrainingEngine(TrainingEngine):
    """
    Collect whole episodes, then take one step from a batch of them.

    REINFORCE has no critic, so its target is the actual return from each state
    to the end of the episode. Nothing can be learned from a half-finished
    episode: the return is not known until the car crashes or crosses the line.
    Collection therefore runs until an episode ends rather than until a fixed
    number of transitions have been gathered.

    A worker that finishes is *parked* instead of restarted. Restarting it would
    begin an episode under the current policy that would still be running when
    the policy changed, and the batch would then mix trajectories from two
    different policies — which the estimator this implements does not allow.

    A batch needing more trajectories than there are workers is filled over
    several waves. No optimizer step happens between waves, so one batch still
    holds trajectories from a single policy.

    Fields:
        * active_trajectories: Transitions gathered so far, per worker.
        * batch: Completed trajectories waiting to become one update.
    """

    def _setup(self) -> None:
        """
        Open the per-worker trajectories and park the workers a wave excludes.
        """
        self.active_trajectories: list[list[TrainingTransition]] = [
            [] for _ in range(self.worker_count)
        ]
        self.batch: list[OnPolicyRollout] = []
        self.parked = ~self._wave_mask()

    def train(
        self, interaction_budget: int, *, finalize: bool = True
    ) -> TrainingRunState:
        """
        Collect and optimize until the exact requested interaction budget is reached.
        """
        if interaction_budget < self.training_interactions:
            raise ValueError("Training budget cannot be below consumed interactions.")
        while self.training_interactions < interaction_budget:
            rows = interaction_budget - self.training_interactions
            if self.schedule.interval is not None:
                rows = min(
                    rows,
                    self.schedule.interval
                    - (self.training_interactions % self.schedule.interval),
                )
            active_indices = np.flatnonzero(~self.parked)[:rows]
            if active_indices.size == 0:
                # Every worker is parked: either the batch is complete and can
                # be learned from, or the next wave has to be released.
                self._update_if_batch_complete(final=False)
                if self.parked.all():
                    self._start_wave()
                continue
            active = np.zeros(self.worker_count, dtype=np.bool_)
            active[active_indices] = True

            self._collect_step(active)
            self._update_if_batch_complete(final=False)
            self.evaluate_if_due()
        self._update_if_batch_complete(final=finalize)
        return self.state()

    def _collect_step(self, active: np.ndarray) -> None:
        """
        Advance one step, extending each worker's trajectory and parking finishers.
        """
        with self.timer.collecting():
            step = self.collector.step(
                active,
                training_interactions=self.training_interactions,
                evaluation_interactions=self.schedule.evaluation_interactions,
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
        self.batch.append(
            OnPolicyRollout(tuple(self.active_trajectories[worker_index]))
        )
        self.active_trajectories[worker_index] = []
        self.parked[worker_index] = True

    def _wave_mask(self) -> np.ndarray:
        """
        Select only the workers the current batch still has room for.

        The worker count is an execution choice shared with A2C and PPO, while
        the batch size belongs to REINFORCE, so the two need not agree.
        """
        wave_size = min(self.agent.collection_size - len(self.batch), self.worker_count)
        wave = np.zeros(self.worker_count, dtype=np.bool_)
        wave[:wave_size] = True
        return wave

    def _start_wave(self) -> None:
        """
        Reset and unpark the workers that collect the next group of trajectories.
        """
        wave = self._wave_mask()
        self.parked = ~wave
        self.collector.reset_workers(wave)

    def _update_if_batch_complete(self, *, final: bool) -> None:
        """
        Take the policy-gradient step once the batch holds its full episode count.
        """
        del final  # A partial batch is never learned from: its returns are complete
        # but its size is not, and the batch size is what the estimator averages over.
        if len(self.batch) != self.agent.collection_size:
            return
        update_input = AgentUpdateInput(
            mode=CollectionMode.COMPLETE_EPISODES,
            episodes=tuple(self.batch),
        )
        with self.timer.optimizing() as elapsed:
            output = self.agent.update(update_input)
        self.record_update(output, elapsed.seconds)
        self.batch = []
        self._start_wave()

    def collector_state(self) -> dict[str, Any]:
        """
        Return the partial trajectories and the completed batch for a checkpoint.
        """
        return {
            "mode": CollectionMode.COMPLETE_EPISODES.value,
            "active": [
                [transition_to_dict(row) for row in episode]
                for episode in self.active_trajectories
            ],
            "completed": [
                [transition_to_dict(row) for row in episode.transitions]
                for episode in self.batch
            ],
        }

    def restore_collector(self, state: dict[str, Any]) -> None:
        """
        Restore partial trajectories, so resume continues the episodes in flight.
        """
        if CollectionMode(state["mode"]) is not CollectionMode.COMPLETE_EPISODES:
            raise ValueError("checkpoint collection mode does not match the agent.")
        self.active_trajectories = [
            [TrainingTransition(**row) for row in episode]
            for episode in state["active"]
        ]
        self.batch = [
            OnPolicyRollout(tuple(TrainingTransition(**row) for row in episode))
            for episode in state["completed"]
        ]
