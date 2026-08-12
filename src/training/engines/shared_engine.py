"""Shared parallel collection, update, evaluation, timing, and resume lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from agents.types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedActionBatch,
    CollectionMode,
    OnPolicyAgent,
)
from configs import ExecutionConfig
from envs.racing import RacingEnv
from recording.records import (
    DeterministicEvaluationRecord,
    EpisodeOutcome,
    EpisodeRecord,
    MetricScope,
    ObservationNormalizerStateRecord,
    RunCategory,
    ScalarSummaryRecord,
    TimingRecord,
)

from ..buffers import (
    OnPolicyRollout,
    TrainingTransition,
    VectorRolloutBuffer,
)
from ..checkpoints import load_checkpoint, save_checkpoint
from ..evaluation import circuit_geometry_summary, evaluate_deterministic
from ..normalization import RunningObservationNormalizer
from ..vector_environment import (
    PersistentRacingVectorEnv,
    VectorRacingState,
    vector_info,
    vector_worker_info,
)


@dataclass(slots=True)
class TrainingCounters:
    """
    Keep independent training, evaluation, update, and episode counters.

    Fields:
        * training_interactions: Environment steps eligible for the learning budget.
        * evaluation_interactions: Isolated deterministic evaluation steps.
        * finished_episodes: Training episodes that reached an environment boundary.
        * optimizer_updates: Completed calls to the agent update operation.
        * next_episode_identity: Identity assigned at the next environment reset.
        * next_evaluation_identity: Identity assigned at the next scheduled evaluation.
    """

    training_interactions: int = 0
    evaluation_interactions: int = 0
    finished_episodes: int = 0
    optimizer_updates: int = 0
    next_episode_identity: int = 0
    next_evaluation_identity: int = 0


@dataclass(slots=True)
class _ActiveEpisode:
    """
    Accumulate semantic metrics for one worker's active training episode.

    Fields:
        * identity: Stable episode identity used by rollout records.
        * circuit_identity: Stable logical identity of the selected track.
        * step_index: Zero-based action position for the next transition.
        * undiscounted_return: Reward sum accumulated through the prior action.
        * maximum_progress: Greatest normalized progress observed so far.
        * speeds: Pre-action speed samples.
        * throttles: Applied throttle/brake samples.
        * absolute_steering: Absolute steering samples.
    """

    identity: int
    circuit_identity: str
    step_index: int = 0
    undiscounted_return: float = 0.0
    maximum_progress: float = 0.0
    speeds: list[float] = field(default_factory=list)
    throttles: list[float] = field(default_factory=list)
    absolute_steering: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TrainingRunState:
    """
    Return a read-only snapshot of engine counters and accumulated timing.

    Fields:
        * counters: Independent interaction, episode, update, and evaluation counters.
        * timing: Non-overlapping component timings accumulated by the engine.
    """

    counters: TrainingCounters
    timing: TimingRecord


@dataclass(frozen=True, slots=True)
class TrainingUpdate:
    """
    Attach one agent update result to its training boundary and exclusive duration.

    Fields:
        * update_index: Zero-based update identity within the run.
        * training_interactions: Training steps consumed before this update.
        * output: Algorithm-owned scalar diagnostics.
        * optimization_duration: Exclusive duration of the agent update call.
    """

    update_index: int
    training_interactions: int
    output: AgentUpdateOutput
    optimization_duration: float


class OnPolicyTrainingEngine:
    """
    Run one on-policy agent through persistent parallel racing workers.

    Neural inference is batched in the parent process. Environment dynamics run
    in separately spawned CPU processes that persist across rollouts, updates,
    evaluations, checkpoints, and repeated calls to `train`. The engine keeps
    complete REINFORCE trajectories separate and fixed actor-critic rollouts in
    time-by-worker form.

    Fields:
        * agent: Owner of policy/value models, optimizers, and sampling streams.
        * environment: Unstepped prototype used for fixed configuration and geometry.
        * environments: Persistent process-based training environment pool.
        * normalizer: Training-updated and evaluation-frozen observation normalizer.
        * execution_config: Device, worker, threading, and deterministic settings.
        * run_category: Run namespace carried by emitted metric summaries.
        * counters: Independent training, evaluation, episode, and update counts.
    """

    STATE_VERSION = 2

    def __init__(
        self,
        agent: OnPolicyAgent,
        environment: RacingEnv,
        normalizer: RunningObservationNormalizer,
        *,
        run_category: RunCategory,
        evaluation_environment_factory: Callable[[], RacingEnv] | None = None,
        evaluation_interval: int | None = None,
        environment_reset_generator: np.random.Generator | None = None,
        environment_reset_generators: Sequence[np.random.Generator] | None = None,
        track_selection_generators: Sequence[np.random.Generator] | None = None,
        execution_config: ExecutionConfig | None = None,
        evaluation_seed: int = 0,
        root_identity: int | None = None,
        circuit_identity: str | None = None,
        circuit_split: str | None = None,
        near_saturated_steering_threshold: float | None = None,
    ) -> None:
        """
        Spawn workers once and initialize one active episode in every process.
        """
        if agent.collection_size <= 0:
            raise ValueError("An on-policy agent collection size must be positive.")
        if evaluation_interval is not None and evaluation_interval <= 0:
            raise ValueError("Evaluation interval must be positive when enabled.")
        if near_saturated_steering_threshold is not None and not (
            0.0 < near_saturated_steering_threshold <= 1.0
        ):
            raise ValueError("Near-saturated steering threshold must be in (0, 1].")
        self.execution_config = execution_config or ExecutionConfig(
            device="cpu", environment_workers=1
        )
        worker_count = self.execution_config.environment_workers
        if worker_count <= 0:
            raise ValueError("Training requires a positive environment-worker count.")
        sampling_generators = getattr(agent, "sampling_generators", None)
        if sampling_generators is not None and len(sampling_generators) != worker_count:
            raise ValueError("One policy-sampling stream is required per worker.")

        reset_generators = _resolve_generators(
            environment_reset_generators,
            environment_reset_generator,
            worker_count,
            "reset",
        )
        selection_generators = _resolve_generators(
            track_selection_generators,
            None,
            worker_count,
            "track-selection",
        )
        self.agent = agent
        self.environment = environment
        self.normalizer = normalizer
        self.run_category = run_category
        self.evaluation_environment_factory = evaluation_environment_factory
        self.evaluation_interval = evaluation_interval
        self.environment_reset_generator = reset_generators[0]
        self.evaluation_seed = evaluation_seed
        self.root_identity = root_identity
        self.circuit_identity = circuit_identity or str(
            environment.track.generation.seed
        )
        self.circuit_split = circuit_split
        self.near_saturated_steering_threshold = near_saturated_steering_threshold
        self.counters = TrainingCounters()
        self._collection_seconds = 0.0
        self._optimization_seconds = 0.0
        self._evaluation_seconds = 0.0
        self._persistence_seconds = 0.0
        self._started_at = perf_counter()
        self.episode_records: list[EpisodeRecord] = []
        self.evaluations: list[DeterministicEvaluationRecord] = []
        self.updates: list[TrainingUpdate] = []
        self.environments = PersistentRacingVectorEnv(
            (environment.track_with_geometry,),
            environment.config,
            self.execution_config,
            reset_generators,
            selection_generators,
        )
        self._current_observations, reset_infos = self.environments.reset()
        self._active_episodes: list[_ActiveEpisode] = []
        for environment_index in range(worker_count):
            self._active_episodes.append(
                self._new_episode(
                    str(vector_info(reset_infos, "circuit_identity", environment_index))
                )
            )
        self._reinforce_active: list[list[TrainingTransition]] | None = None
        self._reinforce_batch: list[OnPolicyRollout] | None = None
        self._parked = np.zeros(worker_count, dtype=np.bool_)
        self._rollout_buffer: VectorRolloutBuffer | None = None
        if agent.collection_mode is CollectionMode.COMPLETE_EPISODES:
            self._reinforce_active = [[] for _ in range(worker_count)]
            self._reinforce_batch = []
            self._parked = ~self._reinforce_wave_mask()
        else:
            self._rollout_buffer = VectorRolloutBuffer(
                agent.collection_size, worker_count
            )

    def train(
        self, interaction_budget: int, *, finalize: bool = True
    ) -> TrainingRunState:
        """
        Collect and optimize until the exact requested interaction budget is reached.
        """
        if interaction_budget < self.counters.training_interactions:
            raise ValueError("Training budget cannot be below consumed interactions.")
        while self.counters.training_interactions < interaction_budget:
            available = ~self._parked
            maximum_rows = interaction_budget - self.counters.training_interactions
            if self._rollout_buffer is not None:
                maximum_rows = min(
                    maximum_rows, self._rollout_buffer.remaining_capacity
                )
            if self.evaluation_interval is not None:
                distance = self.evaluation_interval - (
                    self.counters.training_interactions % self.evaluation_interval
                )
                maximum_rows = min(maximum_rows, distance)
            active_indices = np.flatnonzero(available)[:maximum_rows]
            if active_indices.size == 0:
                self._update_ready_collection(final=False)
                if self._reinforce_batch is not None and self._parked.all():
                    self._start_reinforce_wave()
                continue
            active = np.zeros(self.execution_config.environment_workers, dtype=np.bool_)
            active[active_indices] = True
            self._collect_step(active)
            self._update_ready_collection(final=False)
            self._evaluate_if_due()
        self._update_ready_collection(final=finalize)
        return self.state()

    def state(self) -> TrainingRunState:
        """
        Return copied counters and accumulated non-overlapping timing categories.
        """
        return TrainingRunState(
            counters=TrainingCounters(**asdict(self.counters)),
            timing=self.timing(),
        )

    def timing(self) -> TimingRecord:
        """
        Return mutually exclusive component times and elapsed end-to-end time.
        """
        return TimingRecord(
            run_category=self.run_category,
            scope=MetricScope.TRAINING,
            collection=self._collection_seconds,
            optimization=self._optimization_seconds,
            evaluation=self._evaluation_seconds,
            persistence=self._persistence_seconds,
            end_to_end=perf_counter() - self._started_at,
        )

    def save(self, path: str | Path) -> None:
        """
        Atomically save every worker and collector state for exact resume.
        """
        started = perf_counter()
        save_checkpoint(path, self._state_dict())
        self._persistence_seconds += perf_counter() - started

    def restore(self, path: str | Path, *, map_location: str = "cpu") -> None:
        """
        Restore a checkpoint onto equivalent agent and worker instances.
        """
        started = perf_counter()
        self._restore_state_dict(load_checkpoint(path, map_location=map_location))
        self._persistence_seconds += perf_counter() - started

    def close(self) -> None:
        """
        Close the persistent worker processes and prototype environment.
        """
        self.environments.close()
        self.environment.close()

    def _collect_step(self, active: np.ndarray) -> None:
        started = perf_counter()
        normalized = self.normalizer.update_and_normalize_batch(
            self._current_observations, active
        )
        active_indices = np.flatnonzero(active)
        decisions = self._collect_actions(normalized, active_indices)
        actions = np.zeros(
            (self.execution_config.environment_workers, 2), dtype=np.float32
        )
        actions[active_indices] = decisions.env_actions
        next_observations, rewards, terminated, truncated, infos = (
            self.environments.step(actions, active)
        )
        next_normalized = np.stack(
            [
                self.normalizer.normalize(next_observations[index])
                for index in active_indices
            ]
        )
        next_values = self._bootstrap_values(next_normalized)
        transition_step: list[TrainingTransition | None] = [
            None
        ] * self.execution_config.environment_workers
        reset_mask = np.zeros(self.execution_config.environment_workers, dtype=np.bool_)

        for decision_index, environment_index_value in enumerate(active_indices):
            environment_index = int(environment_index_value)
            info = vector_worker_info(infos, environment_index)
            is_terminated = bool(terminated[environment_index])
            is_truncated = bool(truncated[environment_index])
            active_episode = self._active_episodes[environment_index]
            transition = TrainingTransition(
                normalized_observation=normalized[environment_index],
                raw_action=decisions.raw_actions[decision_index],
                env_action=decisions.env_actions[decision_index],
                reward=float(rewards[environment_index]),
                behaviour_log_probability=float(
                    decisions.behaviour_log_probabilities[decision_index]
                ),
                current_value=(
                    None
                    if decisions.current_values is None
                    else float(decisions.current_values[decision_index])
                ),
                next_value=(
                    None
                    if next_values is None
                    else (0.0 if is_terminated else float(next_values[decision_index]))
                ),
                terminated=is_terminated,
                truncated=is_truncated,
                next_normalized_observation=next_normalized[decision_index],
                episode_identity=active_episode.identity,
                episode_step_index=active_episode.step_index,
                circuit_identity=active_episode.circuit_identity,
                environment_index=environment_index,
            )
            transition_step[environment_index] = transition
            self._append_transition(environment_index, transition)
            self._record_active_step(
                environment_index,
                decisions.env_actions[decision_index],
                float(rewards[environment_index]),
                info,
            )
            self.counters.training_interactions += 1
            if is_terminated or is_truncated:
                self._finish_training_episode(
                    environment_index, is_terminated, is_truncated, info
                )
                if self.agent.collection_mode is CollectionMode.COMPLETE_EPISODES:
                    self._finish_reinforce_trajectory(environment_index)
                else:
                    reset_mask[environment_index] = True

        if self._rollout_buffer is not None:
            self._rollout_buffer.append_step(transition_step)
        self._current_observations = next_observations
        if np.any(reset_mask):
            self._reset_workers(reset_mask)
        self._collection_seconds += perf_counter() - started

    def _collect_actions(
        self, normalized: np.ndarray, active_indices: np.ndarray
    ) -> CollectedActionBatch:
        batch_method = getattr(self.agent, "collect_actions", None)
        if callable(batch_method):
            return cast(Callable[..., CollectedActionBatch], batch_method)(
                normalized[active_indices],
                environment_indices=[int(index) for index in active_indices],
            )
        rows = [
            self.agent.collect_action(normalized[index]) for index in active_indices
        ]
        return CollectedActionBatch(
            raw_actions=np.stack([row.raw_action for row in rows]),
            env_actions=np.stack([row.env_action for row in rows]),
            behaviour_log_probabilities=np.asarray(
                [row.behaviour_log_probability or 0.0 for row in rows],
                dtype=np.float32,
            ),
            current_values=(
                None
                if any(row.current_value is None for row in rows)
                else np.asarray([row.current_value for row in rows], dtype=np.float32)
            ),
        )

    def _bootstrap_values(self, next_normalized: np.ndarray) -> np.ndarray | None:
        batch_method = getattr(self.agent, "bootstrap_values", None)
        if callable(batch_method):
            return cast(Callable[[np.ndarray], np.ndarray | None], batch_method)(
                next_normalized
            )
        values = [self.agent.bootstrap_value(row) for row in next_normalized]
        if any(value is None for value in values):
            return None
        return np.asarray(values, dtype=np.float32)

    def _append_transition(
        self, environment_index: int, transition: TrainingTransition
    ) -> None:
        if self._reinforce_active is not None:
            self._reinforce_active[environment_index].append(transition)

    def _finish_reinforce_trajectory(self, environment_index: int) -> None:
        if self._reinforce_active is None or self._reinforce_batch is None:
            raise RuntimeError("REINFORCE collector was not constructed.")
        self._reinforce_batch.append(
            OnPolicyRollout(tuple(self._reinforce_active[environment_index]))
        )
        self._reinforce_active[environment_index] = []
        self._parked[environment_index] = True

    def _reinforce_wave_mask(self) -> np.ndarray:
        """
        Select only the workers the current update batch still has room for.

        The worker count is an execution choice shared with A2C and PPO, while
        the batch size belongs to REINFORCE, so a batch that needs more
        trajectories than there are workers is filled over several waves. No
        optimizer step happens between waves, so one batch still holds
        trajectories from a single policy.
        """
        if self._reinforce_batch is None:
            raise RuntimeError("REINFORCE collector was not constructed.")
        wave_size = min(
            self.agent.collection_size - len(self._reinforce_batch),
            self.execution_config.environment_workers,
        )
        wave = np.zeros(self.execution_config.environment_workers, dtype=np.bool_)
        wave[:wave_size] = True
        return wave

    def _start_reinforce_wave(self) -> None:
        """
        Reset and unpark the workers that collect the next group of trajectories.
        """
        wave = self._reinforce_wave_mask()
        self._parked = ~wave
        self._reset_workers(wave)

    def _update_ready_collection(self, *, final: bool) -> None:
        update_input: AgentUpdateInput | None = None
        if (
            self._reinforce_batch is not None
            and len(self._reinforce_batch) == self.agent.collection_size
        ):
            update_input = AgentUpdateInput(
                mode=CollectionMode.COMPLETE_EPISODES,
                episodes=tuple(self._reinforce_batch),
            )
        elif self._rollout_buffer is not None and (
            self._rollout_buffer.transition_count == self.agent.collection_size
            or (final and self._rollout_buffer.transition_count)
        ):
            update_input = AgentUpdateInput(
                mode=CollectionMode.FIXED_ROLLOUT,
                rollout=self._rollout_buffer.finalize(),
            )
        if update_input is None:
            return
        started = perf_counter()
        output = self.agent.update(update_input)
        duration = perf_counter() - started
        self._optimization_seconds += duration
        self.updates.append(
            TrainingUpdate(
                update_index=self.counters.optimizer_updates,
                training_interactions=self.counters.training_interactions,
                output=output,
                optimization_duration=duration,
            )
        )
        self.counters.optimizer_updates += 1
        if self._reinforce_batch is not None:
            self._reinforce_batch = []
            self._start_reinforce_wave()

    def _evaluate_if_due(self) -> None:
        if (
            self.evaluation_interval is None
            or self.evaluation_environment_factory is None
            or self.counters.training_interactions == 0
            or self.counters.training_interactions % self.evaluation_interval != 0
        ):
            return
        started = perf_counter()
        identity = self.counters.next_evaluation_identity
        reset_seed = int(
            np.random.SeedSequence([self.evaluation_seed, identity]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        evaluation = evaluate_deterministic(
            self.evaluation_environment_factory,
            self.agent,
            self.normalizer,
            run_category=self.run_category,
            evaluation_index=identity,
            training_interactions=self.counters.training_interactions,
            evaluation_interactions_before=self.counters.evaluation_interactions,
            reset_seed=reset_seed,
            root_identity=self.root_identity,
            circuit_identity=self.circuit_identity,
            circuit_split=self.circuit_split,
            collection_duration=self._collection_seconds,
            optimization_duration=self._optimization_seconds,
            near_saturated_steering_threshold=self.near_saturated_steering_threshold,
        )
        self._evaluation_seconds += perf_counter() - started
        self.evaluations.append(evaluation)
        self.counters.evaluation_interactions += evaluation.record.episode.interactions
        self.counters.next_evaluation_identity += 1

    def _reset_workers(self, reset_mask: np.ndarray) -> None:
        observations, infos = self.environments.reset(reset_mask)
        self._current_observations = observations
        for environment_index_value in np.flatnonzero(reset_mask):
            environment_index = int(environment_index_value)
            self._active_episodes[environment_index] = self._new_episode(
                str(vector_info(infos, "circuit_identity", environment_index))
            )

    def _new_episode(self, circuit_identity: str) -> _ActiveEpisode:
        identity = self.counters.next_episode_identity
        self.counters.next_episode_identity += 1
        return _ActiveEpisode(identity, circuit_identity)

    def _record_active_step(
        self,
        environment_index: int,
        action: np.ndarray,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        active = self._active_episodes[environment_index]
        active.speeds.append(float(self._current_observations[environment_index, 2]))
        active.throttles.append(float(action[0]))
        active.absolute_steering.append(abs(float(action[1])))
        active.step_index += 1
        active.undiscounted_return += reward
        progress = float(info["episode_progress"]) / float(info["track_length"])
        active.maximum_progress = max(active.maximum_progress, progress)

    def _finish_training_episode(
        self,
        environment_index: int,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        active = self._active_episodes[environment_index]
        outcome = _outcome(terminated, truncated, info)
        self.episode_records.append(
            EpisodeRecord(
                run_category=self.run_category,
                scope=MetricScope.TRAINING,
                episode_index=active.identity,
                outcome=outcome,
                undiscounted_return=active.undiscounted_return,
                training_target_total=None,
                interactions=active.step_index,
                simulated_time=float(info["elapsed_time"]),
                final_progress=float(info["episode_progress"])
                / float(info["track_length"]),
                maximum_progress=active.maximum_progress,
                lap_time=(
                    float(info["elapsed_time"])
                    if outcome is EpisodeOutcome.COMPLETED
                    else None
                ),
                training_interactions=self.counters.training_interactions,
                evaluation_interactions=self.counters.evaluation_interactions,
                circuit_identity=active.circuit_identity,
                root_identity=self.root_identity,
                observation_type="frenet",
                circuit_seed=int(info["track_seed"]),
                circuit_split=self.circuit_split,
                speed=_summary(active.speeds),
                throttle=_summary(active.throttles),
                absolute_steering=_summary(active.absolute_steering),
                positive_throttle_fraction=float(
                    np.mean(np.asarray(active.throttles) > 0)
                ),
                braking_fraction=float(np.mean(np.asarray(active.throttles) < 0)),
                near_saturated_steering_fraction=(
                    None
                    if self.near_saturated_steering_threshold is None
                    else float(
                        np.mean(
                            np.asarray(active.absolute_steering)
                            >= self.near_saturated_steering_threshold
                        )
                    )
                ),
                circuit_geometry=circuit_geometry_summary(self.environment),
            )
        )
        self.counters.finished_episodes += 1

    def _state_dict(self) -> dict[str, Any]:
        return {
            "engine_state_version": self.STATE_VERSION,
            "engine_configuration": self._engine_configuration(),
            "agent": self.agent.state_dict(),
            "normalizer": self.normalizer.state().to_dict(),
            "counters": asdict(self.counters),
            "current_observations": self._current_observations.tolist(),
            "active_episodes": [asdict(episode) for episode in self._active_episodes],
            "vector_environment": self.environments.state(),
            "parked": self._parked.tolist(),
            "collector": self._collector_state(),
            "history": {
                "episode_records": self.episode_records,
                "evaluations": self.evaluations,
                "updates": self.updates,
            },
            "timing": {
                "collection": self._collection_seconds,
                "optimization": self._optimization_seconds,
                "evaluation": self._evaluation_seconds,
                "persistence": self._persistence_seconds,
            },
        }

    def _restore_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("engine_state_version") != self.STATE_VERSION:
            raise ValueError(
                "checkpoint has an incompatible training-engine state version."
            )
        if _mapping(state, "engine_configuration") != self._engine_configuration():
            raise ValueError(
                "checkpoint configuration does not match this training engine."
            )
        self.agent.load_state_dict(_mapping(state, "agent"))
        normalizer_state = _mapping(state, "normalizer")
        self.normalizer.restore(
            ObservationNormalizerStateRecord(
                count=int(normalizer_state["count"]),
                sums=tuple(float(value) for value in normalizer_state["sums"]),
                squared_sums=tuple(
                    float(value) for value in normalizer_state["squared_sums"]
                ),
            )
        )
        self.counters = TrainingCounters(**_mapping(state, "counters"))
        self._current_observations = np.asarray(
            state["current_observations"], dtype=np.float32
        )
        self._active_episodes = [
            _ActiveEpisode(**episode) for episode in state["active_episodes"]
        ]
        vector_state = state["vector_environment"]
        if not isinstance(vector_state, VectorRacingState):
            raise TypeError("checkpoint vector environment state is invalid.")
        self.environments.restore(vector_state)
        self._parked = np.asarray(state["parked"], dtype=np.bool_)
        self._restore_collector(_mapping(state, "collector"))
        history = _mapping(state, "history")
        self.episode_records = _typed_list(history, "episode_records", EpisodeRecord)
        self.evaluations = _typed_list(
            history, "evaluations", DeterministicEvaluationRecord
        )
        self.updates = _typed_list(history, "updates", TrainingUpdate)
        timing = _mapping(state, "timing")
        self._collection_seconds = float(timing["collection"])
        self._optimization_seconds = float(timing["optimization"])
        self._evaluation_seconds = float(timing["evaluation"])
        self._persistence_seconds = float(timing["persistence"])

    def _engine_configuration(self) -> dict[str, Any]:
        return {
            "run_category": self.run_category.value,
            "collection_mode": self.agent.collection_mode.value,
            "collection_size": self.agent.collection_size,
            "environment_workers": self.execution_config.environment_workers,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_seed": self.evaluation_seed,
            "root_identity": self.root_identity,
            "circuit_identity": self.circuit_identity,
            "circuit_split": self.circuit_split,
            "near_saturated_steering_threshold": (
                self.near_saturated_steering_threshold
            ),
            "environment_config": self.environment.config.to_dict(),
            "track_seed": self.environment.track.generation.seed,
            "track_length": self.environment.track.track_length,
            "track_samples": int(self.environment.track.s.size),
            "normalizer_dimensions": self.normalizer.observation_dimensions,
        }

    def _collector_state(self) -> dict[str, Any]:
        if self._reinforce_active is not None:
            if self._reinforce_batch is None:
                raise RuntimeError("REINFORCE completed collector is missing.")
            return {
                "mode": CollectionMode.COMPLETE_EPISODES.value,
                "active": [
                    [_transition_to_dict(row) for row in episode]
                    for episode in self._reinforce_active
                ],
                "completed": [
                    [_transition_to_dict(row) for row in episode.transitions]
                    for episode in self._reinforce_batch
                ],
            }
        if self._rollout_buffer is None:
            raise RuntimeError("Training engine has no collection buffer.")
        return {
            "mode": CollectionMode.FIXED_ROLLOUT.value,
            "steps": [
                [None if row is None else _transition_to_dict(row) for row in step]
                for step in self._rollout_buffer.transition_steps
            ],
            "previous": [
                None if row is None else _transition_to_dict(row)
                for row in self._rollout_buffer.previous_transitions
            ],
        }

    def _restore_collector(self, state: dict[str, Any]) -> None:
        mode = CollectionMode(state["mode"])
        if mode is not self.agent.collection_mode:
            raise ValueError("checkpoint collection mode does not match the agent.")
        if mode is CollectionMode.COMPLETE_EPISODES:
            if self._reinforce_active is None or self._reinforce_batch is None:
                raise RuntimeError("REINFORCE collector was not constructed.")
            self._reinforce_active = [
                [_transition_from_dict(row) for row in episode]
                for episode in state["active"]
            ]
            self._reinforce_batch = [
                OnPolicyRollout(tuple(_transition_from_dict(row) for row in episode))
                for episode in state["completed"]
            ]
            return
        if self._rollout_buffer is None:
            raise RuntimeError("Vector rollout collector was not constructed.")
        steps = [
            [None if row is None else _transition_from_dict(row) for row in step]
            for step in state["steps"]
        ]
        previous = [
            None if row is None else _transition_from_dict(row)
            for row in state["previous"]
        ]
        self._rollout_buffer.restore(steps, previous)


def _resolve_generators(
    generators: Sequence[np.random.Generator] | None,
    single_generator: np.random.Generator | None,
    worker_count: int,
    role: str,
) -> tuple[np.random.Generator, ...]:
    if generators is not None:
        values = tuple(generators)
    elif single_generator is not None:
        values = (single_generator,)
    else:
        values = tuple(np.random.default_rng(index) for index in range(worker_count))
    if len(values) != worker_count:
        raise ValueError(f"One {role} generator is required per environment worker.")
    return values


def _summary(values: list[float]) -> ScalarSummaryRecord:
    """
    Summarize one non-empty recorded training signal with population dispersion.
    """
    array = np.asarray(values, dtype=np.float64)
    return ScalarSummaryRecord(
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        quantiles={
            "q25": float(np.quantile(array, 0.25)),
            "q50": float(np.quantile(array, 0.50)),
            "q75": float(np.quantile(array, 0.75)),
            "q90": float(np.quantile(array, 0.90)),
        },
    )


def _outcome(terminated: bool, truncated: bool, info: dict[str, Any]) -> EpisodeOutcome:
    """
    Convert an explicit racing lifecycle boundary into a metrics outcome.
    """
    if terminated and bool(info["lap_completed"]):
        return EpisodeOutcome.COMPLETED
    if terminated and bool(info["collision"]):
        return EpisodeOutcome.CRASHED
    if truncated:
        return EpisodeOutcome.TIME_LIMIT
    raise ValueError("Terminal transition lacks a supported RacingEnv outcome.")


def _transition_to_dict(row: TrainingTransition) -> dict[str, Any]:
    """
    Serialize one immutable rollout row without retaining framework objects.
    """
    return {
        "normalized_observation": row.normalized_observation.tolist(),
        "raw_action": row.raw_action.tolist(),
        "env_action": row.env_action.tolist(),
        "reward": row.reward,
        "behaviour_log_probability": row.behaviour_log_probability,
        "current_value": row.current_value,
        "next_value": row.next_value,
        "terminated": row.terminated,
        "truncated": row.truncated,
        "next_normalized_observation": row.next_normalized_observation.tolist(),
        "episode_identity": row.episode_identity,
        "episode_step_index": row.episode_step_index,
        "circuit_identity": row.circuit_identity,
        "environment_index": row.environment_index,
    }


def _transition_from_dict(data: dict[str, Any]) -> TrainingTransition:
    """
    Reconstruct one immutable rollout row from checkpoint-safe primitives.
    """
    return TrainingTransition(**data)


def _mapping(state: dict[str, Any], name: str) -> dict[str, Any]:
    """
    Read one required mapping from a checkpoint with a concise failure mode.
    """
    value = state.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint field {name!r} must be a dictionary.")
    return value


def _typed_list[T](state: dict[str, Any], name: str, item_type: type[T]) -> list[T]:
    """
    Restore one required list only when every semantic record has the expected type.
    """
    value = state.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, item_type) for item in value
    ):
        raise TypeError(f"checkpoint field {name!r} has invalid record types.")
    return value
