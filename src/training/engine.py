"""Shared collection, update, evaluation, timing, and resume lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from agents.types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectionMode,
    OnPolicyAgent,
)
from envs.racing import RacingEnv, RacingEnvState
from envs.racing.lifecycle import EpisodeLifecycleState
from envs.vehicle import VehicleState
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

from .buffers import (
    FixedRolloutBuffer,
    OnPolicyRollout,
    ReinforceEpisodeBuffer,
    TrainingTransition,
)
from .checkpoints import load_checkpoint, save_checkpoint
from .evaluation import circuit_geometry_summary, evaluate_deterministic
from .normalization import RunningObservationNormalizer


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
    Accumulate semantic metrics for the training episode currently being acted on.

    Fields:
        * identity: Stable episode identity used by rollout records.
        * circuit_identity: Stable logical identity of the active track.
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
    Run one on-policy agent while preserving shared collection and resume semantics.

    The engine updates observations only when they become current training inputs,
    isolates deterministic evaluation in a fresh environment, and holds only
    semantic environment state for checkpoint resume.

    Fields:
        * agent: Owner of policy/value models, optimizers, and agent random streams.
        * environment: Training-only environment whose episode can span updates.
        * normalizer: Training-updated and evaluation-frozen observation normalizer.
        * run_category: Run namespace carried by emitted metric summaries.
        * counters: Independent training, evaluation, episode, and update counts.
    """

    STATE_VERSION = 1

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
        evaluation_seed: int = 0,
        root_identity: int | None = None,
        circuit_identity: str | None = None,
        circuit_split: str | None = None,
        near_saturated_steering_threshold: float | None = None,
    ) -> None:
        """
        Initialize a fresh training episode and the buffer implied by the agent mode.
        """
        if agent.collection_size <= 0:
            raise ValueError("An on-policy agent collection size must be positive.")
        if evaluation_interval is not None and evaluation_interval <= 0:
            raise ValueError("Evaluation interval must be positive when enabled.")
        if near_saturated_steering_threshold is not None and not (
            0.0 < near_saturated_steering_threshold <= 1.0
        ):
            raise ValueError("Near-saturated steering threshold must be in (0, 1].")
        self.agent = agent
        self.environment = environment
        self.normalizer = normalizer
        self.run_category = run_category
        self.evaluation_environment_factory = evaluation_environment_factory
        self.evaluation_interval = evaluation_interval
        self.environment_reset_generator = (
            environment_reset_generator
            if environment_reset_generator is not None
            else np.random.default_rng(0)
        )
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
        self._episode_buffer = (
            ReinforceEpisodeBuffer()
            if agent.collection_mode is CollectionMode.COMPLETE_EPISODES
            else None
        )
        self._rollout_buffer = (
            FixedRolloutBuffer(agent.collection_size)
            if agent.collection_mode is CollectionMode.FIXED_ROLLOUT
            else None
        )
        self._current_observation: np.ndarray
        self._active_episode: _ActiveEpisode
        self._reset_training_episode()

    def train(
        self, interaction_budget: int, *, finalize: bool = True
    ) -> TrainingRunState:
        """
        Collect and optimize until the exact requested training-step budget is reached.
        """
        if interaction_budget < self.counters.training_interactions:
            raise ValueError("Training budget cannot be below consumed interactions.")
        while self.counters.training_interactions < interaction_budget:
            self._collect_one()
            self._update_ready_collection(final=False)
            self._evaluate_if_due()
        self._update_ready_collection(final=finalize)
        return self.state()

    def state(self) -> TrainingRunState:
        """
        Return copied counters and accumulated non-overlapping timing categories.
        """
        counters = TrainingCounters(**asdict(self.counters))
        return TrainingRunState(counters=counters, timing=self.timing())

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
        Atomically save enough semantic state to resume within an active episode.
        """
        started = perf_counter()
        save_checkpoint(path, self._state_dict())
        self._persistence_seconds += perf_counter() - started

    def restore(self, path: str | Path, *, map_location: str = "cpu") -> None:
        """
        Restore an engine checkpoint onto equivalent agent and environment instances.
        """
        started = perf_counter()
        state = load_checkpoint(path, map_location=map_location)
        self._restore_state_dict(state)
        self._persistence_seconds += perf_counter() - started

    def _collect_one(self) -> None:
        started = perf_counter()
        normalized = self.normalizer.update_and_normalize(self._current_observation)
        decision = self.agent.collect_action(normalized)
        next_observation, reward, terminated, truncated, raw_info = (
            self.environment.step(decision.env_action)
        )
        info = cast(dict[str, Any], raw_info)
        next_normalized = self.normalizer.normalize(next_observation)
        next_value = 0.0 if terminated else self.agent.bootstrap_value(next_normalized)
        transition = TrainingTransition(
            normalized_observation=normalized,
            raw_action=decision.raw_action,
            env_action=decision.env_action,
            reward=float(reward),
            behaviour_log_probability=decision.behaviour_log_probability,
            current_value=decision.current_value,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
            next_normalized_observation=next_normalized,
            episode_identity=self._active_episode.identity,
            episode_step_index=self._active_episode.step_index,
            circuit_identity=self._active_episode.circuit_identity,
        )
        self._append_transition(transition)
        self._record_active_step(decision.env_action, reward, info)
        self.counters.training_interactions += 1
        self._collection_seconds += perf_counter() - started

        if terminated or truncated:
            self._finish_training_episode(terminated, truncated, info)
            self._reset_training_episode()
        else:
            self._current_observation = np.asarray(
                next_observation, dtype=np.float32
            ).copy()

    def _append_transition(self, transition: TrainingTransition) -> None:
        if self._episode_buffer is not None:
            self._episode_buffer.append(transition)
            if transition.ends_episode:
                self._episode_buffer.finalize_episode()
            return
        if self._rollout_buffer is None:
            raise RuntimeError("Training engine has no buffer for its collection mode.")
        self._rollout_buffer.append(transition)

    def _update_ready_collection(self, *, final: bool) -> None:
        update_input: AgentUpdateInput | None = None
        if self._episode_buffer is not None:
            episodes = self._episode_buffer.take_completed_batch(
                self.agent.collection_size
            )
            if episodes is not None:
                update_input = AgentUpdateInput(
                    mode=CollectionMode.COMPLETE_EPISODES,
                    episodes=episodes,
                )
        elif self._rollout_buffer is not None and (
            len(self._rollout_buffer.transitions) == self.agent.collection_size
            or (final and self._rollout_buffer.transitions)
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
            near_saturated_steering_threshold=(self.near_saturated_steering_threshold),
        )
        self._evaluation_seconds += perf_counter() - started
        self.evaluations.append(evaluation)
        self.counters.evaluation_interactions += evaluation.record.episode.interactions
        self.counters.next_evaluation_identity += 1

    def _reset_training_episode(self) -> None:
        reset_seed = int(
            self.environment_reset_generator.integers(0, 2**32, dtype=np.uint32)
        )
        observation, _ = self.environment.reset(seed=reset_seed)
        identity = self.counters.next_episode_identity
        self.counters.next_episode_identity += 1
        self._current_observation = np.asarray(observation, dtype=np.float32).copy()
        self._active_episode = _ActiveEpisode(identity, self.circuit_identity)

    def _record_active_step(
        self, action: np.ndarray, reward: float, info: dict[str, Any]
    ) -> None:
        active = self._active_episode
        active.speeds.append(float(self._current_observation[2]))
        active.throttles.append(float(action[0]))
        active.absolute_steering.append(abs(float(action[1])))
        active.step_index += 1
        active.undiscounted_return += float(reward)
        progress = float(info["episode_progress"]) / self.environment.track.track_length
        active.maximum_progress = max(active.maximum_progress, progress)

    def _finish_training_episode(
        self, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> None:
        active = self._active_episode
        outcome = _outcome(terminated, truncated, info)
        final_progress = (
            float(info["episode_progress"]) / self.environment.track.track_length
        )
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
                final_progress=final_progress,
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
                circuit_seed=self.environment.track.generation.seed,
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
            "current_observation": self._current_observation.tolist(),
            "active_episode": asdict(self._active_episode),
            "environment": _environment_to_dict(self.environment.snapshot()),
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
            "random_generators": {
                "environment_reset": self.environment_reset_generator.bit_generator.state,
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
        self._current_observation = np.asarray(
            state["current_observation"], dtype=np.float32
        )
        active = _mapping(state, "active_episode")
        self._active_episode = _ActiveEpisode(**active)
        self.environment.restore(_environment_from_dict(_mapping(state, "environment")))
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
        random_state = _mapping(state, "random_generators")
        self.environment_reset_generator.bit_generator.state = random_state[
            "environment_reset"
        ]

    def _engine_configuration(self) -> dict[str, Any]:
        """
        Return immutable run facts that must match before checkpoint restoration.
        """
        return {
            "run_category": self.run_category.value,
            "collection_mode": self.agent.collection_mode.value,
            "collection_size": self.agent.collection_size,
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
        if self._episode_buffer is not None:
            return {
                "mode": CollectionMode.COMPLETE_EPISODES.value,
                "completed": [
                    [_transition_to_dict(row) for row in episode.transitions]
                    for episode in self._episode_buffer.completed_episodes
                ],
                "active": [
                    _transition_to_dict(row)
                    for row in self._episode_buffer.active_episode
                ],
            }
        if self._rollout_buffer is None:
            raise RuntimeError("Training engine has no collection buffer.")
        return {
            "mode": CollectionMode.FIXED_ROLLOUT.value,
            "transitions": [
                _transition_to_dict(row) for row in self._rollout_buffer.transitions
            ],
            "previous": (
                None
                if self._rollout_buffer.previous_transition is None
                else _transition_to_dict(self._rollout_buffer.previous_transition)
            ),
        }

    def _restore_collector(self, state: dict[str, Any]) -> None:
        mode = CollectionMode(state["mode"])
        if mode is not self.agent.collection_mode:
            raise ValueError("checkpoint collection mode does not match the agent.")
        if mode is CollectionMode.COMPLETE_EPISODES:
            if self._episode_buffer is None:
                raise RuntimeError("complete-episode buffer was not constructed.")
            completed = [
                OnPolicyRollout(tuple(_transition_from_dict(row) for row in episode))
                for episode in state["completed"]
            ]
            active = [_transition_from_dict(row) for row in state["active"]]
            self._episode_buffer.restore(completed, active)
            return
        if self._rollout_buffer is None:
            raise RuntimeError("fixed-rollout buffer was not constructed.")
        transitions = [_transition_from_dict(row) for row in state["transitions"]]
        previous = state["previous"]
        previous_transition = (
            None if previous is None else _transition_from_dict(previous)
        )
        self._rollout_buffer.restore(transitions, previous_transition)


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
    }


def _transition_from_dict(data: dict[str, Any]) -> TrainingTransition:
    """
    Reconstruct one immutable rollout row from checkpoint-safe primitives.
    """
    return TrainingTransition(**data)


def _environment_to_dict(state: RacingEnvState) -> dict[str, Any]:
    """
    Serialize focused dynamics state while deliberately omitting rendering state.
    """
    return {
        "vehicle": None if state.vehicle_state is None else asdict(state.vehicle_state),
        "lifecycle": (
            None if state.lifecycle_state is None else asdict(state.lifecycle_state)
        ),
        "episode_finished": state.episode_finished,
        "numpy_random_state": state.numpy_random_state,
    }


def _environment_from_dict(data: dict[str, Any]) -> RacingEnvState:
    """
    Reconstruct focused dynamics state for an equivalent RacingEnv instance.
    """
    vehicle_data = data["vehicle"]
    lifecycle_data = data["lifecycle"]
    return RacingEnvState(
        vehicle_state=None if vehicle_data is None else VehicleState(**vehicle_data),
        lifecycle_state=(
            None
            if lifecycle_data is None
            else EpisodeLifecycleState(
                wrapped_progress=float(lifecycle_data["wrapped_progress"]),
                episode_progress=float(lifecycle_data["episode_progress"]),
                agent_steps=int(lifecycle_data["agent_steps"]),
                previous_position=tuple(lifecycle_data["previous_position"]),
                previous_segment_index=lifecycle_data["previous_segment_index"],
            )
        ),
        episode_finished=bool(data["episode_finished"]),
        numpy_random_state=data["numpy_random_state"],
    )


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
