"""Shared parallel collection, update, evaluation, timing, and resume lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.types import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectionMode,
    OnPolicyAgent,
)
from configs import ExecutionConfig, ObservationRepresentation
from envs.racing import RacingEnv
from recording.records import (
    DeterministicEvaluationRecord,
    EpisodeRecord,
    ObservationNormalizerStateRecord,
    RunCategory,
    TimingRecord,
)

from ..buffers import (
    OnPolicyRollout,
    TrainingTransition,
    VectorRolloutBuffer,
)
from ..circuits import (
    CircuitSplit,
    EvaluationCircuit,
    TrainingCircuitSchedule,
    generated_evaluation_circuits,
)
from ..normalization import RunningObservationNormalizer
from ..vector_environment import PersistentRacingVectorEnv, VectorRacingState
from .checkpointing import EngineCheckpoint, mapping, typed_list
from .episode_recording import EpisodeRecorder, episode_outcome
from .evaluation_schedule import EvaluationSchedule, check_evaluation_observations
from .stepping import StepCollector
from .timing import TrainingTimer

# Re-exported for the tests and callers that assert on outcome classification.
_outcome = episode_outcome


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

    What is shared with every other engine — stepping the workers, recording an
    episode, evaluating a checkpoint, timing, checkpointing — is delegated to
    collaborators, so what remains here is the collection strategy itself.

    Fields:
        * agent: Owner of policy/value models, optimizers, and sampling streams.
        * environment: Unstepped prototype used for fixed configuration and geometry.
        * environments: Persistent process-based training environment pool.
        * normalizer: Training-updated and evaluation-frozen observation normalizer.
        * execution_config: Device, worker, threading, and deterministic settings.
        * run_category: Run namespace carried by emitted metric summaries.
    """

    def __init__(
        self,
        agent: OnPolicyAgent,
        environment: RacingEnv,
        normalizer: RunningObservationNormalizer,
        *,
        run_category: RunCategory,
        evaluation_environment_factory: Callable[[], RacingEnv] | None = None,
        evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
        evaluation_interval: int | None = None,
        environment_reset_generator: np.random.Generator | None = None,
        environment_reset_generators: Sequence[np.random.Generator] | None = None,
        track_selection_generators: Sequence[np.random.Generator] | None = None,
        training_circuit_schedule: TrainingCircuitSchedule | None = None,
        execution_config: ExecutionConfig | None = None,
        evaluation_seed: int = 0,
        root_identity: int | None = None,
        circuit_identity: str | None = None,
        circuit_split: str | None = None,
        observation_type: ObservationRepresentation = ObservationRepresentation.FRENET,
        near_saturated_steering_threshold: float | None = None,
    ) -> None:
        """
        Spawn workers once and initialize one active episode in every process.
        """
        if agent.collection_size <= 0:
            raise ValueError("An on-policy agent collection size must be positive.")
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
        self.environment_reset_generator = reset_generators[0]
        self.root_identity = root_identity
        self.circuit_identity = circuit_identity or str(
            environment.track.generation.seed
        )
        self.circuit_split = circuit_split
        self.observation_type = observation_type
        self.training_circuit_schedule = training_circuit_schedule
        self.near_saturated_steering_threshold = near_saturated_steering_threshold

        self.timer = TrainingTimer()
        self.recorder = EpisodeRecorder(
            worker_count,
            run_category=run_category,
            observation_type=observation_type,
            root_identity=root_identity,
            circuit_split=(
                CircuitSplit.TRAINING.value
                if training_circuit_schedule is not None
                else circuit_split
            ),
            near_saturated_steering_threshold=near_saturated_steering_threshold,
        )
        self.schedule = EvaluationSchedule(
            _resolve_evaluation_circuits(
                evaluation_circuits,
                evaluation_environment_factory,
                identity=self.circuit_identity,
                split=circuit_split,
            ),
            interval=evaluation_interval,
            evaluation_seed=evaluation_seed,
            run_category=run_category,
            observation_type=observation_type,
            root_identity=root_identity,
            near_saturated_steering_threshold=near_saturated_steering_threshold,
        )
        check_evaluation_observations(self.schedule.circuits, environment)

        self.training_interactions = 0
        self.optimizer_updates = 0
        self.updates: list[TrainingUpdate] = []
        self.environments = PersistentRacingVectorEnv(
            (environment.track_with_geometry,),
            environment.config,
            self.execution_config,
            reset_generators,
            selection_generators,
            training_circuit_schedule,
        )
        self.collector = StepCollector(
            agent,
            self.environments,
            normalizer,
            self.recorder,
            worker_count=worker_count,
        )
        self.checkpoint = EngineCheckpoint(self._engine_configuration())

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

    @property
    def evaluation_interval(self) -> int | None:
        """
        Return the training interactions between deterministic checkpoints.
        """
        return self.schedule.interval

    @property
    def evaluation_circuits(self) -> tuple[EvaluationCircuit, ...]:
        """
        Return the circuits evaluated at every checkpoint.
        """
        return self.schedule.circuits

    @property
    def episode_records(self) -> list[EpisodeRecord]:
        """
        Return every finished training episode, in completion order.
        """
        return self.recorder.records

    @property
    def evaluations(self) -> list[DeterministicEvaluationRecord]:
        """
        Return every deterministic evaluation, in the order it was run.
        """
        return self.schedule.records

    @property
    def counters(self) -> TrainingCounters:
        """
        Return the run's counters, each read from whichever object owns it.
        """
        return TrainingCounters(
            training_interactions=self.training_interactions,
            evaluation_interactions=self.schedule.evaluation_interactions,
            finished_episodes=len(self.recorder.records),
            optimizer_updates=self.optimizer_updates,
            next_episode_identity=self.recorder.next_episode_identity,
            next_evaluation_identity=self.schedule.next_evaluation_identity,
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
            available = ~self._parked
            maximum_rows = interaction_budget - self.training_interactions
            if self._rollout_buffer is not None:
                maximum_rows = min(
                    maximum_rows, self._rollout_buffer.remaining_capacity
                )
            if self.schedule.interval is not None:
                distance = self.schedule.interval - (
                    self.training_interactions % self.schedule.interval
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
        return TrainingRunState(counters=self.counters, timing=self.timing())

    def timing(self) -> TimingRecord:
        """
        Return mutually exclusive component times and elapsed end-to-end time.
        """
        return self.timer.record(self.run_category)

    def save(self, path: str | Path) -> None:
        """
        Atomically save every worker and collector state for exact resume.
        """
        with self.timer.persisting():
            self.checkpoint.save(path, self._state_sections())

    def restore(self, path: str | Path, *, map_location: str = "cpu") -> None:
        """
        Restore a checkpoint onto equivalent agent and worker instances.
        """
        with self.timer.persisting():
            self._restore_sections(
                self.checkpoint.load(path, map_location=map_location)
            )

    def close(self) -> None:
        """
        Close the persistent worker processes and prototype environment.
        """
        self.environments.close()
        self.environment.close()

    def training_reference_circuits(self, count: int) -> tuple[EvaluationCircuit, ...]:
        """
        Name circuits this run actually trained on, for an in-sample reference.

        They are taken in per-worker episode order rather than in the order
        episodes happened to finish, so two paired runs name the same circuits
        even though they completed episodes at different times.
        """
        if self.training_circuit_schedule is None:
            raise ValueError("Only a scheduled run has training circuits to revisit.")
        ordered = sorted(
            self.recorder.records,
            key=lambda record: (
                record.worker_episode_index or 0,
                record.collection_worker or 0,
            ),
        )
        identities: list[str] = []
        for record in ordered:
            if record.circuit_identity not in identities:
                identities.append(record.circuit_identity)
            if len(identities) == count:
                break
        if len(identities) < count:
            raise ValueError(
                f"The run raced {len(identities)} circuits, fewer than the {count} "
                "requested as a training reference."
            )
        return generated_evaluation_circuits(
            (int(identity) for identity in identities),
            split=CircuitSplit.TRAINING_REFERENCE,
            environment_config=self.environment.config,
            namespace=self.training_circuit_schedule.namespace,
        )

    def evaluate_circuits(
        self, circuits: Sequence[EvaluationCircuit]
    ) -> tuple[DeterministicEvaluationRecord, ...]:
        """
        Evaluate the current deterministic policy once on each circuit given.

        Held-out circuits are evaluated through this method after training ends,
        which is why it takes its circuits as an argument: the engine never holds
        a reference to a test circuit while it is still learning.
        """
        check_evaluation_observations(circuits, self.environment)
        with self.timer.evaluating():
            return self.schedule.run(
                circuits,
                agent=self.agent,
                normalizer=self.normalizer,
                training_interactions=self.training_interactions,
                collection_duration=self.timer.collection,
                optimization_duration=self.timer.optimization,
            )

    def _evaluate_if_due(self) -> None:
        if self.schedule.due(self.training_interactions):
            self.evaluate_circuits(self.schedule.circuits)

    def _collect_step(self, active: np.ndarray) -> None:
        with self.timer.collecting():
            step = self.collector.step(
                active,
                training_interactions=self.training_interactions,
                evaluation_interactions=self.schedule.evaluation_interactions,
            )
            self.training_interactions += step.interactions

            reset_mask = np.zeros(
                self.execution_config.environment_workers, dtype=np.bool_
            )
            if self._reinforce_active is not None:
                for worker_index, transition in enumerate(step.transitions):
                    if transition is not None:
                        self._reinforce_active[worker_index].append(transition)
                for worker_index in step.finished:
                    self._finish_reinforce_trajectory(worker_index)
            else:
                for worker_index in step.finished:
                    reset_mask[worker_index] = True
            if self._rollout_buffer is not None:
                self._rollout_buffer.append_step(step.transitions)
            if np.any(reset_mask):
                self.collector.reset_workers(reset_mask)

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
        self.collector.reset_workers(wave)

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
        with self.timer.optimizing() as elapsed:
            output = self.agent.update(update_input)
        self.updates.append(
            TrainingUpdate(
                update_index=self.optimizer_updates,
                training_interactions=self.training_interactions,
                output=output,
                optimization_duration=elapsed.seconds,
            )
        )
        self.optimizer_updates += 1
        if self._reinforce_batch is not None:
            self._reinforce_batch = []
            self._start_reinforce_wave()

    def _state_sections(self) -> dict[str, Any]:
        return {
            "agent": self.agent.state_dict(),
            "normalizer": self.normalizer.state().to_dict(),
            "training_interactions": self.training_interactions,
            "optimizer_updates": self.optimizer_updates,
            "stepping": self.collector.state(),
            "recorder": self.recorder.state(),
            "schedule": self.schedule.state(),
            "vector_environment": self.environments.state(),
            "parked": self._parked.tolist(),
            "collector": self._collector_state(),
            "updates": self.updates,
            "timing": self.timer.state(),
        }

    def _restore_sections(self, state: dict[str, Any]) -> None:
        self.agent.load_state_dict(mapping(state, "agent"))
        normalizer_state = mapping(state, "normalizer")
        self.normalizer.restore(
            ObservationNormalizerStateRecord(
                count=int(normalizer_state["count"]),
                sums=tuple(float(value) for value in normalizer_state["sums"]),
                squared_sums=tuple(
                    float(value) for value in normalizer_state["squared_sums"]
                ),
            )
        )
        self.training_interactions = int(state["training_interactions"])
        self.optimizer_updates = int(state["optimizer_updates"])
        self.collector.restore(mapping(state, "stepping"))
        self.recorder.restore(mapping(state, "recorder"))
        self.schedule.restore(mapping(state, "schedule"))
        vector_state = state["vector_environment"]
        if not isinstance(vector_state, VectorRacingState):
            raise TypeError("checkpoint vector environment state is invalid.")
        self.environments.restore(vector_state)
        self._parked = np.asarray(state["parked"], dtype=np.bool_)
        self._restore_collector(mapping(state, "collector"))
        self.updates = typed_list(state, "updates", TrainingUpdate)
        self.timer.restore(mapping(state, "timing"))

    def _engine_configuration(self) -> dict[str, Any]:
        # A scheduled run has no single circuit, so identifying it by the
        # prototype's geometry would reject every legitimate resume. The
        # schedule namespace is what has to match instead.
        circuit_configuration: dict[str, Any] = (
            {"training_circuit_schedule": self.training_circuit_schedule.namespace.name}
            if self.training_circuit_schedule is not None
            else {
                "track_seed": self.environment.track.generation.seed,
                "track_length": self.environment.track.track_length,
                "track_samples": int(self.environment.track.s.size),
            }
        )
        return {
            **circuit_configuration,
            "run_category": self.run_category.value,
            "collection_mode": self.agent.collection_mode.value,
            "collection_size": self.agent.collection_size,
            "environment_workers": self.execution_config.environment_workers,
            "evaluation_interval": self.schedule.interval,
            "evaluation_seed": self.schedule.evaluation_seed,
            "root_identity": self.root_identity,
            "circuit_identity": self.circuit_identity,
            "circuit_split": self.circuit_split,
            "observation_type": self.observation_type.value,
            "near_saturated_steering_threshold": (
                self.near_saturated_steering_threshold
            ),
            "environment_config": self.environment.config.to_dict(),
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


def _resolve_evaluation_circuits(
    circuits: Sequence[EvaluationCircuit] | None,
    factory: Callable[[], RacingEnv] | None,
    *,
    identity: str,
    split: str | None,
) -> tuple[EvaluationCircuit, ...]:
    """
    Treat a lone evaluation environment as a one-circuit evaluation set.

    A fixed-circuit run and a multi-circuit run then share one evaluation path,
    so the difference between them stays in how many circuits they name.
    """
    if circuits is not None:
        if factory is not None:
            raise ValueError(
                "Provide either evaluation circuits or one evaluation factory."
            )
        return tuple(circuits)
    if factory is None:
        return ()
    return (
        EvaluationCircuit(
            identity=identity,
            split=None if split is None else CircuitSplit(split),
            factory=factory,
        ),
    )


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
