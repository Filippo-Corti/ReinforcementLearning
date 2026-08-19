"""Everything a training engine needs that is not its collection strategy.

Each algorithm owns its training loop, in its own file, written in its own
terms. What is not about the algorithm — spawning workers, recording an episode,
evaluating a checkpoint, timing, saving and resuming — is the same work three
times over, so it is assembled here and written once.

Nothing in this class is a loop. Reading `PPOTrainingEngine.train` tells you
what PPO does; reading this tells you where its records go.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.types import AgentUpdateOutput, OnPolicyAgent
from circuits import (
    CircuitSplit,
    EvaluationCircuit,
    TrainingCircuitSchedule,
    generated_evaluation_circuits,
)
from configs import EnvironmentConfig, ExecutionConfig, ObservationRepresentation
from envs.racing import RacingEnv, observation_space_for
from envs.tracks import TrackWithGeometry
from evaluation import EvaluationScheduler
from normalization import RunningObservationNormalizer
from recording.records import (
    DeterministicEvaluationRecord,
    EpisodeRecord,
    ObservationNormalizerStateRecord,
    RunCategory,
    TimingRecord,
)

from ..checkpointing import EngineCheckpoint, mapping, typed_list
from ..multienvs import MultiEnvironmentManager
from ..timing import TrainingProgress, TrainingTimer


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


class TrainingEngine(ABC):
    """
    Hold the machinery every on-policy engine needs, and none of their loops.

    Neural inference is batched in the parent process, where the TrainingEngine persists.
    Environment dynamics run in separately spawned CPU processes that persist across
    rollouts, updates, evaluations, checkpoints, and repeated calls to `train`.

    Fields:
        * agent: Owner of policy/value models, optimizers, and sampling streams.
        * track_with_geometry: Circuit fixing the run's geometry when not scheduled.
        * environment_config: Behaviour configuration shared by every environment.
        * normalizer: Training-updated and evaluation-frozen observation normalizer.
        * envs_manager: Owns the worker pool, steps it, and accumulates episodes.
        * evaluation_scheduler: Evaluates the deterministic policy on a set of circuits.
        * timer: Tracker of how much time the training spends in each of its phases.
        * progress: Optional bar reporting how far a training call has got.
        * updates: One entry per completed optimizer update.
    """

    def __init__(
        self,
        agent: OnPolicyAgent,
        track: TrackWithGeometry,
        environment_config: EnvironmentConfig,
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
        show_progress: bool = False,
    ) -> None:
        """
        Spawn workers once and open one episode in every process.

        `track` fixes the run's geometry for a fixed-circuit run. A scheduled
        run replaces its circuit at every worker reset instead, so there
        `track` only supplies the observation shape and a default identity
        before the first reset ever happens.
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

        reset_generators = resolve_generators(
            environment_reset_generators,
            environment_reset_generator,
            worker_count,
            "reset",
        )
        selection_generators = resolve_generators(
            track_selection_generators,
            None,
            worker_count,
            "track-selection",
        )
        self.agent = agent
        self.track_with_geometry = track
        self.environment_config = environment_config
        self.normalizer = normalizer
        self.run_category = run_category
        self.evaluation_environment_factory = evaluation_environment_factory
        self.environment_reset_generator = reset_generators[0]
        self.root_identity = root_identity
        self.circuit_identity = circuit_identity or str(track.track.generation.seed)
        self.circuit_split = circuit_split
        self.observation_type = observation_type
        self.training_circuit_schedule = training_circuit_schedule
        self.near_saturated_steering_threshold = near_saturated_steering_threshold

        self.timer = TrainingTimer()
        self.progress = TrainingProgress(enabled=show_progress)
        self.evaluation_scheduler = EvaluationScheduler(
            resolve_evaluation_circuits(
                evaluation_circuits,
                evaluation_environment_factory,
                identity=self.circuit_identity,
                split=circuit_split,
            ),
            expected_observation_space=observation_space_for(track, environment_config),
            interval=evaluation_interval,
            evaluation_seed=evaluation_seed,
            run_category=run_category,
            observation_type=observation_type,
            root_identity=root_identity,
            near_saturated_steering_threshold=near_saturated_steering_threshold,
        )

        self.training_interactions = 0
        self.optimizer_updates = 0
        self.updates: list[TrainingUpdate] = []
        self.envs_manager = MultiEnvironmentManager(
            agent,
            normalizer,
            track=track,
            environment_config=environment_config,
            execution_config=self.execution_config,
            reset_generators=reset_generators,
            selection_generators=selection_generators,
            training_circuit_schedule=training_circuit_schedule,
            run_category=run_category,
            observation_type=observation_type,
            root_identity=root_identity,
            circuit_split=circuit_split,
            near_saturated_steering_threshold=near_saturated_steering_threshold,
        )
        self.checkpoint = EngineCheckpoint(self._engine_configuration())
        self.parked_mask = np.zeros(worker_count, dtype=np.bool_)
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """
        Create whatever the algorithm collects into, once the workers exist.
        """

    @property
    def worker_count(self) -> int:
        """
        Return the number of persistent collection workers.
        """
        return self.execution_config.environment_workers

    @property
    def evaluation_interval(self) -> int | None:
        """
        Return the training interactions between deterministic checkpoints.
        """
        return self.evaluation_scheduler.interval

    @property
    def evaluation_circuits(self) -> tuple[EvaluationCircuit, ...]:
        """
        Return the circuits evaluated at every checkpoint.
        """
        return self.evaluation_scheduler.circuits

    @property
    def episode_records(self) -> list[EpisodeRecord]:
        """
        Return every finished training episode, in completion order.
        """
        return self.envs_manager.episode_records

    @property
    def evaluations(self) -> list[DeterministicEvaluationRecord]:
        """
        Return every deterministic evaluation, in the order it was run.
        """
        return self.evaluation_scheduler.records

    @property
    def counters(self) -> TrainingCounters:
        """
        Return the run's counters, each read from whichever object owns it.
        """
        return TrainingCounters(
            training_interactions=self.training_interactions,
            evaluation_interactions=self.evaluation_scheduler.evaluation_interactions,
            finished_episodes=len(self.envs_manager.episode_records),
            optimizer_updates=self.optimizer_updates,
            next_episode_identity=self.envs_manager.next_episode_identity,
            next_evaluation_identity=self.evaluation_scheduler.next_evaluation_identity,
        )

    @abstractmethod
    def train(
        self, interaction_budget: int, *, finalize: bool = True
    ) -> TrainingRunState:
        """
        Collect and optimize until the exact requested interaction budget is reached.
        """

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
            self.checkpoint.save(
                path,
                {
                    "agent": self.agent.state_dict(),
                    "normalizer": self.normalizer.state().to_dict(),
                    "training_interactions": self.training_interactions,
                    "optimizer_updates": self.optimizer_updates,
                    "environments": self.envs_manager.state(),
                    "schedule": self.evaluation_scheduler.state(),
                    "parked_mask": self.parked_mask.tolist(),
                    "collector": self.collector_state(),
                    "updates": self.updates,
                    "timing": self.timer.state(),
                },
            )

    def restore(self, path: str | Path, *, map_location: str = "cpu") -> None:
        """
        Restore a checkpoint onto equivalent agent and worker instances.
        """
        with self.timer.persisting():
            state = self.checkpoint.load(path, map_location=map_location)
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
        self.envs_manager.restore(mapping(state, "environments"))
        self.evaluation_scheduler.restore(mapping(state, "schedule"))
        self.parked_mask = np.asarray(state["parked_mask"], dtype=np.bool_)
        self.restore_collector(mapping(state, "collector"))
        self.updates = typed_list(state, "updates", TrainingUpdate)
        self.timer.restore(mapping(state, "timing"))

    def close(self) -> None:
        """
        Close the persistent worker processes.
        """
        self.envs_manager.close()

    def training_reference_circuits(self, count: int) -> tuple[EvaluationCircuit, ...]:
        """
        Prepare evaluation circuits out of the ones this run actually trained on.

        Both halves of the name are literal, which is what makes them look
        contradictory. *Training* says where the circuits come from: the first
        `count` distinct circuits this run raced. *EvaluationCircuit* says what
        comes back, because a circuit only becomes measurable once it is paired
        with a factory that builds a deterministic environment for it, and that
        pairing is what an `EvaluationCircuit` is. So this hands back circuits
        of training provenance, packaged to be evaluated.

        The purpose is an in-sample reference. A held-out score means little on
        its own; it means something against the same deterministic evaluation
        run on circuits the policy has already seen, and the gap between the two
        is what this project calls generalization. That is also why they carry
        `CircuitSplit.TRAINING_REFERENCE` rather than the plain training split:
        they are trained-on circuits being used as a measurement.

        They are taken in per-worker episode order rather than in the order
        episodes happened to finish, so two paired runs name the same circuits
        even though they completed episodes at different times.
        """
        if self.training_circuit_schedule is None:
            raise ValueError("Only a scheduled run has training circuits to revisit.")
        ordered = sorted(
            self.envs_manager.episode_records,
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
            environment_config=self.environment_config,
            namespace=self.training_circuit_schedule.namespace,
        )

    def evaluate(
        self, circuits: Sequence[EvaluationCircuit] | None = None
    ) -> tuple[DeterministicEvaluationRecord, ...]:
        """
        Evaluate the deterministic policy, on the schedule or on demand.

        With no circuits given, this evaluates the checkpoint schedule's own
        circuits, but only when a checkpoint is due here. With circuits given
        explicitly — the held-out set, run once after training ends — it always
        evaluates them: passing them in is itself the request, so nothing the
        engine holds can leak a test result into learning by evaluating it early.
        """
        if circuits is None:
            if not self.evaluation_scheduler.due(self.training_interactions):
                return ()
            circuits = self.evaluation_scheduler.circuits
        with self.timer.evaluating():
            return self.evaluation_scheduler.run(
                circuits,
                agent=self.agent,
                normalizer=self.normalizer,
                training_interactions=self.training_interactions,
                collection_duration=self.timer.collection,
                optimization_duration=self.timer.optimization,
            )

    def record_update(self, output: AgentUpdateOutput, duration: float) -> None:
        """
        Attach one completed optimizer update to the boundary it ran at.
        """
        self.updates.append(
            TrainingUpdate(
                update_index=self.optimizer_updates,
                training_interactions=self.training_interactions,
                output=output,
                optimization_duration=duration,
            )
        )
        self.optimizer_updates += 1

    @abstractmethod
    def collector_state(self) -> dict[str, Any]:
        """
        Return the algorithm's partially collected experience for a checkpoint.
        """

    @abstractmethod
    def restore_collector(self, state: dict[str, Any]) -> None:
        """
        Restore the algorithm's partially collected experience.
        """

    def _engine_configuration(self) -> dict[str, Any]:
        # A scheduled run has no single circuit, so identifying it by the
        # fixed track's geometry would reject every legitimate resume. The
        # schedule namespace is what has to match instead.
        circuit_configuration: dict[str, Any] = (
            {"training_circuit_schedule": self.training_circuit_schedule.namespace.name}
            if self.training_circuit_schedule is not None
            else {
                "track_seed": self.track_with_geometry.track.generation.seed,
                "track_length": self.track_with_geometry.track.track_length,
                "track_samples": int(self.track_with_geometry.track.s.size),
            }
        )
        return {
            **circuit_configuration,
            "run_category": self.run_category.value,
            "collection_mode": self.agent.collection_mode.value,
            "collection_size": self.agent.collection_size,
            "environment_workers": self.execution_config.environment_workers,
            "evaluation_interval": self.evaluation_scheduler.interval,
            "evaluation_seed": self.evaluation_scheduler.evaluation_seed,
            "root_identity": self.root_identity,
            "circuit_identity": self.circuit_identity,
            "circuit_split": self.circuit_split,
            "observation_type": self.observation_type.value,
            "near_saturated_steering_threshold": (
                self.near_saturated_steering_threshold
            ),
            "environment_config": self.environment_config.to_dict(),
            "normalizer_dimensions": self.normalizer.observation_dimensions,
        }


def resolve_evaluation_circuits(
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


def resolve_generators(
    generators: Sequence[np.random.Generator] | None,
    single_generator: np.random.Generator | None,
    worker_count: int,
    role: str,
) -> tuple[np.random.Generator, ...]:
    """
    Accept one generator per worker, or one generator for a single worker.
    """
    if generators is not None:
        values = tuple(generators)
    elif single_generator is not None:
        values = (single_generator,)
    else:
        values = tuple(np.random.default_rng(index) for index in range(worker_count))
    if len(values) != worker_count:
        raise ValueError(f"One {role} generator is required per environment worker.")
    return values


def transition_to_dict(row: Any) -> dict[str, Any]:
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
