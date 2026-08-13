from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectedActionBatch,
    CollectionMode,
)
from configs import (
    EnvironmentConfig,
    ExecutionConfig,
    ObservationNormalizationConfig,
    SimulationConfig,
    StartStateConfig,
)
from envs.observations import FrenetObservation
from envs.racing import RacingEnv
from envs.tracks import Track, TrackWithGeometry
from recording import EpisodeOutcome, RunCategory
from training import (
    CircuitSplit,
    EvaluationCircuit,
    OnPolicyTrainingEngine,
    RunningObservationNormalizer,
    TrainingCircuitSchedule,
    TrainingTransition,
    circuit_track_seed,
)
from training.engines.shared_engine import _outcome
from utils.random import SeedNamespace


class _FixedAgent:
    collection_mode: CollectionMode
    collection_size: int

    def __init__(self, mode: CollectionMode, collection_size: int) -> None:
        self.collection_mode = mode
        self.collection_size = collection_size
        self.training_actions: list[tuple[float, float]] = []
        self.updates: list[int] = []
        self.received: list[tuple[TrainingTransition, ...]] = []
        self.generator = np.random.default_rng(789)

    def collect_action(self, normalized_observation: np.ndarray) -> CollectedAction:
        del normalized_observation
        action = self.generator.uniform(-0.05, 0.05, size=2).astype(np.float32)
        self.training_actions.append((float(action[0]), float(action[1])))
        return CollectedAction(action, action, 0.0, 0.0)

    def collect_actions(
        self,
        normalized_observations: np.ndarray,
        environment_indices: Sequence[int] | None = None,
    ) -> CollectedActionBatch:
        # Delegating keeps one draw per row in the same order the engine's own
        # single-row fallback would have used.
        del environment_indices
        rows = [self.collect_action(row) for row in normalized_observations]
        return CollectedActionBatch(
            raw_actions=np.stack([row.raw_action for row in rows]),
            env_actions=np.stack([row.env_action for row in rows]),
            behaviour_log_probabilities=np.zeros(len(rows), dtype=np.float32),
            current_values=np.zeros(len(rows), dtype=np.float32),
        )

    def deterministic_action(self, normalized_observation: np.ndarray) -> np.ndarray:
        del normalized_observation
        return np.asarray((0.0, 0.0), dtype=np.float32)

    def bootstrap_value(self, normalized_observation: np.ndarray) -> float:
        del normalized_observation
        return 0.0

    def bootstrap_values(self, normalized_observations: np.ndarray) -> np.ndarray:
        return np.zeros(len(normalized_observations), dtype=np.float32)

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        rows = (
            sum(len(episode.transitions) for episode in update_input.episodes)
            if update_input.rollout is None
            else len(update_input.rollout.transitions)
        )
        self.updates.append(rows)
        if update_input.rollout is not None:
            self.received.append(update_input.rollout.transitions)
        else:
            self.received.append(
                tuple(
                    row
                    for episode in update_input.episodes
                    for row in episode.transitions
                )
            )
        return AgentUpdateOutput({"rows": rows})

    def state_dict(self) -> dict[str, Any]:
        return {
            "training_actions": self.training_actions,
            "updates": self.updates,
            "generator": deepcopy(self.generator.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.training_actions = [tuple(row) for row in state["training_actions"]]
        self.updates = list(state["updates"])
        self.generator.bit_generator.state = state["generator"]


def _environment() -> RacingEnv:
    root = Path(__file__).parents[1]
    track = TrackWithGeometry(
        Track.load(root / "fixtures" / "tracks" / "valid_circle.json")
    )
    config = EnvironmentConfig(
        simulation=SimulationConfig(max_episode_steps=3),
        start=StartStateConfig(randomized=False),
    )
    return RacingEnv(track, config=config)


def _engine(
    mode: CollectionMode = CollectionMode.FIXED_ROLLOUT,
    size: int = 4,
    *,
    evaluation_interval: int | None = None,
    workers: int | None = None,
    training_circuit_schedule: TrainingCircuitSchedule | None = None,
    evaluation_circuits: tuple[EvaluationCircuit, ...] | None = None,
) -> tuple[OnPolicyTrainingEngine, _FixedAgent]:
    agent = _FixedAgent(mode, size)
    normalizer = RunningObservationNormalizer(
        FrenetObservation.DIMENSIONS, ObservationNormalizationConfig()
    )
    default_workers = size if mode is CollectionMode.COMPLETE_EPISODES else 1
    worker_count = default_workers if workers is None else workers
    engine = OnPolicyTrainingEngine(
        agent,
        _environment(),
        normalizer,
        run_category=RunCategory.REDUCED_VALIDATION,
        evaluation_environment_factory=(
            None if evaluation_circuits is not None else _environment
        ),
        evaluation_circuits=evaluation_circuits,
        evaluation_interval=evaluation_interval,
        environment_reset_generators=tuple(
            np.random.default_rng(123 + index) for index in range(worker_count)
        ),
        track_selection_generators=tuple(
            np.random.default_rng(321 + index) for index in range(worker_count)
        ),
        training_circuit_schedule=training_circuit_schedule,
        execution_config=ExecutionConfig(
            device="cpu", environment_workers=worker_count
        ),
        evaluation_seed=456,
    )
    return engine, agent


def test_fixed_rollout_preserves_transition_and_counter_semantics() -> None:
    engine, agent = _engine(size=4)

    state = engine.train(5)

    assert state.counters.training_interactions == 5
    assert state.counters.finished_episodes == 1
    assert state.counters.optimizer_updates == 2
    assert agent.updates == [4, 1]
    assert [update.training_interactions for update in engine.updates] == [4, 5]
    rows = agent.received[0]
    assert [row.episode_identity for row in rows] == [0, 0, 0, 1]
    assert [row.episode_step_index for row in rows] == [0, 1, 2, 0]
    assert all(row.behaviour_log_probability == 0.0 for row in rows)
    assert all(row.current_value == 0.0 and row.next_value == 0.0 for row in rows)
    assert rows[2].truncated and not rows[2].terminated
    assert engine.episode_records[0].interactions == 3
    assert engine.episode_records[0].training_interactions == 3
    assert engine.normalizer.count == 5


@pytest.mark.parametrize(
    ("terminated", "truncated", "info", "expected"),
    [
        (True, False, {"lap_completed": True}, EpisodeOutcome.COMPLETED),
        (True, False, {"collision": True}, EpisodeOutcome.CRASHED),
        (True, False, {"stalled": True}, EpisodeOutcome.STALLED),
        (False, True, {}, EpisodeOutcome.TIME_LIMIT),
    ],
)
def test_every_environment_lifecycle_boundary_has_a_recorded_outcome(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    expected: EpisodeOutcome,
) -> None:
    """
    A stall is a terminal state, so it must not reach the unsupported-outcome error.
    """
    flags = {"lap_completed": False, "collision": False, "stalled": False, **info}

    assert _outcome(terminated, truncated, flags) is expected


def test_complete_episode_collection_waits_for_complete_batch() -> None:
    engine, agent = _engine(CollectionMode.COMPLETE_EPISODES, size=2)

    engine.train(7)

    assert agent.updates == [6]
    assert engine.state().counters.finished_episodes == 2
    assert engine.state().counters.optimizer_updates == 1


def test_complete_episode_batch_is_filled_over_several_worker_waves() -> None:
    """
    Fewer workers than batch episodes must still produce one update per batch.

    Each three-step episode ends by truncation, so two workers need two waves to
    supply a four-episode batch, and no optimizer step may happen in between.
    """
    engine, agent = _engine(CollectionMode.COMPLETE_EPISODES, size=4, workers=2)

    engine.train(12)

    assert agent.updates == [12]
    assert engine.state().counters.finished_episodes == 4
    assert engine.state().counters.optimizer_updates == 1


def test_every_validation_circuit_is_evaluated_at_each_checkpoint() -> None:
    """
    One checkpoint answers the same question on every validation circuit.

    A completion rate over circuits is the quantity the convergence rule reads,
    so a checkpoint has to produce one record per circuit rather than one record
    standing in for all of them.
    """
    circuits = tuple(
        EvaluationCircuit(
            identity=str(index),
            split=CircuitSplit.VALIDATION,
            factory=_environment,
        )
        for index in range(3)
    )
    engine, _ = _engine(size=4, evaluation_interval=4, evaluation_circuits=circuits)

    engine.train(8)
    records = [evaluation.record for evaluation in engine.evaluations]

    assert len(records) == 6
    assert [record.episode.circuit_identity for record in records] == [
        "0",
        "1",
        "2",
    ] * 2
    assert {record.episode.circuit_split for record in records} == {"validation"}
    # Evaluation identities stay unique so their trajectories cannot collide.
    assert len({record.evaluation_index for record in records}) == 6
    assert records[-1].evaluation_interactions == sum(
        record.episode.interactions for record in records
    )


def test_held_out_circuits_are_evaluated_only_when_asked() -> None:
    """
    Test circuits must not be reachable from the training loop.

    The engine never holds them, so opening the test set is an explicit call
    made after training rather than something a checkpoint can trigger.
    """
    engine, _ = _engine(size=4, evaluation_interval=4)
    held_out = (
        EvaluationCircuit(
            identity="test-0", split=CircuitSplit.TEST, factory=_environment
        ),
    )

    engine.train(4)
    assert [record.record.episode.circuit_split for record in engine.evaluations] == [
        None
    ]

    produced = engine.evaluate_circuits(held_out)

    assert len(produced) == 1
    assert produced[0].record.episode.circuit_split == "test"
    assert produced[0].record.training_interactions == 4
    assert engine.evaluations[-1] is produced[0]


def test_scheduled_episodes_record_the_circuit_each_one_actually_raced() -> None:
    """
    Every episode must carry its own circuit, not the prototype environment's.

    Before the schedule existed there was one circuit and the prototype was a
    faithful stand-in for it. With workers on different circuits at the same
    time, reading geometry from the prototype would label every episode with a
    circuit almost none of them drove.
    """
    engine, _ = _engine(
        size=4,
        workers=2,
        training_circuit_schedule=TrainingCircuitSchedule(),
    )

    engine.train(12)
    episodes = engine.episode_records

    assert len(episodes) == 4
    assert {episode.circuit_split for episode in episodes} == {"training"}
    identities = [episode.circuit_identity for episode in episodes]
    assert len(set(identities)) == len(identities)
    geometries = [episode.circuit_geometry for episode in episodes]
    assert all(geometry is not None for geometry in geometries)
    lengths = {geometry.track_length for geometry in geometries if geometry is not None}
    assert len(lengths) == len(episodes)
    for episode in episodes:
        assert episode.circuit_seed == circuit_track_seed(
            SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, int(episode.circuit_identity)
        )
    # Two workers, two episodes each: the pairing coordinate a paired run is
    # matched on has to be recorded, since episode identity is completion order.
    by_worker: dict[int | None, list[int | None]] = {}
    for episode in episodes:
        by_worker.setdefault(episode.collection_worker, []).append(
            episode.worker_episode_index
        )
    assert by_worker == {0: [0, 1], 1: [0, 1]}


def test_construction_consumes_the_reset_stream_independently_of_collection_mode() -> (
    None
):
    """
    Paired roots line up only if construction draws the same reset randomness.

    Complete-episode collection parks the workers a batch has no room for, and
    parking must not cost an extra environment reset relative to the fixed
    rollout used by A2C and PPO.
    """
    rollout, _ = _engine(CollectionMode.FIXED_ROLLOUT, size=4, workers=2)
    episodes, _ = _engine(CollectionMode.COMPLETE_EPISODES, size=4, workers=2)

    assert rollout.environment_reset_generator.integers(0, 2**32) == (
        episodes.environment_reset_generator.integers(0, 2**32)
    )


def test_complete_episode_batch_survives_a_checkpoint_between_waves(
    tmp_path: Path,
) -> None:
    uninterrupted, uninterrupted_agent = _engine(
        CollectionMode.COMPLETE_EPISODES, size=4, workers=2
    )
    uninterrupted.train(12)

    interrupted, _ = _engine(CollectionMode.COMPLETE_EPISODES, size=4, workers=2)
    interrupted.train(6, finalize=False)
    checkpoint = tmp_path / "training.pt"
    interrupted.save(str(checkpoint))
    resumed, resumed_agent = _engine(
        CollectionMode.COMPLETE_EPISODES, size=4, workers=2
    )
    resumed.restore(str(checkpoint))
    resumed.train(12)

    assert resumed_agent.updates == uninterrupted_agent.updates == [12]
    assert resumed.state().counters == uninterrupted.state().counters
    assert resumed.episode_records == uninterrupted.episode_records


def test_evaluation_runs_at_exact_boundary_inside_active_episode() -> None:
    engine, _ = _engine(size=4, evaluation_interval=2)
    checksum_before = engine.normalizer.checksum()

    engine.train(5)

    assert [
        evaluation.record.training_interactions for evaluation in engine.evaluations
    ] == [
        2,
        4,
    ]
    assert all(
        evaluation.record.episode.interactions == 3 for evaluation in engine.evaluations
    )
    assert engine.normalizer.checksum() != checksum_before
    assert engine.state().counters.evaluation_interactions == 6


def test_evaluation_cadence_does_not_change_training_actions_or_agent_state() -> None:
    without_evaluation, plain_agent = _engine(size=4)
    with_evaluation, evaluated_agent = _engine(size=4, evaluation_interval=2)

    without_evaluation.train(7)
    with_evaluation.train(7)

    assert evaluated_agent.training_actions == plain_agent.training_actions
    assert evaluated_agent.state_dict() == plain_agent.state_dict()


def test_checkpoint_mid_episode_matches_uninterrupted_collection(
    tmp_path: Path,
) -> None:
    uninterrupted, uninterrupted_agent = _engine(size=4, evaluation_interval=2)
    uninterrupted.train(8)

    interrupted, _ = _engine(size=4, evaluation_interval=2)
    interrupted.train(4, finalize=False)
    checkpoint = tmp_path / "training.pt"
    interrupted.save(str(checkpoint))
    resumed, resumed_agent = _engine(size=4, evaluation_interval=2)
    resumed.restore(str(checkpoint))
    resumed.train(8)

    assert resumed_agent.state_dict() == uninterrupted_agent.state_dict()
    assert resumed.state().counters == uninterrupted.state().counters
    assert resumed.normalizer.state() == uninterrupted.normalizer.state()
    assert resumed.episode_records == uninterrupted.episode_records
    assert resumed.evaluations == uninterrupted.evaluations
    assert [update.training_interactions for update in resumed.updates] == [4, 8]
    assert [update.output for update in resumed.updates] == [
        update.output for update in uninterrupted.updates
    ]
    assert resumed.environment_reset_generator.integers(0, 2**32) == (
        uninterrupted.environment_reset_generator.integers(0, 2**32)
    )


def test_timing_categories_are_non_negative_and_reconcile() -> None:
    engine, _ = _engine(size=2, evaluation_interval=2)

    timing = engine.train(4).timing

    assert timing.collection >= 0
    assert timing.optimization >= 0
    assert timing.evaluation >= 0
    assert timing.persistence >= 0
    assert timing.training_only == timing.collection + timing.optimization
    assert timing.end_to_end >= sum(
        (timing.collection, timing.optimization, timing.evaluation, timing.persistence)
    )


def test_checkpoint_rejects_incompatible_engine_state(tmp_path: Path) -> None:
    engine, _ = _engine(size=4)
    checkpoint = tmp_path / "training.pt"
    engine.save(str(checkpoint))
    state = engine._state_dict()
    state["engine_state_version"] = 999
    from training.checkpoints import save_checkpoint

    save_checkpoint(checkpoint, state)

    with pytest.raises(ValueError, match="incompatible"):
        engine.restore(str(checkpoint))


def test_checkpoint_rejects_a_different_collection_configuration(
    tmp_path: Path,
) -> None:
    engine, _ = _engine(size=4)
    checkpoint = tmp_path / "training.pt"
    engine.save(str(checkpoint))
    incompatible, _ = _engine(size=5)

    with pytest.raises(ValueError, match="configuration does not match"):
        incompatible.restore(str(checkpoint))


def test_environment_snapshot_restores_the_next_transition_exactly() -> None:
    original = _environment()
    original.reset(seed=4)
    original.step(np.asarray((0.0, 0.0), dtype=np.float32))
    snapshot = original.snapshot()
    expected = original.step(np.asarray((0.0, 0.0), dtype=np.float32))
    restored = _environment()
    restored.restore(snapshot)
    actual = restored.step(np.asarray((0.0, 0.0), dtype=np.float32))

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
