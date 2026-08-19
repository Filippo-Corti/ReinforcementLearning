"""Deterministic evaluation over a set of circuits, on a fixed interaction cadence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from circuits import EvaluationCircuit
from configs import ObservationRepresentation
from normalization import RunningObservationNormalizer
from recording.records import DeterministicEvaluationRecord, RunCategory

from .deterministic import evaluate_deterministic

if TYPE_CHECKING:
    # See deterministic.py: deferred to avoid importing `agents` (and, through
    # it, `training`) back into this package while it is still loading.
    from agents.types import OnPolicyAgent


class EvaluationScheduler:
    """
    Evaluate the deterministic policy on every circuit a checkpoint asks about.

    One circuit or sixteen makes no difference to the caller: a fixed-circuit run
    names one and a generalization run names a split, and both are answered the
    same way. Held-out circuits are never held here — they are passed to `run`
    after training, so nothing the engine holds can leak a test result into
    learning.

    Every circuit is checked against `expected_observation_space` before it is
    ever evaluated. Circuits are prepared by the caller, so nothing otherwise
    stops a LiDAR run from being measured on Frenet environments — that would
    not crash, since the normalizer would reject the width, but only after a
    full run's training. Checking here makes it a startup error instead.

    Fields:
        * circuits: Circuits evaluated at every due checkpoint.
        * interval: Training interactions between checkpoints, or none.
        * records: Evaluations produced so far, in the order they were run.
    """

    def __init__(
        self,
        circuits: Sequence[EvaluationCircuit],
        *,
        expected_observation_space: gym.spaces.Box,
        interval: int | None,
        evaluation_seed: int,
        run_category: RunCategory,
        observation_type: ObservationRepresentation,
        root_identity: int | None = None,
        near_saturated_steering_threshold: float | None = None,
    ) -> None:
        if interval is not None and interval <= 0:
            raise ValueError("Evaluation interval must be positive when enabled.")
        self.circuits = tuple(circuits)
        self._expected_observation_space = expected_observation_space
        self._assert_circuits_match(self.circuits)
        self.interval = interval
        self.evaluation_seed = evaluation_seed
        self.run_category = run_category
        self.observation_type = observation_type
        self.root_identity = root_identity
        self.near_saturated_steering_threshold = near_saturated_steering_threshold
        self.records: list[DeterministicEvaluationRecord] = []
        self._next_index = 0
        self._evaluation_interactions = 0

    @property
    def next_evaluation_identity(self) -> int:
        """
        Return the identity the next evaluation record will take.
        """
        return self._next_index

    @property
    def evaluation_interactions(self) -> int:
        """
        Return the interactions spent evaluating, which no budget counts.
        """
        return self._evaluation_interactions

    def due(self, training_interactions: int) -> bool:
        """
        Report whether a scheduled checkpoint falls exactly here.
        """
        return (
            self.interval is not None
            and bool(self.circuits)
            and training_interactions > 0
            and training_interactions % self.interval == 0
        )

    def run(
        self,
        circuits: Sequence[EvaluationCircuit],
        *,
        agent: OnPolicyAgent,
        normalizer: RunningObservationNormalizer,
        training_interactions: int,
        collection_duration: float = 0.0,
        optimization_duration: float = 0.0,
    ) -> tuple[DeterministicEvaluationRecord, ...]:
        """
        Evaluate once on each circuit given, recording one result per circuit.
        Returns the produced records in the order of the circuits given.
        """
        self._assert_circuits_match(circuits)
        produced = [
            self._evaluate(
                circuit,
                agent=agent,
                normalizer=normalizer,
                training_interactions=training_interactions,
                collection_duration=collection_duration,
                optimization_duration=optimization_duration,
            )
            for circuit in circuits
        ]
        return tuple(produced)

    def _evaluate(
        self,
        circuit: EvaluationCircuit,
        *,
        agent: OnPolicyAgent,
        normalizer: RunningObservationNormalizer,
        training_interactions: int,
        collection_duration: float,
        optimization_duration: float,
    ) -> DeterministicEvaluationRecord:
        index = self._next_index
        reset_seed = int(
            np.random.SeedSequence([self.evaluation_seed, index]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        evaluation = evaluate_deterministic(
            circuit.factory,
            agent,
            normalizer,
            run_category=self.run_category,
            evaluation_index=index,
            training_interactions=training_interactions,
            evaluation_interactions_before=self._evaluation_interactions,
            reset_seed=reset_seed,
            root_identity=self.root_identity,
            circuit_identity=circuit.identity,
            circuit_split=None if circuit.split is None else circuit.split.value,
            observation_type=self.observation_type,
            collection_duration=collection_duration,
            optimization_duration=optimization_duration,
            near_saturated_steering_threshold=(self.near_saturated_steering_threshold),
        )
        self.records.append(evaluation)
        self._evaluation_interactions += evaluation.record.episode.interactions
        self._next_index += 1
        return evaluation

    def state(self) -> dict[str, object]:
        """
        Return the evaluation counters and results for a checkpoint.
        """
        return {
            "next_identity": self._next_index,
            "evaluation_interactions": self._evaluation_interactions,
            "records": list(self.records),
        }

    def restore(self, state: dict[str, object]) -> None:
        """
        Resume evaluation identities and totals from a checkpoint.
        """
        self._next_index = int(state["next_identity"])  # type: ignore[arg-type]
        self._evaluation_interactions = int(state["evaluation_interactions"])  # type: ignore[arg-type]
        records = state["records"]
        if not isinstance(records, list) or not all(
            isinstance(record, DeterministicEvaluationRecord) for record in records
        ):
            raise TypeError("checkpoint evaluation records have invalid types.")
        self.records = list(records)

    def _assert_circuits_match(self, circuits: Sequence[EvaluationCircuit]) -> None:
        """
        Refuse circuits that would show the policy a different representation.
        """
        for circuit in circuits[:1]:
            evaluation_environment = circuit.factory()
            try:
                actual = evaluation_environment.observation_space
                if actual.shape != self._expected_observation_space.shape:
                    raise ValueError(
                        "Evaluation circuits must use the training observation type: "
                        f"training observes {self._expected_observation_space.shape}, "
                        f"circuit {circuit.identity!r} observes {actual.shape}."
                    )
            finally:
                evaluation_environment.close()
