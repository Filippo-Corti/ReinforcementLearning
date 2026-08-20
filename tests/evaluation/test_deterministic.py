from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from agents import (
    AgentUpdateInput,
    AgentUpdateOutput,
    CollectedAction,
    CollectedActionBatch,
    CollectionMode,
    OnPolicyAgent,
)
from configs import (
    EnvironmentConfig,
    ObservationNormalizationConfig,
    SimulationConfig,
    StartStateConfig,
)
from envs.observations import FrenetObservation
from envs.racing import RacingEnv
from envs.tracks import Track, TrackWithGeometry
from recording import RunCategory
from training import RunningObservationNormalizer, evaluate_deterministic


class _EvaluationAgent(OnPolicyAgent):
    collection_mode = CollectionMode.FIXED_ROLLOUT
    collection_size = 1

    def __init__(self) -> None:
        """
        Skip the real construction, since this stand-in owns no models.
        """

    def collect_action(self, normalized_observation: np.ndarray) -> CollectedAction:
        action = self.deterministic_action(normalized_observation)
        return CollectedAction(action, action, 0.0, 0.0)

    def collect_actions(
        self,
        normalized_observations: np.ndarray,
        environment_indices: Sequence[int] | None = None,
    ) -> CollectedActionBatch:
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
        del update_input
        raise AssertionError("Evaluation must not optimize the policy.")

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state

    @property
    def actor_parameter_count(self) -> int:
        """
        Report no trainable parameters, since this stand-in owns no models.
        """
        return 0

    @property
    def critic_parameter_count(self) -> int | None:
        """
        Report no critic, since this stand-in owns no models.
        """
        return None


def _environment() -> RacingEnv:
    root = Path(__file__).parents[1]
    track = TrackWithGeometry(
        Track.load(root / "fixtures" / "tracks" / "valid_circle.json")
    )
    return RacingEnv(
        track,
        config=EnvironmentConfig(
            simulation=SimulationConfig(max_episode_steps=3),
            start=StartStateConfig(randomized=False),
        ),
    )


def test_deterministic_evaluation_freezes_normalizer_and_records_summary() -> None:
    normalizer = RunningObservationNormalizer(
        FrenetObservation.DIMENSIONS, ObservationNormalizationConfig()
    )
    normalizer.update_and_normalize(
        np.zeros(FrenetObservation.DIMENSIONS, dtype=np.float64)
    )
    before = normalizer.state()

    evaluation = evaluate_deterministic(
        _environment,
        _EvaluationAgent(),
        normalizer,
        run_category=RunCategory.REDUCED_VALIDATION,
        evaluation_index=4,
        training_interactions=12,
        evaluation_interactions_before=8,
        reset_seed=9,
    )

    assert normalizer.state() == before
    assert evaluation.record.training_interactions == 12
    assert evaluation.record.evaluation_interactions == 11
    assert len(evaluation.transitions) == 3
