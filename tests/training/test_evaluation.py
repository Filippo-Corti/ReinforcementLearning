from __future__ import annotations

from pathlib import Path

import numpy as np

from agents import AgentUpdateInput, AgentUpdateOutput, CollectedAction, CollectionMode
from configs import EnvironmentConfig, ObservationNormalizationConfig, SimulationConfig
from envs.racing import RacingEnv
from envs.tracks import Track, TrackWithGeometry
from recording import RunCategory
from training import RunningObservationNormalizer, evaluate_deterministic


class _EvaluationAgent:
    collection_mode = CollectionMode.FIXED_ROLLOUT
    collection_size = 1

    def collect_action(self, normalized_observation: np.ndarray) -> CollectedAction:
        action = self.deterministic_action(normalized_observation)
        return CollectedAction(action, action, 0.0, 0.0)

    def deterministic_action(self, normalized_observation: np.ndarray) -> np.ndarray:
        del normalized_observation
        return np.asarray((0.0, 0.0), dtype=np.float32)

    def bootstrap_value(self, normalized_observation: np.ndarray) -> float:
        del normalized_observation
        return 0.0

    def update(self, update_input: AgentUpdateInput) -> AgentUpdateOutput:
        del update_input
        raise AssertionError("Evaluation must not optimize the policy.")

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


def _environment() -> RacingEnv:
    root = Path(__file__).parents[1]
    track = TrackWithGeometry(
        Track.load(root / "fixtures" / "tracks" / "valid_circle.json")
    )
    return RacingEnv(
        track,
        config=EnvironmentConfig(simulation=SimulationConfig(max_episode_steps=3)),
    )


def test_deterministic_evaluation_freezes_normalizer_and_records_summary() -> None:
    normalizer = RunningObservationNormalizer(4, ObservationNormalizationConfig())
    normalizer.update_and_normalize(np.asarray((0.0, 0.0, 0.0, 0.0)))
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
