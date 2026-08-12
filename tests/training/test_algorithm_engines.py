from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from agents import A2CAgent, PPOAgent, ReinforceAgent
from configs import (
    A2CConfig,
    ActorConfig,
    CriticConfig,
    EnvironmentConfig,
    ExecutionConfig,
    ObservationNormalizationConfig,
    PPOConfig,
    ReinforceConfig,
    SimulationConfig,
)
from envs.tracks import TrackWithGeometry
from training.engines.a2c import A2CTrainingEngine
from training.engines.ppo import PPOTrainingEngine
from training.engines.reinforce import ReinforceTrainingEngine
from training.normalization import RunningObservationNormalizer


def _tracks() -> tuple[TrackWithGeometry, TrackWithGeometry]:
    source = Path(__file__).parents[1] / "fixtures" / "tracks" / "valid_circle.json"
    first = TrackWithGeometry.load(source)
    second_track = replace(
        first.track,
        generation=replace(first.track.generation, seed=1),
    )
    return first, TrackWithGeometry(second_track)


def _environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(simulation=SimulationConfig(max_episode_steps=1))


def _normalizer() -> RunningObservationNormalizer:
    return RunningObservationNormalizer(4, ObservationNormalizationConfig())


def _actor_config() -> ActorConfig:
    return ActorConfig(
        name="small",
        hidden_sizes=(4, 4),
        learning_rate=0.01,
    )


def test_reinforce_engine_waits_for_complete_episode_batches() -> None:
    tracks = _tracks()
    selection_seed = 19
    completed_episodes: list[tuple[int, int]] = []
    agent = ReinforceAgent(
        observation_dimensions=4,
        actor_config=_actor_config(),
        config=ReinforceConfig(completed_episodes_per_update=2),
        initialization_generator=torch.Generator().manual_seed(1),
        sampling_generator=(
            torch.Generator().manual_seed(2),
            torch.Generator().manual_seed(3),
        ),
    )
    engine = ReinforceTrainingEngine(
        agent,
        tracks,
        _environment_config(),
        _normalizer(),
        (np.random.default_rng(3), np.random.default_rng(4)),
        (
            np.random.default_rng(selection_seed),
            np.random.default_rng(selection_seed + 1),
        ),
        execution_config=ExecutionConfig(device="cpu", environment_workers=2),
    )

    history = engine.train(
        5,
        on_episode_end=lambda record, current_history: completed_episodes.append(
            (record.episode_index, len(current_history.updates))
        ),
    )

    expected_generators = (
        np.random.default_rng(selection_seed),
        np.random.default_rng(selection_seed + 1),
    )
    expected_circuits = tuple(
        str(
            tracks[
                int(expected_generators[index % 2].integers(0, len(tracks)))
            ].track.generation.seed
        )
        for index in range(5)
    )
    assert tuple(record.circuit_identity for record in history.episodes) == (
        expected_circuits
    )
    assert history.training_interactions == 5
    assert len(history.updates) == 2
    assert [record.transition_count for record in history.updates] == [2, 2]
    assert completed_episodes == [(0, 1), (1, 1), (2, 2), (3, 2), (4, 2)]
    assert all(np.isfinite(record.mean_speed) for record in history.episodes)
    assert all(
        np.isfinite(record.mean_throttle_magnitude) for record in history.episodes
    )
    engine.close()


def test_a2c_engine_updates_full_and_final_short_rollouts() -> None:
    completed_episodes: list[tuple[int, int]] = []
    agent = A2CAgent(
        observation_dimensions=4,
        actor_config=_actor_config(),
        critic_config=CriticConfig(hidden_sizes=(4, 4)),
        config=A2CConfig(transitions_per_rollout=2),
        critic_learning_rate=0.01,
        actor_initialization_generator=torch.Generator().manual_seed(1),
        critic_initialization_generator=torch.Generator().manual_seed(2),
        sampling_generator=(
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
        ),
    )
    engine = A2CTrainingEngine(
        agent,
        _tracks(),
        _environment_config(),
        _normalizer(),
        (np.random.default_rng(4), np.random.default_rng(5)),
        (np.random.default_rng(6), np.random.default_rng(7)),
        execution_config=ExecutionConfig(device="cpu", environment_workers=2),
    )

    history = engine.train(
        3,
        on_episode_end=lambda record, current_history: completed_episodes.append(
            (record.episode_index, len(current_history.episodes))
        ),
    )

    assert history.training_interactions == 3
    assert len(history.episodes) == 3
    assert [record.transition_count for record in history.updates] == [2, 1]
    assert completed_episodes == [(0, 1), (1, 2), (2, 3)]
    assert all(np.isfinite(record.mean_speed) for record in history.episodes)
    assert all(
        np.isfinite(record.mean_throttle_magnitude) for record in history.episodes
    )
    engine.close()


def test_ppo_engine_updates_full_and_final_short_rollouts() -> None:
    completed_episodes: list[tuple[int, int]] = []
    agent = PPOAgent(
        observation_dimensions=4,
        actor_config=_actor_config(),
        critic_config=CriticConfig(hidden_sizes=(4, 4)),
        config=PPOConfig(
            transitions_per_rollout=2,
            optimization_epochs=1,
            minibatch_size=2,
        ),
        critic_learning_rate=0.01,
        actor_initialization_generator=torch.Generator().manual_seed(1),
        critic_initialization_generator=torch.Generator().manual_seed(2),
        sampling_generator=(
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
        ),
        optimization_generator=torch.Generator().manual_seed(4),
    )
    engine = PPOTrainingEngine(
        agent,
        _tracks(),
        _environment_config(),
        _normalizer(),
        (np.random.default_rng(5), np.random.default_rng(6)),
        (np.random.default_rng(7), np.random.default_rng(8)),
        execution_config=ExecutionConfig(device="cpu", environment_workers=2),
    )

    history = engine.train(
        3,
        on_episode_end=lambda record, current_history: completed_episodes.append(
            (record.episode_index, len(current_history.episodes))
        ),
    )

    assert history.training_interactions == 3
    assert len(history.episodes) == 3
    assert [record.transition_count for record in history.updates] == [2, 1]
    assert completed_episodes == [(0, 1), (1, 2), (2, 3)]
    assert all(np.isfinite(record.mean_speed) for record in history.episodes)
    assert all(
        np.isfinite(record.mean_throttle_magnitude) for record in history.episodes
    )
    engine.close()
