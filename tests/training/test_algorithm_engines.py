"""Each algorithm's engine driven by its real agent, not a stub.

`test_engine.py` pins the shared lifecycle against a fixed agent, which makes
its assertions exact but says nothing about whether a real actor-critic can be
driven through these loops. These tests close that gap: they check that each
engine updates when its own collection strategy says it should, using the agent
the reported runs use.
"""

from __future__ import annotations

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
    StartStateConfig,
)
from envs.observations import FrenetObservation
from envs.tracks import TrackWithGeometry
from recording import RunCategory
from training import (
    A2CTrainingEngine,
    PPOTrainingEngine,
    ReinforceTrainingEngine,
    RunningObservationNormalizer,
    TrainingCircuitSchedule,
)
from training.engines import TrainingEngine


def _track() -> TrackWithGeometry:
    source = Path(__file__).parents[1] / "fixtures" / "tracks" / "valid_circle.json"
    return TrackWithGeometry.load(source)


def _config() -> EnvironmentConfig:
    return EnvironmentConfig(
        simulation=SimulationConfig(max_episode_steps=1),
        start=StartStateConfig(randomized=False),
    )


def _actor_config() -> ActorConfig:
    return ActorConfig(name="small", hidden_sizes=(4, 4), learning_rate=0.01)


def _build(engine_type: type[TrainingEngine], agent, workers: int, **extra):
    return engine_type(
        agent,
        _track(),
        _config(),
        RunningObservationNormalizer(
            FrenetObservation.DIMENSIONS, ObservationNormalizationConfig()
        ),
        run_category=RunCategory.REDUCED_VALIDATION,
        environment_reset_generators=tuple(
            np.random.default_rng(3 + index) for index in range(workers)
        ),
        track_selection_generators=tuple(
            np.random.default_rng(19 + index) for index in range(workers)
        ),
        execution_config=ExecutionConfig(device="cpu", environment_workers=workers),
        **extra,
    )


def test_reinforce_updates_once_its_episode_batch_is_complete() -> None:
    """
    Two workers, two episodes per update: one update per pair of episodes.
    """
    agent = ReinforceAgent(
        observation_dimensions=FrenetObservation.DIMENSIONS,
        actor_config=_actor_config(),
        config=ReinforceConfig(completed_episodes_per_update=2),
        initialization_generator=torch.Generator().manual_seed(1),
        sampling_generator=(
            torch.Generator().manual_seed(2),
            torch.Generator().manual_seed(3),
        ),
    )
    engine = _build(ReinforceTrainingEngine, agent, workers=2)

    state = engine.train(4)

    assert state.counters.training_interactions == 4
    assert state.counters.finished_episodes == 4
    assert state.counters.optimizer_updates == 2
    engine.close()


def test_reinforce_fills_a_batch_larger_than_its_worker_count() -> None:
    """
    A batch may need more trajectories than there are workers.

    The worker count is an execution choice shared with A2C and PPO, while the
    batch size belongs to the algorithm, so the engine fills one batch over as
    many collection waves as it needs and updates only once it is complete.
    """
    agent = ReinforceAgent(
        observation_dimensions=FrenetObservation.DIMENSIONS,
        actor_config=_actor_config(),
        config=ReinforceConfig(completed_episodes_per_update=4),
        initialization_generator=torch.Generator().manual_seed(1),
        sampling_generator=(
            torch.Generator().manual_seed(2),
            torch.Generator().manual_seed(3),
        ),
    )
    engine = _build(ReinforceTrainingEngine, agent, workers=2)

    state = engine.train(8)

    assert state.counters.training_interactions == 8
    assert state.counters.finished_episodes == 8
    assert state.counters.optimizer_updates == 2
    assert [record.episode_index for record in engine.episode_records] == list(range(8))
    engine.close()


def test_a2c_updates_on_a_full_rollout_and_on_a_final_short_one() -> None:
    agent = A2CAgent(
        observation_dimensions=FrenetObservation.DIMENSIONS,
        actor_config=_actor_config(),
        critic_config=CriticConfig(hidden_sizes=(4, 4), learning_rate=0.01),
        config=A2CConfig(transitions_per_rollout=2),
        actor_initialization_generator=torch.Generator().manual_seed(1),
        critic_initialization_generator=torch.Generator().manual_seed(2),
        sampling_generator=(
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
        ),
    )
    engine = _build(A2CTrainingEngine, agent, workers=2)

    state = engine.train(3)

    assert state.counters.training_interactions == 3
    assert state.counters.finished_episodes == 3
    # Two transitions fill a rollout; the third is learned from as a remainder.
    assert state.counters.optimizer_updates == 2
    assert [update.training_interactions for update in engine.updates] == [2, 3]
    engine.close()


def test_ppo_updates_on_a_full_rollout_and_on_a_final_short_one() -> None:
    agent = PPOAgent(
        observation_dimensions=FrenetObservation.DIMENSIONS,
        actor_config=_actor_config(),
        critic_config=CriticConfig(hidden_sizes=(4, 4), learning_rate=0.01),
        config=PPOConfig(
            transitions_per_rollout=2, optimization_epochs=1, minibatch_size=2
        ),
        actor_initialization_generator=torch.Generator().manual_seed(1),
        critic_initialization_generator=torch.Generator().manual_seed(2),
        sampling_generator=(
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
        ),
        optimization_generator=torch.Generator().manual_seed(4),
    )
    engine = _build(PPOTrainingEngine, agent, workers=2)

    state = engine.train(3)

    assert state.counters.training_interactions == 3
    assert state.counters.optimizer_updates == 2
    assert [update.training_interactions for update in engine.updates] == [2, 3]
    engine.close()


def test_every_engine_records_the_circuits_a_schedule_gives_it() -> None:
    """
    The circuit schedule is shared, so every engine must honour it identically.

    A2C and PPO collect the same way and REINFORCE does not, which is exactly
    why this is worth checking on all three rather than on one.
    """
    for engine_type, agent in (
        (
            ReinforceTrainingEngine,
            ReinforceAgent(
                observation_dimensions=FrenetObservation.DIMENSIONS,
                actor_config=_actor_config(),
                config=ReinforceConfig(completed_episodes_per_update=2),
                initialization_generator=torch.Generator().manual_seed(1),
                sampling_generator=(
                    torch.Generator().manual_seed(2),
                    torch.Generator().manual_seed(3),
                ),
            ),
        ),
        (
            A2CTrainingEngine,
            A2CAgent(
                observation_dimensions=FrenetObservation.DIMENSIONS,
                actor_config=_actor_config(),
                critic_config=CriticConfig(hidden_sizes=(4, 4), learning_rate=0.01),
                config=A2CConfig(transitions_per_rollout=2),
                actor_initialization_generator=torch.Generator().manual_seed(1),
                critic_initialization_generator=torch.Generator().manual_seed(2),
                sampling_generator=(
                    torch.Generator().manual_seed(3),
                    torch.Generator().manual_seed(4),
                ),
            ),
        ),
        (
            PPOTrainingEngine,
            PPOAgent(
                observation_dimensions=FrenetObservation.DIMENSIONS,
                actor_config=_actor_config(),
                critic_config=CriticConfig(hidden_sizes=(4, 4), learning_rate=0.01),
                config=PPOConfig(
                    transitions_per_rollout=2, optimization_epochs=1, minibatch_size=2
                ),
                actor_initialization_generator=torch.Generator().manual_seed(1),
                critic_initialization_generator=torch.Generator().manual_seed(2),
                sampling_generator=(
                    torch.Generator().manual_seed(3),
                    torch.Generator().manual_seed(4),
                ),
                optimization_generator=torch.Generator().manual_seed(4),
            ),
        ),
    ):
        engine = _build(
            engine_type,
            agent,
            workers=2,
            training_circuit_schedule=TrainingCircuitSchedule(),
        )
        engine.train(4)
        records = engine.episode_records

        assert {record.circuit_split for record in records} == {"training"}
        assert len({record.circuit_identity for record in records}) == len(records)
        engine.close()
