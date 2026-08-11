"""Tests for persistent process-based racing environment collection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from configs import EnvironmentConfig, ExecutionConfig, SimulationConfig
from envs.tracks import TrackWithGeometry
from training import PersistentRacingVectorEnv
from utils.random import RunSeedStreams, SeedNamespace, SeedStream


def _tracks() -> tuple[TrackWithGeometry, TrackWithGeometry]:
    source = Path(__file__).parents[1] / "fixtures" / "tracks" / "valid_circle.json"
    first = TrackWithGeometry.load(source)
    second = TrackWithGeometry(
        replace(
            first.track,
            generation=replace(first.track.generation, seed=1),
        )
    )
    return first, second


def _pool() -> PersistentRacingVectorEnv:
    streams = RunSeedStreams(SeedNamespace.REDUCED_BUDGET_VALIDATION, 7)
    worker_count = 2
    return PersistentRacingVectorEnv(
        _tracks(),
        EnvironmentConfig(simulation=SimulationConfig(max_episode_steps=2)),
        ExecutionConfig(device="cpu", environment_workers=worker_count),
        tuple(
            streams.get_numpy_generator(
                SeedStream.ENVIRONMENT_RESETS,
                substream_identity=index,
            )
            for index in range(worker_count)
        ),
        tuple(
            streams.get_numpy_generator(
                SeedStream.TRAINING_TRACK_SELECTION,
                substream_identity=index,
            )
            for index in range(worker_count)
        ),
    )


def test_workers_persist_and_apply_deterministic_torch_settings() -> None:
    with _pool() as pool:
        pool.reset()
        initial_pids = pool.worker_pids
        pool.step(np.zeros((2, 2), dtype=np.float32))
        pool.step(np.zeros((2, 2), dtype=np.float32))

        assert pool.worker_pids == initial_pids
        assert len(set(initial_pids)) == 2
        assert all(
            state.intraop_threads == 1 for state in pool.worker_torch_determinism
        )
        assert all(
            state.interop_threads == 1 for state in pool.worker_torch_determinism
        )
        assert all(
            state.deterministic_algorithms for state in pool.worker_torch_determinism
        )
        assert all(not state.cudnn_benchmark for state in pool.worker_torch_determinism)


def test_only_selected_workers_reset_and_parked_workers_do_not_advance() -> None:
    with _pool() as pool:
        observations, _ = pool.reset()
        first_next, _, _, _, _ = pool.step(
            np.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=np.float32)
        )
        pool.reset(np.asarray((True, False), dtype=np.bool_))
        after_step, rewards, terminated, truncated, infos = pool.step(
            np.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=np.float32),
            np.asarray((True, False), dtype=np.bool_),
        )

        assert not np.array_equal(first_next[0], observations[0])
        np.testing.assert_array_equal(after_step[1], first_next[1])
        assert rewards[1] == 0.0
        assert not terminated[1]
        assert not truncated[1]
        assert not bool(infos["transition_valid"][1])


def test_vector_state_restores_worker_and_scheduler_progress() -> None:
    actions = np.asarray(((0.5, 0.1), (0.2, -0.1)), dtype=np.float32)
    with _pool() as pool:
        pool.reset()
        pool.step(actions)
        state = pool.state()
        expected = pool.step(actions)
        pool.restore(state)
        actual = pool.step(actions)

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        np.testing.assert_array_equal(actual[3], expected[3])
