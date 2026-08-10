"""Tests for reproducible, independent project seed streams."""

from __future__ import annotations

import numpy as np
import torch

from configs import ExecutionConfig
from utils.seeding import (
    ROOT_PROTOCOL_KEY,
    RunSeedStreams,
    SeedNamespace,
    SeedStream,
    configure_torch_determinism,
    track_seed,
)


def test_equal_run_identities_reproduce_every_named_stream() -> None:
    first = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 0)
    second = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 0)

    for stream in SeedStream:
        assert first.integer_seed(stream) == second.integer_seed(stream)
        assert np.array_equal(
            first.numpy_generator(stream).integers(0, 2**32, size=8, dtype=np.uint32),
            second.numpy_generator(stream).integers(0, 2**32, size=8, dtype=np.uint32),
        )


def test_named_streams_are_independent_and_do_not_consume_each_other() -> None:
    streams = RunSeedStreams(SeedNamespace.EXPERIMENT_2_REPORTED, 3)
    expected = streams.numpy_generator(SeedStream.POLICY_ACTIONS).integers(0, 100, 16)

    streams.numpy_generator(SeedStream.EVALUATION_REFERENCE).random(100)
    torch.rand(
        100,
        generator=streams.torch_generator(SeedStream.EVALUATION_REFERENCE),
    )
    actual = streams.numpy_generator(SeedStream.POLICY_ACTIONS).integers(0, 100, 16)

    assert np.array_equal(actual, expected)
    assert len({streams.integer_seed(stream) for stream in SeedStream}) == len(
        SeedStream
    )


def test_different_root_identities_change_derived_streams() -> None:
    first = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 0)
    second = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 1)

    assert first.integer_seed(SeedStream.ACTOR_INITIALIZATION) != second.integer_seed(
        SeedStream.ACTOR_INITIALIZATION
    )
    assert track_seed(SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 0, 4) == track_seed(
        SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 0, 4
    )
    assert track_seed(SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 0, 4) != track_seed(
        SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 0, 5
    )


def test_seed_derivation_does_not_change_global_numpy_or_torch_rng_state() -> None:
    np.random.seed(91)
    torch.manual_seed(37)
    expected_numpy = np.random.random(10)
    expected_torch = torch.rand(10)
    np.random.seed(91)
    torch.manual_seed(37)

    streams = RunSeedStreams(SeedNamespace.LEARNING_RATE_CALIBRATION, 2)
    for stream in SeedStream:
        streams.integer_seed(stream)
        streams.numpy_generator(stream).random(5)
        torch.rand(5, generator=streams.torch_generator(stream))
    track_seed(SeedNamespace.EXPERIMENT_1_CIRCUIT_CANDIDATE, 8)

    assert ROOT_PROTOCOL_KEY == 20_260_810
    assert np.array_equal(expected_numpy, np.random.random(10))
    assert torch.equal(expected_torch, torch.rand(10))


def test_torch_determinism_configuration_applies_documented_global_policy() -> None:
    state = configure_torch_determinism(ExecutionConfig(device="cpu"))

    assert state.device == "cpu"
    assert state.intraop_threads == 1
    assert state.interop_threads == 1
    assert state.deterministic_algorithms
    assert not state.deterministic_warn_only
    assert not state.cudnn_benchmark
    assert state.cudnn_deterministic
