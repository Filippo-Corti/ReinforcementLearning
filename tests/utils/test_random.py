"""Tests for reproducible, independent project random streams."""

from __future__ import annotations

import numpy as np
import torch

from configs import ExecutionConfig
from utils.random import (
    ROOT_PROTOCOL_KEY,
    RunSeedStreams,
    SeedNamespace,
    SeedStream,
    configure_torch_determinism,
)


def test_equal_run_identities_reproduce_every_named_stream() -> None:
    first = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 0)
    second = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 0)

    for stream in SeedStream:
        assert np.array_equal(
            first.get_numpy_generator(stream).integers(
                0, 2**32, size=8, dtype=np.uint32
            ),
            second.get_numpy_generator(stream).integers(
                0, 2**32, size=8, dtype=np.uint32
            ),
        )
        assert torch.equal(
            torch.rand(8, generator=first.get_torch_generator(stream)),
            torch.rand(8, generator=second.get_torch_generator(stream)),
        )


def test_named_streams_are_independent_and_do_not_consume_each_other() -> None:
    streams = RunSeedStreams(SeedNamespace.EXPERIMENT_2_REPORTED, 3)
    expected = streams.get_numpy_generator(SeedStream.POLICY_ACTION_SAMPLING).integers(
        0, 100, 16
    )

    streams.get_numpy_generator(SeedStream.EVALUATION).random(100)
    torch.rand(100, generator=streams.get_torch_generator(SeedStream.EVALUATION))
    actual = streams.get_numpy_generator(SeedStream.POLICY_ACTION_SAMPLING).integers(
        0, 100, 16
    )

    assert np.array_equal(actual, expected)
    first_values = {
        int(streams.get_numpy_generator(stream).integers(0, 2**32))
        for stream in SeedStream
    }
    assert len(first_values) == len(SeedStream)


def test_different_run_and_track_identities_change_their_streams() -> None:
    first = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 0)
    second = RunSeedStreams(SeedNamespace.EXPERIMENT_1_REPORTED, 1)
    assert not torch.equal(
        torch.rand(
            8, generator=first.get_torch_generator(SeedStream.ACTOR_INITIALIZATION)
        ),
        torch.rand(
            8, generator=second.get_torch_generator(SeedStream.ACTOR_INITIALIZATION)
        ),
    )

    track_four = RunSeedStreams(SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 4)
    repeated_track_four = RunSeedStreams(SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 4)
    track_five = RunSeedStreams(SeedNamespace.EXPERIMENT_2_TRAINING_TRACK, 5)
    first_seed = int(
        track_four.get_numpy_generator(SeedStream.TRACK_GENERATION).integers(0, 2**32)
    )
    assert first_seed == int(
        repeated_track_four.get_numpy_generator(SeedStream.TRACK_GENERATION).integers(
            0, 2**32
        )
    )
    assert first_seed != int(
        track_five.get_numpy_generator(SeedStream.TRACK_GENERATION).integers(0, 2**32)
    )


def test_generator_derivation_does_not_change_global_rng_state() -> None:
    np.random.seed(91)
    torch.manual_seed(37)
    expected_numpy = np.random.random(10)
    expected_torch = torch.rand(10)
    np.random.seed(91)
    torch.manual_seed(37)

    streams = RunSeedStreams(SeedNamespace.LEARNING_RATE_CALIBRATION, 2)
    for stream in SeedStream:
        streams.get_numpy_generator(stream).random(5)
        torch.rand(5, generator=streams.get_torch_generator(stream))

    assert ROOT_PROTOCOL_KEY == 0
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
