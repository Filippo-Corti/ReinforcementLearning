"""Tests for deterministic random and scripted driving references."""

from __future__ import annotations

import numpy as np

from envs.racing import RacingEnv
from envs.tracks import TrackWithGeometry, generate_track
from utils.metrics import EpisodeOutcome, RunCategory
from utils.references import (
    ScriptedFrenetController,
    evaluate_reference,
    random_action_reference,
)
from utils.seeding import RunSeedStreams, SeedNamespace


def test_scripted_controller_uses_the_documented_frenet_formula() -> None:
    action = ScriptedFrenetController().action(
        np.asarray((2.0, 0.5, 10.0, 0.01), dtype=np.float32)
    )

    assert np.allclose(action, np.asarray((1.0, -0.2), dtype=np.float32))


def test_random_reference_actions_are_reproducible_from_evaluation_stream() -> None:
    first = random_action_reference(
        RunSeedStreams(SeedNamespace.REDUCED_BUDGET_VALIDATION, 7)
    )
    second = random_action_reference(
        RunSeedStreams(SeedNamespace.REDUCED_BUDGET_VALIDATION, 7)
    )
    observation = np.zeros(4, dtype=np.float32)

    assert np.array_equal(first.action(observation), second.action(observation))


def test_scripted_reference_completes_the_deterministic_reference_track() -> None:
    environment = RacingEnv(TrackWithGeometry(generate_track(0)))
    result = evaluate_reference(
        environment,
        ScriptedFrenetController(),
        run_category=RunCategory.REDUCED_VALIDATION,
    )
    environment.close()

    assert result.episode.outcome is EpisodeOutcome.COMPLETED
    assert result.episode.maximum_progress >= 0.99
    assert result.episode.speed is not None
    assert result.episode.throttle is not None
    assert result.transitions[0].next_observation
    assert result.transitions[-1].lap_completed
