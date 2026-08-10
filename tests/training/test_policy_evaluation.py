"""Tests for deterministic random and scripted baseline policies."""

from __future__ import annotations

import numpy as np

from envs.racing import RacingEnv
from envs.tracks import TrackWithGeometry, generate_track
from models import Policy, RandomPolicy, ScriptedFrenetPolicy
from recording import EpisodeOutcome, RunCategory
from training.policy_evaluation import evaluate_policy_episode
from utils.random import RunSeedStreams, SeedNamespace, SeedStream


def test_scripted_policy_uses_the_documented_frenet_formula() -> None:
    policy = ScriptedFrenetPolicy()

    assert isinstance(policy, Policy)
    action = policy.action(np.asarray((2.0, 0.5, 10.0, 0.01), dtype=np.float32))
    assert np.allclose(action, np.asarray((1.0, -0.2), dtype=np.float32))


def test_random_policy_actions_are_reproducible_from_evaluation_stream() -> None:
    streams = RunSeedStreams(SeedNamespace.REDUCED_BUDGET_VALIDATION, 7)
    first = RandomPolicy(streams.get_numpy_generator(SeedStream.EVALUATION))
    second = RandomPolicy(streams.get_numpy_generator(SeedStream.EVALUATION))
    observation = np.zeros(4, dtype=np.float32)

    assert isinstance(first, Policy)
    assert np.array_equal(first.action(observation), second.action(observation))


def test_scripted_policy_completes_the_deterministic_baseline_track() -> None:
    environment = RacingEnv(TrackWithGeometry(generate_track(0)))
    result = evaluate_policy_episode(
        environment,
        ScriptedFrenetPolicy(),
        run_category=RunCategory.REDUCED_VALIDATION,
    )
    environment.close()

    assert result.episode.outcome is EpisodeOutcome.COMPLETED
    assert result.episode.maximum_progress >= 0.99
    assert result.episode.speed is not None
    assert result.episode.throttle is not None
    assert result.transitions[0].next_observation
    assert result.transitions[-1].lap_completed
