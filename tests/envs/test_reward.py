"""Tests for documented reward reference values."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from configs import RewardConfig, SimulationConfig
from envs import (
    EpisodeLifecycle,
    KinematicTransition,
    PhysicalControls,
    Track,
    TrackGenerationMetadata,
    TrackWithGeometry,
    VehicleState,
)


@pytest.fixture(scope="module")
def circle_geometry() -> TrackWithGeometry:
    """
    Build a circular track for reward reference checks.
    """
    sample_count = 64
    spacing = 10.0
    length = sample_count * spacing
    radius = length / (2.0 * pi)
    angles = np.arange(sample_count, dtype=np.float64) * 2.0 * pi / sample_count
    return TrackWithGeometry(
        Track(
            generation=TrackGenerationMetadata(
                seed=0,
                n_checkpoints=12,
                base_radius=radius,
                radial_jitter=0.0,
                angular_jitter=0.0,
                max_attempts=1,
            ),
            width=12.0,
            sample_spacing=spacing,
            track_length=length,
            start_index=0,
            s=np.arange(sample_count, dtype=np.float64) * spacing,
            x=radius * np.cos(angles),
            y=radius * np.sin(angles),
            heading=angles + pi / 2.0,
            curvature=np.full(sample_count, 1.0 / radius, dtype=np.float64),
        )
    )


def _state(position: np.ndarray) -> VehicleState:
    """
    Construct a stationary state at a Cartesian position.
    """
    return VehicleState(
        x=float(position[0]),
        y=float(position[1]),
        heading=0.0,
        speed=0.0,
    )


def _transition(*states: VehicleState) -> KinematicTransition:
    """
    Construct a transition with explicit substep states for lifecycle testing.
    """
    return KinematicTransition(
        state=states[-1],
        substep_states=tuple(states),
        controls=PhysicalControls(acceleration=0.0, steering_angle=0.0),
    )


def _stationary_total(circle_geometry, steps: int) -> float:
    """
    Return the accumulated reward of remaining stationary for some agent steps.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    state = _state(circle_geometry.position(0.0))
    lifecycle.reset(state)
    return sum(
        lifecycle.process_transition(_transition(state, state, state, state)).reward
        for _ in range(steps)
    )


def _partial_lap_total(circle_geometry, distance: float) -> float:
    """
    Return the accumulated reward of driving a distance and then leaving the track.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    total = 0.0
    for travelled in np.arange(10.0, distance + 10.0, 10.0):
        state = _state(circle_geometry.position(float(travelled)))
        total += lifecycle.process_transition(_transition(state)).reward
    crashed = _state(
        circle_geometry.position(distance) + 7.0 * circle_geometry.normal(distance)
    )
    return total + lifecycle.process_transition(_transition(crashed)).reward


def test_target_lap_step_cost_matches_documented_total(circle_geometry) -> None:
    """
    Eighteen seconds of non-terminal stationary steps costs minus nine.
    """
    assert _stationary_total(circle_geometry, 450) == pytest.approx(-9.0)


def test_stationary_timeout_matches_documented_total(circle_geometry) -> None:
    """
    Remaining stationary through 1,000 agent steps returns minus twenty.
    """
    assert _stationary_total(circle_geometry, 1_000) == pytest.approx(-20.0)


def test_crashing_anywhere_beats_never_leaving_the_start_line(circle_geometry) -> None:
    """
    Trying to drive must outscore idling, or standing still is the only optimum.

    The crash penalty has to stay below the cost of idling to the time limit.
    When it does not, every exploratory attempt to drive is a losing bet against
    doing nothing, and every policy-gradient method collapses onto a stalled
    policy regardless of its own update rule.
    """
    idling = _stationary_total(circle_geometry, 1_000)

    assert _partial_lap_total(circle_geometry, 10.0) > idling
    assert _partial_lap_total(circle_geometry, 320.0) > idling


def test_driving_further_before_crashing_scores_strictly_higher(
    circle_geometry,
) -> None:
    """
    The dense progress term must dominate the one-off terminal penalty.

    Without this ordering the return is effectively a function of "did I crash"
    alone, so no algorithm can learn to reach further along the circuit.
    """
    totals = [
        _partial_lap_total(circle_geometry, distance)
        for distance in (40.0, 160.0, 320.0, 480.0)
    ]

    assert totals == sorted(totals)
    assert totals[-1] - totals[0] > abs(-5.0)


def test_normalized_progress_dominates_the_crash_penalty() -> None:
    """
    One lap of shaping must be worth far more than one terminal event.
    """
    reward = RewardConfig()
    idling_cost = (
        reward.time_penalty_rate
        * SimulationConfig().agent_timestep
        * SimulationConfig().max_episode_steps
    )

    assert reward.crash_penalty < idling_cost
    assert reward.progress_coefficient > 10.0 * reward.crash_penalty


def test_normalized_progress_still_sums_to_one_lap(circle_geometry) -> None:
    """
    Dividing signed distance by track length keeps one lap normalized to one.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    progress = 0.0

    for distance in range(10, 640, 10):
        state = _state(circle_geometry.position(float(distance)))
        progress += lifecycle.process_transition(_transition(state)).progress_delta
    finish = _state(circle_geometry.position(0.0))
    progress += lifecycle.process_transition(_transition(finish)).progress_delta

    assert progress / circle_geometry.track.track_length == pytest.approx(1.0)


def test_immediate_crash_matches_documented_penalty(circle_geometry) -> None:
    """
    Leaving the track on the first substep receives the crash penalty.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    crashed = _state(circle_geometry.position(0.0) + 7.0 * circle_geometry.normal(0.0))

    result = lifecycle.process_transition(_transition(crashed))

    assert result.reward == pytest.approx(-5.0)
