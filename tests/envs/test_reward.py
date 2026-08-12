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


def _idling_cost() -> float:
    """
    Return what standing still until the time limit costs.
    """
    simulation = SimulationConfig()
    return (
        RewardConfig().time_penalty_rate
        * simulation.agent_timestep
        * simulation.max_episode_steps
    )


def _stationary_total(circle_geometry) -> float:
    """
    Return the accumulated reward of standing still until the episode ends.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    state = _state(circle_geometry.position(0.0))
    lifecycle.reset(state)
    total = 0.0
    for _ in range(SimulationConfig().max_episode_steps):
        outcome = lifecycle.process_transition(_transition(state, state, state, state))
        total += outcome.reward
        if outcome.terminated or outcome.truncated:
            return total
    raise AssertionError("A stationary episode must reach a boundary.")


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


def _completed_lap_total(circle_geometry, steps_per_sample: int) -> float:
    """
    Return the reward of one full lap driven at a chosen number of steps.

    Each agent action advances the same distance, so a larger step count is a
    slower lap over exactly the same path.
    """
    length = circle_geometry.track.track_length
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    total = 0.0
    for step in range(1, steps_per_sample + 1):
        state = _state(circle_geometry.position(length * step / steps_per_sample))
        outcome = lifecycle.process_transition(_transition(state))
        total += outcome.reward
        if outcome.lap_completed:
            return total
    raise AssertionError("A full lap must cross the finish gate.")


def test_stationary_episode_costs_the_full_time_limit(circle_geometry) -> None:
    """
    Standing still is charged the whole episode clock even when it ends early.

    The stall rule exists to stop simulating a car that has given up, not to
    make giving up cheap. Ending the episode early must therefore leave the
    return identical to idling all the way to the time limit.
    """
    assert _stationary_total(circle_geometry) == pytest.approx(-_idling_cost())


def test_crashing_anywhere_beats_never_leaving_the_start_line(circle_geometry) -> None:
    """
    Trying to drive must outscore idling, or standing still is the only optimum.

    The crash penalty has to stay below the cost of idling to the time limit.
    When it does not, every exploratory attempt to drive is a losing bet against
    doing nothing, and every policy-gradient method collapses onto a stalled
    policy regardless of its own update rule.
    """
    idling = _stationary_total(circle_geometry)

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


def test_a_fast_lap_scores_clearly_higher_than_a_slow_one(circle_geometry) -> None:
    """
    Lap time must be a visible share of the return, or the task is not racing.

    Progress and the per-step time penalty alone leave a crawling lap worth
    almost as much as an attacking one, so the completion reward carries the
    difference. The step counts below span the range real policies drive in, and
    a lap taking three times as long must lose a fifth of its return.
    """
    fast = _completed_lap_total(circle_geometry, 250)
    slow = _completed_lap_total(circle_geometry, 750)

    assert slow < fast
    assert (fast - slow) / fast > 0.2


def test_finishing_slowly_still_beats_crashing_immediately(circle_geometry) -> None:
    """
    No lap-time pressure may ever make abandoning the lap the better option.
    """
    slowest_lap = _completed_lap_total(circle_geometry, 512)

    assert slowest_lap > -RewardConfig().crash_penalty
    assert slowest_lap > -_idling_cost()


def test_normalized_progress_dominates_the_crash_penalty() -> None:
    """
    One lap of shaping must be worth far more than one terminal event.
    """
    reward = RewardConfig()

    assert reward.crash_penalty < _idling_cost()
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
