"""Tests for collision, finish and episode outcome rules."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from configs import SimulationConfig
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
    Build a circular track with a canonical gate at its eastern point.
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
        x=float(position[0]), y=float(position[1]), heading=0.0, speed=0.0
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


def _advance_to(
    lifecycle: EpisodeLifecycle,
    position: np.ndarray,
) -> None:
    """
    Advance a test transition whose four substeps all end at one position.
    """
    state = _state(position)
    lifecycle.process_transition(_transition(state, state, state, state))


def test_collision_is_checked_after_each_physics_substep(
    circle_geometry: TrackWithGeometry,
) -> None:
    """
    A point on the boundary crashes at the substep where it reaches it.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    start = _state(circle_geometry.position(0.0))
    on_track = _state(circle_geometry.position(1.0))
    boundary = _state(circle_geometry.position(1.0) + 7.0 * circle_geometry.normal(1.0))
    lifecycle.reset(start)

    result = lifecycle.process_transition(
        _transition(on_track, boundary, boundary, boundary)
    )

    assert result.collision
    assert result.terminated
    assert not result.truncated
    assert result.collision_substep == 2


def test_reset_gate_crossing_does_not_finish(
    circle_geometry: TrackWithGeometry,
) -> None:
    """
    Moving forward from the reset pose cannot immediately complete a lap.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))

    _advance_to(lifecycle, circle_geometry.position(10.0))

    assert lifecycle.episode_progress > 0.0
    assert lifecycle.episode_progress < circle_geometry.track.track_length


def test_insufficient_and_backward_gate_crossings_do_not_finish(
    circle_geometry: TrackWithGeometry,
) -> None:
    """
    Only a near-full forward lap can complete the episode.
    """
    insufficient = EpisodeLifecycle(circle_geometry)
    insufficient.reset(_state(circle_geometry.position(630.0)))
    _advance_to(insufficient, circle_geometry.position(0.0))

    backward = EpisodeLifecycle(circle_geometry)
    backward.reset(_state(circle_geometry.position(0.0)))
    _advance_to(backward, circle_geometry.position(630.0))

    assert insufficient.episode_progress == pytest.approx(10.0)
    assert not insufficient.process_transition(
        _transition(_state(circle_geometry.position(0.0)))
    ).lap_completed
    assert backward.episode_progress == pytest.approx(-10.0)


def test_forward_full_lap_crossing_terminates(
    circle_geometry: TrackWithGeometry,
) -> None:
    """
    Accumulated forward progress plus a gate crossing completes a lap.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    for s in range(10, 640, 10):
        _advance_to(lifecycle, circle_geometry.position(float(s)))

    result = lifecycle.process_transition(
        _transition(_state(circle_geometry.position(0.0)))
    )

    assert result.lap_completed
    assert result.terminated
    assert result.reward == pytest.approx(100.0 + 100.0 * (1.0 - 64 / 1_000))


def test_collision_takes_precedence_over_finish(
    circle_geometry: TrackWithGeometry,
) -> None:
    """
    Crossing the gate while leaving the track receives the crash outcome.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    for s in range(10, 640, 10):
        _advance_to(lifecycle, circle_geometry.position(float(s)))
    off_track_gate = circle_geometry.position(0.0) + 6.1 * circle_geometry.normal(0.0)

    result = lifecycle.process_transition(_transition(_state(off_track_gate)))

    assert result.collision
    assert not result.lap_completed
    assert result.reward == pytest.approx(-5.0)


def test_time_limit_truncates_without_termination(
    circle_geometry: TrackWithGeometry,
) -> None:
    """
    Reaching the time limit remains distinct from an MDP terminal state.
    """
    lifecycle = EpisodeLifecycle(
        circle_geometry,
        simulation_config=SimulationConfig(max_episode_steps=1),
    )
    start = _state(circle_geometry.position(0.0))
    lifecycle.reset(start)

    result = lifecycle.process_transition(_transition(start, start, start, start))

    assert result.truncated
    assert not result.terminated
    assert result.reward == pytest.approx(-0.04)
