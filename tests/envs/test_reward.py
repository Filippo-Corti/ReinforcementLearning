"""Tests for documented reward reference values."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

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


def test_stationary_timeout_matches_documented_total(circle_geometry) -> None:
    """
    Remaining stationary through 5,000 agent steps returns minus ten.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    state = _state(circle_geometry.position(0.0))
    lifecycle.reset(state)

    total = sum(
        lifecycle.process_transition(_transition(state, state, state, state)).reward
        for _ in range(5_000)
    )

    assert total == pytest.approx(-10.0)


def test_immediate_crash_matches_documented_penalty(circle_geometry) -> None:
    """
    Leaving the track on the first substep receives the crash penalty.
    """
    lifecycle = EpisodeLifecycle(circle_geometry)
    lifecycle.reset(_state(circle_geometry.position(0.0)))
    crashed = _state(circle_geometry.position(0.0) + 7.0 * circle_geometry.normal(0.0))

    result = lifecycle.process_transition(_transition(crashed))

    assert result.reward == pytest.approx(-20.0)
