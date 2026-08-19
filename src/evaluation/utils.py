"""Vehicle and circuit geometry read for logging, not for the policy itself."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.observations import FrenetObservation
from envs.racing import RacingEnv


@dataclass(frozen=True, slots=True)
class TrajectoryState:
    """
    Preserve pre-action geometry and vehicle state for one retained trajectory row.

    Fields:
        * position: Cartesian vehicle position.
        * heading: Vehicle heading in radians.
        * current_curvature: Centerline curvature at the closest projection.
        * preview_curvature: Frenet preview curvature when present.
        * speed: Vehicle speed.
        * lateral_acceleration_proxy: Speed squared times absolute curvature.
    """

    position: tuple[float, float]
    heading: float
    current_curvature: float
    preview_curvature: float | None
    speed: float
    lateral_acceleration_proxy: float


def trajectory_state(
    environment: RacingEnv, observation: np.ndarray
) -> TrajectoryState:
    """
    Read the current vehicle and projected circuit geometry before an action.
    """
    if environment.state is None:
        raise RuntimeError(
            "RacingEnv must have active vehicle state during evaluation."
        )
    state = environment.state
    projection = environment.track_with_geometry.centerline_projector.project(
        state.position()
    )
    s = float(
        environment.track.s[projection.segment_index]
        + projection.fraction
        * environment.track_with_geometry.centerline_projector.lengths[
            projection.segment_index
        ]
    )
    current_curvature = environment.track_with_geometry.curvature(s)
    preview_curvature = (
        float(observation[-1])
        if observation.shape == (FrenetObservation.DIMENSIONS,)
        else None
    )
    return TrajectoryState(
        position=(float(state.x), float(state.y)),
        heading=float(state.heading),
        current_curvature=current_curvature,
        preview_curvature=preview_curvature,
        speed=float(state.speed),
        lateral_acceleration_proxy=float(state.speed**2 * abs(current_curvature)),
    )
