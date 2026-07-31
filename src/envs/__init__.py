"""Racing environment, dynamics, geometry, and observation components."""

from .racing import EpisodeLifecycle, EpisodeTransition, RacingEnv, RacingRenderer
from .tracks import (
    FrenetProjection,
    FrenetProjector,
    PolylineProjector,
    SegmentProjection,
    Track,
    TrackGenerationError,
    TrackGenerationMetadata,
    TrackGeometry,
    TrackValidationError,
    generate_track,
    generate_track_file,
    signed_progress,
    validate_track_geometry,
    wrap_angle,
)
from .vehicle import (
    DynamicsTransition,
    NormalizedAction,
    PhysicalControls,
    VehicleState,
    map_action,
    transition,
)

__all__ = [
    "DynamicsTransition",
    "EpisodeLifecycle",
    "EpisodeTransition",
    "FrenetProjection",
    "FrenetProjector",
    "NormalizedAction",
    "PhysicalControls",
    "PolylineProjector",
    "RacingEnv",
    "RacingRenderer",
    "SegmentProjection",
    "Track",
    "TrackGenerationError",
    "TrackGenerationMetadata",
    "TrackGeometry",
    "TrackValidationError",
    "VehicleState",
    "generate_track",
    "generate_track_file",
    "map_action",
    "signed_progress",
    "transition",
    "validate_track_geometry",
    "wrap_angle",
]
