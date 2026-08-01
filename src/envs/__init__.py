"""Racing environment, dynamics, geometry, tracks, and observations."""

from .geometry import (
    PolylineProjector,
    SegmentProjection,
    project_to_segment,
    segment_distance,
    segments_intersect,
    wrap_angle,
)
from .observations import (
    FrenetObservation,
    FrenetObserver,
    FrenetProjection,
    signed_progress,
)
from .racing import (
    ActionOutcome,
    ActionType,
    EpisodeLifecycle,
    ObservationType,
    RacingEnv,
    RacingPygameRenderer,
)
from .tracks import (
    Track,
    TrackGenerationError,
    TrackGenerationMetadata,
    TrackValidationError,
    TrackWithGeometry,
    generate_track,
    generate_track_file,
    validate_track_geometry,
)
from .vehicle import (
    KinematicTransition,
    NormalizedAction,
    PhysicalControls,
    VehicleState,
    normalized_to_physical_controls,
    transition,
)

__all__ = [
    "ActionOutcome",
    "ActionType",
    "EpisodeLifecycle",
    "FrenetObservation",
    "FrenetObserver",
    "FrenetProjection",
    "KinematicTransition",
    "NormalizedAction",
    "ObservationType",
    "PhysicalControls",
    "PolylineProjector",
    "RacingEnv",
    "RacingPygameRenderer",
    "SegmentProjection",
    "Track",
    "TrackGenerationError",
    "TrackGenerationMetadata",
    "TrackValidationError",
    "TrackWithGeometry",
    "VehicleState",
    "generate_track",
    "generate_track_file",
    "normalized_to_physical_controls",
    "project_to_segment",
    "segment_distance",
    "segments_intersect",
    "signed_progress",
    "transition",
    "validate_track_geometry",
    "wrap_angle",
]
