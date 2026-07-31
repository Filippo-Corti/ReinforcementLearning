"""Track data, generation, geometry, projection, and observations."""

from .generation import TrackGenerationError, generate_track, generate_track_file
from .geometry import TrackGeometry, wrap_angle
from .model import Track, TrackGenerationMetadata, TrackValidationError
from .observations import FrenetProjection, FrenetProjector, signed_progress
from .projection import PolylineProjector, SegmentProjection
from .validation import validate_track_geometry

__all__ = [
    "FrenetProjection",
    "FrenetProjector",
    "PolylineProjector",
    "SegmentProjection",
    "Track",
    "TrackGenerationError",
    "TrackGenerationMetadata",
    "TrackGeometry",
    "TrackValidationError",
    "generate_track",
    "generate_track_file",
    "signed_progress",
    "validate_track_geometry",
    "wrap_angle",
]
