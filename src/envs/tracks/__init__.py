"""Track data, preparation, generation, persistence, and validation."""

from .errors import TrackGenerationError, TrackValidationError
from .generation import generate_track, generate_track_file
from .geometry_summary import track_geometry_summary
from .track import Track, TrackGenerationMetadata, TrackWithGeometry
from .validation import validate_track_geometry

__all__ = [
    "Track",
    "TrackGenerationError",
    "TrackGenerationMetadata",
    "TrackValidationError",
    "TrackWithGeometry",
    "generate_track",
    "generate_track_file",
    "track_geometry_summary",
    "validate_track_geometry",
]
