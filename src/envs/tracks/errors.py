"""Track-specific validation and generation errors."""


class TrackValidationError(ValueError):
    """
    Raised when track data or derived geometry violates the track specification.
    """


class TrackGenerationError(RuntimeError):
    """
    Raised when all deterministic generation attempts are rejected.
    """
