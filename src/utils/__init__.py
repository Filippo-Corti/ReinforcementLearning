"""Small helpers shared across otherwise independent project domains."""

from .random import (
    ROOT_PROTOCOL_KEY,
    RunSeedStreams,
    SeedNamespace,
    SeedStream,
    TorchDeterminismState,
    configure_torch_determinism,
)
from .vectors import optional_scalar, optional_tensor, to_vector

__all__ = [
    "ROOT_PROTOCOL_KEY",
    "RunSeedStreams",
    "SeedNamespace",
    "SeedStream",
    "TorchDeterminismState",
    "configure_torch_determinism",
    "optional_scalar",
    "optional_tensor",
    "to_vector",
]
