"""Small helpers shared across otherwise independent project domains."""

from .random import (
    ROOT_PROTOCOL_KEY,
    RunSeedStreams,
    SeedNamespace,
    SeedStream,
    TorchDeterminismState,
    configure_torch_determinism,
)
from .statistics import scalar_summary
from .vectors import optional_scalar, optional_tensor, to_tensor, to_vector

# `evaluate_policy_episode` is deliberately not re-exported here. It depends on
# the `evaluation` package, which depends on `circuits` and `normalization` to
# build the environments it evaluates — both of which depend on this package
# for small helpers of their own. Importing it eagerly in this file would
# require `utils` to finish loading before `utils` finishes loading. Importing
# `utils.policy_evaluation` directly, once every package involved has loaded,
# does not.
__all__ = [
    "ROOT_PROTOCOL_KEY",
    "RunSeedStreams",
    "SeedNamespace",
    "SeedStream",
    "TorchDeterminismState",
    "configure_torch_determinism",
    "optional_scalar",
    "optional_tensor",
    "scalar_summary",
    "to_tensor",
    "to_vector",
]
