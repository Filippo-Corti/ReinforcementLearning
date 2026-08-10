"""Deterministic independent random generators for one logical run.

A single global generator makes unrelated sources of randomness sensitive to
each other's call order. Here, a run is identified by a namespace and local
identity, while each purpose receives its own stable child stream. Requesting
one generator therefore never consumes or changes another stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from configs.training import ExecutionConfig

if TYPE_CHECKING:
    import torch

ROOT_PROTOCOL_KEY = 0


class SeedNamespace(IntEnum):
    """
    Separate reported experiments, calibration, and development randomness.
    """

    EXPERIMENT_1_REPORTED = 1
    EXPERIMENT_2_REPORTED = 2
    LEARNING_RATE_CALIBRATION = 3
    CAPABILITY_AND_GRIP_DIAGNOSIS = 4
    REDUCED_BUDGET_VALIDATION = 5
    CONTROLLED_PROBLEM_VALIDATION = 6
    EXPERIMENT_1_CIRCUIT_CANDIDATE = 7
    MULTI_CIRCUIT_DEVELOPMENT = 8
    EXPERIMENT_2_TRAINING_TRACK = 9
    EXPERIMENT_2_VALIDATION_TRACK = 10
    EXPERIMENT_2_TEST_TRACK = 11
    EXECUTION_ORDER = 12


class SeedStream(IntEnum):
    """
    Identify each independent source of randomness within one run.
    """

    ACTOR_INITIALIZATION = 1
    CRITIC_INITIALIZATION = 2
    POLICY_ACTION_SAMPLING = 3
    ENVIRONMENT_RESETS = 4
    TRAINING_TRACK_SELECTION = 5
    OPTIMIZATION_BATCH_ORDER = 6
    EVALUATION = 7
    TRACK_GENERATION = 8


@dataclass(frozen=True, slots=True)
class RunSeedStreams:
    """
    Provide fresh NumPy and PyTorch generators for one logical run.

    The stream seed is derived from
    `(ROOT_PROTOCOL_KEY, namespace, local_identity, stream)`. A fresh generator
    restarts that named stream; callers that need sequential draws should retain
    the returned generator.

    Fields:
        * namespace: Category that separates logically different uses of seeds.
        * local_identity: Run or track identity inside that category.
    """

    namespace: SeedNamespace
    local_identity: int

    def get_numpy_generator(self, stream: SeedStream) -> np.random.Generator:
        """
        Return a fresh NumPy generator for one named source of randomness.
        """
        return np.random.default_rng(self._seed_sequence(stream))

    def get_torch_generator(
        self, stream: SeedStream, device: str = "cpu"
    ) -> torch.Generator:
        """
        Return a fresh local PyTorch generator without changing global state.
        """
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(self._integer_seed(stream))
        return generator

    def _seed_sequence(self, stream: SeedStream) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            entropy=[ROOT_PROTOCOL_KEY, int(self.namespace), self.local_identity],
            spawn_key=(int(stream),),
        )

    def _integer_seed(self, stream: SeedStream) -> int:
        return int(self._seed_sequence(stream).generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True, slots=True)
class TorchDeterminismState:
    """
    Record PyTorch settings observed after deterministic configuration.

    Fields:
        * device: Requested torch device policy.
        * intraop_threads: Configured intra-operation thread count.
        * interop_threads: Configured inter-operation thread count.
        * deterministic_algorithms: Whether deterministic kernels are required.
        * deterministic_warn_only: Whether nondeterministic kernels only warn.
        * cudnn_benchmark: Whether cuDNN benchmarking is enabled.
        * cudnn_deterministic: Whether cuDNN deterministic mode is enabled.
    """

    device: str
    intraop_threads: int
    interop_threads: int
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool


def configure_torch_determinism(
    execution: ExecutionConfig,
) -> TorchDeterminismState:
    """
    Apply the configured process-wide PyTorch deterministic execution policy.

    This must run before parallel PyTorch work because inter-op thread settings
    may become immutable. Deterministic kernels do not promise identical values
    across PyTorch versions, platforms, or CPU and GPU implementations.
    """
    import torch

    if torch.get_num_threads() != execution.intraop_threads:
        torch.set_num_threads(execution.intraop_threads)
    if torch.get_num_interop_threads() != execution.interop_threads:
        try:
            torch.set_num_interop_threads(execution.interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "Configure PyTorch inter-op threads before creating parallel work."
            ) from error

    torch.use_deterministic_algorithms(
        execution.deterministic_algorithms,
        warn_only=execution.deterministic_warn_only,
    )
    torch.backends.cudnn.benchmark = execution.cudnn_benchmark
    torch.backends.cudnn.deterministic = execution.deterministic_algorithms

    return TorchDeterminismState(
        device=execution.device,
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
    )
