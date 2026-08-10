"""Deterministic, independent random streams for a training run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from configs.training import ExecutionConfig

if TYPE_CHECKING:
    import torch

ROOT_PROTOCOL_KEY = 20_260_810


class SeedNamespace(IntEnum):
    """
    Stable namespace codes separating experiment and development randomness.
    """

    EXPERIMENT_1_REPORTED = 10
    EXPERIMENT_2_REPORTED = 11
    LEARNING_RATE_CALIBRATION = 20
    CAPABILITY_AND_GRIP_DIAGNOSIS = 21
    REDUCED_BUDGET_VALIDATION = 30
    CONTROLLED_PROBLEM_VALIDATION = 31
    EXPERIMENT_1_CIRCUIT_CANDIDATE = 40
    MULTI_CIRCUIT_DEVELOPMENT = 50
    EXPERIMENT_2_TRAINING_TRACK = 51
    EXPERIMENT_2_VALIDATION_TRACK = 52
    EXPERIMENT_2_TEST_TRACK = 53
    EXECUTION_ORDER = 60


class SeedStream(IntEnum):
    """
    Fixed child indices for independent sources of run randomness.
    """

    ACTOR_INITIALIZATION = 0
    CRITIC_INITIALIZATION = 1
    POLICY_ACTIONS = 2
    ENVIRONMENT_RESET = 3
    TRACK_SCHEDULE = 4
    MINIBATCHES = 5
    EVALUATION_REFERENCE = 6


@dataclass(frozen=True, slots=True)
class RunSeedStreams:
    """
    Stateless access to deterministic child streams for one logical run.

    Fields:
        * namespace: Namespace that identifies the run category.
        * local_identity: Logical identity within that namespace.
    """

    namespace: SeedNamespace
    local_identity: int

    def numpy_generator(self, stream: SeedStream) -> np.random.Generator:
        """
        Return a fresh, independent NumPy generator for one named job.
        """
        return np.random.default_rng(self.seed_sequence(stream))

    def integer_seed(self, stream: SeedStream) -> int:
        """
        Return a stable uint32 seed suitable for PyTorch or an environment.
        """
        return int(self.seed_sequence(stream).generate_state(1, dtype=np.uint32)[0])

    def torch_generator(
        self, stream: SeedStream, device: str = "cpu"
    ) -> torch.Generator:
        """
        Return a fresh seeded PyTorch generator without changing global RNG state.
        """
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(self.integer_seed(stream))
        return generator

    def root_seed_sequence(self) -> np.random.SeedSequence:
        """
        Return the run root without spawning or consuming any child stream.
        """
        return np.random.SeedSequence(
            entropy=[ROOT_PROTOCOL_KEY, int(self.namespace), self.local_identity]
        )

    def seed_sequence(self, stream: SeedStream) -> np.random.SeedSequence:
        """
        Return a named child SeedSequence without consuming any sibling stream.
        """
        root = self.root_seed_sequence()
        return np.random.SeedSequence(
            entropy=root.entropy,
            spawn_key=(*root.spawn_key, int(stream)),
        )


def track_seed(namespace: SeedNamespace, *identity: int) -> int:
    """
    Return a stable track-generator seed for a documented logical identity.
    """
    return int(
        np.random.SeedSequence(
            entropy=[ROOT_PROTOCOL_KEY, int(namespace), *identity]
        ).generate_state(1, dtype=np.uint32)[0]
    )


@dataclass(frozen=True, slots=True)
class TorchDeterminismState:
    """
    Settings observed immediately after deterministic PyTorch configuration.

    Fields:
        * device: Requested torch device policy.
        * intraop_threads: Configured PyTorch intra-operation thread count.
        * interop_threads: Configured PyTorch inter-operation thread count.
        * deterministic_algorithms: Whether PyTorch requires deterministic kernels.
        * deterministic_warn_only: Whether nondeterministic kernels warn instead of error.
        * cudnn_benchmark: Whether cuDNN algorithm benchmarking is enabled.
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
    Apply the documented global PyTorch deterministic execution policy.

    This operation intentionally changes PyTorch process-wide settings. It must
    run before parallel PyTorch work: PyTorch can reject a late inter-op thread
    change. Deterministic algorithms still cannot make results identical across
    PyTorch releases, platforms, CPU versus GPU, or unsupported kernels.
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
