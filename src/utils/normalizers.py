"""Running observation normalization for policy and value-network inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from configs.training import ObservationNormalizationConfig


@dataclass(frozen=True, slots=True)
class ObservationNormalizerState:
    """
    Store the complete deterministic state of an observation normalizer.

    Fields:
        * count: Number of observations incorporated in the running sums.
        * sums: Componentwise sum of incorporated observations.
        * squared_sums: Componentwise sum of squared incorporated observations.
    """

    count: int
    sums: tuple[float, ...]
    squared_sums: tuple[float, ...]

    def to_dict(self) -> dict[str, int | list[float]]:
        """
        Return a JSON-compatible state with a stable field order.
        """
        return {
            "count": self.count,
            "sums": list(self.sums),
            "squared_sums": list(self.squared_sums),
        }


class RunningObservationNormalizer:
    """
    Normalize vector observations with componentwise float64 running sums.

    Training observations update the count, sum, and squared sum before they
    are normalized. Frozen normalization is used for bootstrap and evaluation
    observations, so those operations cannot change later training inputs.

    Fields:
        * observation_dimensions: Number of components in every observation.
        * config: Numerical safeguards specified by the training contract.
        * count: Number of training observations incorporated so far.
        * sums: Componentwise float64 running sums.
        * squared_sums: Componentwise float64 running squared sums.
    """

    def __init__(
        self,
        observation_dimensions: int,
        config: ObservationNormalizationConfig,
    ) -> None:
        """
        Initialize empty componentwise running sums.
        """
        self.observation_dimensions = observation_dimensions
        self.config = config
        self.count = 0
        self.sums = np.zeros(observation_dimensions, dtype=np.float64)
        self.squared_sums = np.zeros(observation_dimensions, dtype=np.float64)

    def update_and_normalize(
        self, observation: Sequence[float] | NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """
        Update from one current training observation, then normalize it.
        """
        vector = self._vector(observation)
        self.count += 1
        self.sums += vector
        self.squared_sums += np.square(vector)
        return self._normalize(vector)

    def normalize_frozen(
        self, observation: Sequence[float] | NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """
        Normalize one bootstrap or evaluation observation without updating state.
        """
        return self._normalize(self._vector(observation))

    def state(self) -> ObservationNormalizerState:
        """
        Return an immutable copy of the current running-sum state.
        """
        return ObservationNormalizerState(
            count=self.count,
            sums=tuple(float(value) for value in self.sums),
            squared_sums=tuple(float(value) for value in self.squared_sums),
        )

    def restore(self, state: ObservationNormalizerState) -> None:
        """
        Restore a previously saved state for the same observation dimension.
        """
        if len(state.sums) != self.observation_dimensions:
            raise ValueError(
                "Normalizer state has incompatible observation dimensions."
            )
        if len(state.squared_sums) != self.observation_dimensions:
            raise ValueError(
                "Normalizer state has incompatible observation dimensions."
            )
        if state.count < 0:
            raise ValueError("Normalizer count cannot be negative.")
        self.count = state.count
        self.sums = np.asarray(state.sums, dtype=np.float64).copy()
        self.squared_sums = np.asarray(state.squared_sums, dtype=np.float64).copy()

    def checksum(self) -> str:
        """
        Return a stable checksum for checkpoint and evaluation provenance.
        """
        encoded_state = json.dumps(
            self.state().to_dict(), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(encoded_state).hexdigest()

    def _normalize(self, vector: NDArray[np.float64]) -> NDArray[np.float32]:
        if self.count == 0:
            mean = np.zeros(self.observation_dimensions, dtype=np.float64)
            variance = np.zeros(self.observation_dimensions, dtype=np.float64)
        else:
            mean = self.sums / self.count
            variance = np.maximum(self.squared_sums / self.count - np.square(mean), 0.0)
        normalized = (vector - mean) / np.sqrt(variance + self.config.variance_epsilon)
        return np.clip(
            normalized,
            -self.config.normalized_value_limit,
            self.config.normalized_value_limit,
        ).astype(np.float32)

    def _vector(
        self, observation: Sequence[float] | NDArray[np.floating]
    ) -> NDArray[np.float64]:
        vector = np.asarray(observation, dtype=np.float64)
        if vector.shape != (self.observation_dimensions,):
            raise ValueError(
                "Expected observation shape "
                f"({self.observation_dimensions},), received {vector.shape}."
            )
        return vector
