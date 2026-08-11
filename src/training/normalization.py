"""Running observation normalization for policy and value-network inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from configs.training import ObservationNormalizationConfig
from recording.records import ObservationNormalizerStateRecord
from utils.vectors import to_vector


class RunningObservationNormalizer:
    """
    Normalize vector observations with componentwise float64 running sums.

    For component j after n training observations, the running mean is
    mean_j = sum_i(x_i,j) / n and the variance is
    variance_j = sum_i(x_i,j^2) / n - mean_j^2. Each input becomes
    normalized_x_j = (x_j - mean_j) / sqrt(variance_j + epsilon) and is clipped
    to the configured interval. Training observations update both sums before
    this formula is applied. Frozen normalization is used for bootstrap and evaluation
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
        vector = self._to_vector(observation)
        self.count += 1
        self.sums += vector
        self.squared_sums += np.square(vector)
        return self._apply_normalization(vector)

    def normalize(
        self, observation: Sequence[float] | NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """
        Normalize one bootstrap or evaluation observation without updating state.
        """
        return self._apply_normalization(self._to_vector(observation))

    def update_and_normalize_batch(
        self,
        observations: Sequence[Sequence[float]] | NDArray[np.floating],
        active: Sequence[bool] | NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        """
        Update from active synchronous observations, then normalize the batch.

        All active rows enter the running sums before any row is normalized, so
        changing worker iteration order cannot change network inputs.
        """
        vectors = np.asarray(observations, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != self.observation_dimensions:
            raise ValueError(
                "Observation batches must have shape (rows, observation_dimensions)."
            )
        active_mask = (
            np.ones(vectors.shape[0], dtype=np.bool_)
            if active is None
            else np.asarray(active, dtype=np.bool_)
        )
        if active_mask.shape != (vectors.shape[0],):
            raise ValueError("The active mask must contain one value per batch row.")
        selected = vectors[active_mask]
        self.count += int(selected.shape[0])
        self.sums += np.sum(selected, axis=0)
        self.squared_sums += np.sum(np.square(selected), axis=0)
        return np.stack(
            [self._apply_normalization(vector) for vector in vectors]
        ).astype(np.float32, copy=False)

    def state(self) -> ObservationNormalizerStateRecord:
        """
        Return an immutable copy of the current running-sum state.
        """
        return ObservationNormalizerStateRecord(
            count=self.count,
            sums=tuple(float(value) for value in self.sums),
            squared_sums=tuple(float(value) for value in self.squared_sums),
        )

    def restore(self, state: ObservationNormalizerStateRecord) -> None:
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
        Return a stable SHA-256 identity for the current normalization state.

        Evaluation records save this short identity so two results can be
        checked to have used exactly the same count and running sums without
        duplicating the full arrays in every record.
        """
        encoded_state = json.dumps(
            self.state().to_dict(), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(encoded_state).hexdigest()

    def _apply_normalization(self, vector: NDArray[np.float64]) -> NDArray[np.float32]:
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

    def _to_vector(
        self, observation: Sequence[float] | NDArray[np.floating]
    ) -> NDArray[np.float64]:
        return to_vector(
            observation,
            name="observation",
            dimensions=self.observation_dimensions,
            dtype="float64",
        )
