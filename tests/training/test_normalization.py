from __future__ import annotations

import numpy as np
import pytest

from configs.training import ObservationNormalizationConfig
from training.normalization import RunningObservationNormalizer


def _normalizer() -> RunningObservationNormalizer:
    return RunningObservationNormalizer(
        observation_dimensions=2,
        config=ObservationNormalizationConfig(),
    )


def test_training_observation_updates_before_normalization() -> None:
    normalizer = _normalizer()

    first = normalizer.update_and_normalize([2.0, 10.0])
    second = normalizer.update_and_normalize([4.0, 14.0])

    np.testing.assert_array_equal(first, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(second, np.array([1.0, 1.0], dtype=np.float32))
    assert normalizer.count == 2
    assert normalizer.sums.dtype == np.float64
    assert normalizer.squared_sums.dtype == np.float64


def test_frozen_normalization_and_evaluation_leave_state_unchanged() -> None:
    normalizer = _normalizer()
    normalizer.update_and_normalize([1.0, 3.0])
    normalizer.update_and_normalize([3.0, 7.0])
    before = normalizer.state()
    before_checksum = normalizer.checksum()

    normalized = normalizer.normalize([5.0, 11.0])

    np.testing.assert_allclose(normalized, np.array([3.0, 3.0], dtype=np.float32))
    assert normalizer.state() == before
    assert normalizer.checksum() == before_checksum


def test_restore_reproduces_outputs_and_checksum() -> None:
    normalizer = _normalizer()
    normalizer.update_and_normalize([1.0, 2.0])
    normalizer.update_and_normalize([3.0, 8.0])
    state = normalizer.state()
    expected = normalizer.normalize([5.0, 14.0])
    expected_checksum = normalizer.checksum()

    restored = _normalizer()
    restored.restore(state)

    np.testing.assert_array_equal(restored.normalize([5.0, 14.0]), expected)
    assert restored.state() == state
    assert restored.checksum() == expected_checksum


def test_normalized_values_are_float32_and_clipped() -> None:
    normalizer = _normalizer()

    normalized = normalizer.normalize([100.0, -100.0])

    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized, np.array([10.0, -10.0], dtype=np.float32))


def test_batch_update_uses_all_active_rows_before_normalizing() -> None:
    normalizer = _normalizer()
    observations = np.asarray(((1.0, 2.0), (3.0, 4.0), (9.0, 9.0)))

    normalized = normalizer.update_and_normalize_batch(
        observations,
        active=np.asarray((True, True, False)),
    )

    assert normalizer.count == 2
    np.testing.assert_allclose(normalized[:2], np.asarray(((-1.0, -1.0), (1.0, 1.0))))


def test_normalizer_rejects_observations_with_wrong_shape() -> None:
    normalizer = _normalizer()

    with pytest.raises(ValueError, match="observation must have shape"):
        normalizer.update_and_normalize([1.0])
