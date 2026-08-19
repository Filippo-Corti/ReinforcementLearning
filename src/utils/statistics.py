"""General-purpose scalar summaries shared across recorded signals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from recording.records import ScalarSummaryRecord


def scalar_summary(values: Sequence[float]) -> ScalarSummaryRecord:
    """
    Return a population scalar summary for one non-empty sequence of values.
    """
    array = np.asarray(values, dtype=np.float64)
    return ScalarSummaryRecord(
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        quantiles={
            "q25": float(np.quantile(array, 0.25)),
            "q50": float(np.quantile(array, 0.50)),
            "q75": float(np.quantile(array, 0.75)),
            "q90": float(np.quantile(array, 0.90)),
        },
    )
