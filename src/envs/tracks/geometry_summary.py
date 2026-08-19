"""A frozen circuit's geometry, summarized for later stratified analysis."""

from __future__ import annotations

import numpy as np

from recording.records import CircuitGeometrySummaryRecord, ScalarSummaryRecord

from .track import Track


def track_geometry_summary(track: Track) -> CircuitGeometrySummaryRecord:
    """
    Summarize one track's length and curvature distribution.
    """
    absolute_curvature = np.abs(track.curvature)
    return CircuitGeometrySummaryRecord(
        track_length=float(track.track_length),
        absolute_curvature=ScalarSummaryRecord(
            mean=float(np.mean(absolute_curvature)),
            standard_deviation=float(np.std(absolute_curvature)),
            minimum=float(np.min(absolute_curvature)),
            maximum=float(np.max(absolute_curvature)),
            quantiles={
                "q25": float(np.quantile(absolute_curvature, 0.25)),
                "q50": float(np.quantile(absolute_curvature, 0.50)),
                "q75": float(np.quantile(absolute_curvature, 0.75)),
                "q90": float(np.quantile(absolute_curvature, 0.90)),
            },
        ),
    )
