"""Tests for analysis plots built only from machine-readable table rows."""

from __future__ import annotations

import matplotlib.image as mpimg
import numpy as np

from utils.plotting import (
    plot_curvature_controls,
    plot_learning_curves,
    save_figure,
)


def test_learning_and_curvature_plots_preserve_expected_series(tmp_path) -> None:
    curve_rows = [
        {
            "algorithm": "ppo",
            "actor_name": "small",
            "observation_type": "frenet",
            "root_identity": root,
            "training_interactions": interaction,
            "mean_return": value + root,
            "mean_progress": value / 10,
        }
        for root in (0, 1)
        for interaction, value in ((10, 1.0), (20, 2.0))
    ]
    learning = plot_learning_curves(curve_rows)

    assert len(learning.axes) == 2
    assert [
        np.asarray(line.get_xdata()).tolist() for line in learning.axes[0].lines
    ] == [[10.0, 20.0]]
    learning_path = tmp_path / "learning.png"
    save_figure(learning, learning_path)
    assert mpimg.imread(learning_path).size > 0

    curvature_rows = [
        {
            "algorithm": "ppo",
            "actor_name": "small",
            "observation_type": "frenet",
            "curvature_bin": curvature_bin,
            "outcome": outcome,
            "sample_count": 2,
            "mean_speed": 10.0 + index,
            "mean_throttle": 0.5 - index / 10,
            "mean_absolute_steering": index / 10,
        }
        for index, curvature_bin in enumerate(("q1", "q2", "q3", "q4"))
        for outcome in ("completed", "crashed")
    ]
    curvature = plot_curvature_controls(curvature_rows)

    assert len(curvature.axes) == 3
    assert np.asarray(curvature.axes[0].lines[0].get_xdata()).tolist() == [
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    save_figure(curvature, tmp_path / "curvature.png")
