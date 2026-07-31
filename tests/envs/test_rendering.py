"""Tests for Pygame rendering modes."""

from __future__ import annotations

import numpy as np
import pytest

from envs import RacingEnv


def test_rgb_array_render_returns_declared_image() -> None:
    """
    RGB rendering returns an 800-square uint8 frame with visible drawing.
    """
    environment = RacingEnv(track_seed=5, render_mode="rgb_array")
    environment.reset()

    image = environment.render()

    assert image is not None
    assert image.shape == (800, 800, 3)
    assert image.dtype == np.uint8
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 1
    environment.close()


def test_rgb_array_render_updates_after_a_step() -> None:
    """
    Rendering reflects the vehicle state after a transition.
    """
    environment = RacingEnv(track_seed=6, render_mode="rgb_array")
    environment.reset()
    before = environment.render()
    for _ in range(20):
        environment.step(np.asarray([1.0, 0.0], dtype=np.float32))
    after = environment.render()

    assert before is not None
    assert after is not None
    assert not np.array_equal(before, after)
    environment.close()


def test_human_rendering_smoke_test_releases_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Human-mode rendering opens and closes through Pygame's dummy video driver.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    environment = RacingEnv(track_seed=7, render_mode="human")
    environment.reset()

    assert environment.render() is None
    environment.close()
    environment.close()
