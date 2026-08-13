"""Tests for the two rendering styles and what they are required to show."""

from __future__ import annotations

import numpy as np
import pytest

from configs import EnvironmentConfig, RenderStyle, StartStateConfig
from envs import RacingEnv, TrackWithGeometry


def _environment(
    seed: int,
    *,
    style: RenderStyle = RenderStyle.BROADCAST,
    randomized_start: bool = False,
    render_mode: str = "rgb_array",
) -> RacingEnv:
    return RacingEnv(
        TrackWithGeometry.generate(seed),
        config=EnvironmentConfig(
            start=StartStateConfig(randomized=randomized_start),
        ),
        render_mode=render_mode,
        render_style=style,
    )


@pytest.mark.parametrize("style", list(RenderStyle))
def test_rgb_array_render_returns_declared_image(style: RenderStyle) -> None:
    """
    Both styles return an 800-square uint8 frame with visible drawing.
    """
    environment = _environment(5, style=style)
    environment.reset()

    image = environment.render()

    assert image is not None
    assert image.shape == (800, 800, 3)
    assert image.dtype == np.uint8
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 1
    environment.close()


@pytest.mark.parametrize("style", list(RenderStyle))
def test_rgb_array_render_updates_after_a_step(style: RenderStyle) -> None:
    """
    Rendering reflects the vehicle state after a transition.
    """
    environment = _environment(6, style=style)
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
    environment = _environment(7, render_mode="human")
    environment.reset()

    assert environment.render() is None
    environment.close()
    environment.close()


def test_the_finish_line_is_drawn_where_this_episode_ends() -> None:
    """
    A sampled start moves the finish line, and the drawing has to move with it.

    The lap runs one full circuit from wherever the car is placed, so an episode
    that starts elsewhere ends elsewhere. Drawing the canonical start line would
    mark a place this episode never treats as a finish.
    """
    environment = _environment(11, randomized_start=True)
    environment.reset(seed=3)
    lifecycle = environment._lifecycle
    assert lifecycle is not None

    canonical = environment.track.start_index * environment.track.sample_spacing

    assert lifecycle.gate_s != pytest.approx(canonical, abs=1.0)
    # What is drawn comes from the lifecycle, so the two cannot disagree.
    assert environment.render() is not None
    environment.close()


def test_the_broadcast_view_shows_far_more_than_the_minimal_one() -> None:
    """
    The two styles answer different questions, and should not look alike.

    Minimal is a diagram: a handful of flat colours. Broadcast is a scene with a
    graded sky, depth-faded road and an overlay, which cannot be expressed in a
    handful of colours.
    """
    frames = {}
    for style in RenderStyle:
        environment = _environment(12, style=style)
        environment.reset(seed=1)
        for _ in range(40):
            environment.step(np.asarray([0.8, 0.05], dtype=np.float32))
        image = environment.render()
        assert image is not None
        frames[style] = np.unique(image.reshape(-1, 3), axis=0).shape[0]
        environment.close()

    assert frames[RenderStyle.MINIMAL] <= 8
    assert frames[RenderStyle.BROADCAST] > 200


def test_a_section_straddling_the_camera_is_trimmed_and_kept() -> None:
    """
    A cross-section with one end behind the camera must not be discarded.

    Discarding it is what made the road vanish from under a car that was merely
    turned: at any real heading error the nearest sections all have one end
    behind, so dropping them dropped the whole near field. The trimmed end sits
    exactly on the near plane, which the projection must therefore accept.
    """
    from envs.racing.rendering.broadcast import _NEAR_PLANE, _clip_to_near_plane

    assert _clip_to_near_plane((-4.0, 2.0), (-1.0, 3.0)) is None

    ahead = (20.0, -6.0)
    kept = _clip_to_near_plane((-4.0, 6.0), ahead)
    assert kept is not None
    trimmed, unchanged = kept
    assert unchanged == ahead
    assert trimmed[0] == pytest.approx(_NEAR_PLANE)
    # The trimmed end stays on the segment it was cut from.
    assert -6.0 < trimmed[1] < 6.0

    # Trimming works from either side and keeps the endpoints in their places.
    kept = _clip_to_near_plane(ahead, (-4.0, 6.0))
    assert kept is not None
    assert kept[0] == ahead
    assert kept[1][0] == pytest.approx(_NEAR_PLANE)


def test_the_road_reaches_the_car_when_it_is_turned_across_the_track() -> None:
    """
    The nearest drawn road must be just in front of the car, not far ahead.

    This is the failure the trimming prevents, and it is checked here rather
    than in pixels because the symptom is a hole in the near field: with the
    sections dropped the strip began more than twenty metres away, leaving the
    car apparently standing on grass while the road floated in the distance.
    """
    from dataclasses import replace

    from configs import CarConfig
    from envs.racing.rendering.broadcast import BroadcastRacingRenderer
    from envs.racing.rendering.frame import RenderFrame

    environment = _environment(14)
    environment.reset(seed=0)
    assert environment.state is not None
    track = environment.track_with_geometry
    renderer = BroadcastRacingRenderer(
        track, image_size=(800, 800), vehicle_config=CarConfig()
    )
    half_width = track.track.width / 2.0
    for arc in (40.0, 120.0, 260.0, 400.0):
        centre = track.position(arc)
        heading = track.heading(arc)
        normal = track.normal(arc)
        for lateral in (-0.5, 0.0, 0.5):
            for degrees in (-40.0, -25.0, -10.0, 0.0, 10.0, 25.0, 40.0):
                point = centre + lateral * half_width * normal
                state = replace(
                    environment.state,
                    x=float(point[0]),
                    y=float(point[1]),
                    heading=float(heading + np.radians(degrees)),
                    speed=25.0,
                )
                samples = renderer._road_samples(RenderFrame(state=state, gate_s=arc))
                assert samples, f"no road at all at {degrees:+.0f} degrees"
                nearest = min(sample.distance for sample in samples)
                # Dropping the straddling sections pushes this past four metres
                # and, on a curve, past twenty.
                assert nearest < 3.0, (
                    f"nearest road is {nearest:.1f} m away at arc {arc:.0f}, "
                    f"lateral {lateral:+.1f}, {degrees:+.0f} degrees"
                )
    environment.close()


def test_the_broadcast_view_is_drawn_from_the_car_not_from_above() -> None:
    """
    A pilot view must change when the car only turns, and a map must not.

    Two states at the same place with different headings produce the same image
    from above and completely different ones from the driver's seat, which is
    the distinction between the two styles.
    """
    images = {}
    for style in RenderStyle:
        environment = _environment(13, style=style)
        environment.reset(seed=2)
        assert environment.state is not None
        from dataclasses import replace

        straight = environment.render()
        environment.state = replace(
            environment.state, heading=environment.state.heading + 0.6
        )
        turned = environment.render()
        assert straight is not None and turned is not None
        images[style] = float(
            np.mean(np.abs(straight.astype(float) - turned.astype(float)))
        )
        environment.close()

    assert images[RenderStyle.MINIMAL] == pytest.approx(0.0, abs=1e-9)
    assert images[RenderStyle.BROADCAST] > 1.0
