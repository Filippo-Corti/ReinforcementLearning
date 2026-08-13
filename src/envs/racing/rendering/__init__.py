"""Presentation of the racing environment, in one of two styles."""

from __future__ import annotations

import numpy as np
import pygame

from configs import CarConfig, RenderStyle

from ...tracks import TrackWithGeometry
from ...vehicle import VehicleState
from .broadcast import BroadcastRacingRenderer
from .frame import RenderFrame, TrackCamera
from .minimal import MinimalRacingRenderer

__all__ = [
    "BroadcastRacingRenderer",
    "MinimalRacingRenderer",
    "RacingPygameRenderer",
    "RenderFrame",
    "TrackCamera",
]


class RacingPygameRenderer:
    """
    Own the drawing surface and hand each frame to the chosen style.

    Which style is drawing changes nothing the environment does: both are given
    the same immutable frame and neither can reach back into the simulation.

    Fields:
        * track: The sampled track and derived geometry to draw.
        * render_mode: Whether to update a window or return RGB image data.
        * image_size: The output width and height in pixels.
        * style: Which of the two presentations is drawing.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        render_mode: str,
        image_size: tuple[int, int],
        style: RenderStyle = RenderStyle.BROADCAST,
        vehicle_config: CarConfig | None = None,
    ) -> None:
        self.track = track
        self.render_mode = render_mode
        self.image_size = image_size
        self.style = RenderStyle(style)
        self._surface = pygame.Surface(image_size)
        self._window: pygame.Surface | None = None
        self._renderer: BroadcastRacingRenderer | MinimalRacingRenderer = (
            MinimalRacingRenderer(track, image_size=image_size)
            if self.style is RenderStyle.MINIMAL
            else BroadcastRacingRenderer(
                track,
                image_size=image_size,
                vehicle_config=vehicle_config,
            )
        )

    def render(self, frame: RenderFrame | VehicleState) -> np.ndarray | None:
        """
        Draw one frame and update the requested render target.
        """
        drawn = (
            frame
            if isinstance(frame, RenderFrame)
            else RenderFrame(
                state=frame,
                gate_s=float(
                    self.track.track.start_index * self.track.track.sample_spacing
                ),
            )
        )
        self._renderer.draw(self._surface, drawn)
        if self.render_mode == "human":
            self._ensure_window()
            if self._window is None:
                raise RuntimeError("Pygame window initialization failed.")
            pygame.event.pump()
            self._window.blit(self._surface, (0, 0))
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def close(self) -> None:
        """
        Release the renderer's Pygame display resources.
        """
        if self._window is not None:
            pygame.display.quit()
            self._window = None

    def _ensure_window(self) -> None:
        """
        Create the human-mode Pygame window on first use.
        """
        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode(self.image_size)
            pygame.display.set_caption("Racing environment")
