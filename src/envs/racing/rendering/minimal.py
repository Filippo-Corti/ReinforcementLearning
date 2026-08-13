"""The plainest possible view: the circuit, and a dot for the car."""

from __future__ import annotations

import numpy as np
import pygame

from ...tracks import TrackWithGeometry
from .frame import RenderFrame, TrackCamera

_BACKGROUND = (18, 20, 24)
_ROAD = (58, 60, 66)
_EDGE = (150, 152, 158)
_CAR = (240, 96, 72)
_GATE = (235, 235, 235)


class MinimalRacingRenderer:
    """
    Draw the circuit from above with a single dot for the car.

    Nothing here is meant to look like anything. It exists so that a frame can
    be read at a glance, compared between runs, and produced cheaply when what
    is being inspected is the trajectory rather than the driving.

    Fields:
        * track: The sampled circuit being drawn.
        * image_size: Output width and height in pixels.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        image_size: tuple[int, int],
    ) -> None:
        self.track = track
        self.image_size = image_size
        self._camera = TrackCamera(track, image_size)
        self._road = self._camera.road_quads(track)
        self._left = self._camera.points(track.left_boundary)
        self._right = self._camera.points(track.right_boundary)

    def draw(self, surface: pygame.Surface, frame: RenderFrame) -> None:
        """
        Draw the circuit, this episode's finish line, and the car.
        """
        surface.fill(_BACKGROUND)
        for quad in self._road:
            pygame.draw.polygon(surface, _ROAD, quad)
        pygame.draw.lines(surface, _EDGE, True, self._left, width=2)
        pygame.draw.lines(surface, _EDGE, True, self._right, width=2)

        gate = self.track.position(frame.gate_s)
        normal = self.track.normal(frame.gate_s)
        half_width = self.track.track.width / 2.0
        pygame.draw.line(
            surface,
            _GATE,
            self._camera.point(gate + half_width * normal),
            self._camera.point(gate - half_width * normal),
            width=3,
        )

        pygame.draw.circle(
            surface,
            _CAR,
            self._camera.point(np.asarray([frame.state.x, frame.state.y])),
            5,
        )
