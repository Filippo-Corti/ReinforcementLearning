"""Pygame presentation of the racing track and current vehicle state."""

from __future__ import annotations

from math import cos, sin

import numpy as np
import pygame

from .dynamics import VehicleState
from .geometry import TrackGeometry


class RacingRenderer:
    """
    Draw a racing track and a heading-visible point-car marker with Pygame.
    The camera fits the immutable track boundaries, so drawing has no effect on
    simulation state or episode outcomes.

    Fields:
        * geometry: The track geometry to draw.
        * render_mode: Whether to update a window or return RGB image data.
        * image_size: The output width and height in pixels.
    """

    def __init__(
        self,
        geometry: TrackGeometry,
        *,
        render_mode: str,
        image_size: tuple[int, int],
    ) -> None:
        self.geometry = geometry
        self.render_mode = render_mode
        self.image_size = image_size
        self._surface = pygame.Surface(image_size)
        self._window: pygame.Surface | None = None
        self._origin, self._scale = self._camera()

    def render(self, state: VehicleState) -> np.ndarray | None:
        """
        Draw the current state and update the requested render target.
        """
        self._surface.fill((34, 105, 54))
        self._draw_track()
        self._draw_finish_gate()
        self._draw_car(state)
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

    def _camera(self) -> tuple[np.ndarray, float]:
        """
        Return the boundary-fitting camera origin and uniform display scale.
        """
        boundaries = np.vstack(
            (self.geometry.left_boundary, self.geometry.right_boundary)
        )
        minimum = np.min(boundaries, axis=0)
        maximum = np.max(boundaries, axis=0)
        span = maximum - minimum
        available = np.asarray(self.image_size, dtype=np.float64) - 80.0
        scale = float(np.min(available / span))
        return (minimum + maximum) / 2.0, scale

    def _screen_point(self, point: np.ndarray) -> tuple[int, int]:
        """
        Convert a Cartesian point to a Pygame pixel location.
        """
        centered = (point - self._origin) * self._scale
        return (
            round(self.image_size[0] / 2.0 + centered[0]),
            round(self.image_size[1] / 2.0 - centered[1]),
        )

    def _screen_points(self, points: np.ndarray) -> list[tuple[int, int]]:
        """
        Convert Cartesian polyline points to Pygame pixel locations.
        """
        return [self._screen_point(point) for point in points]

    def _draw_track(self) -> None:
        """
        Draw road fill, boundaries and the centerline.
        """
        road = np.vstack(
            (self.geometry.left_boundary, self.geometry.right_boundary[::-1])
        )
        pygame.draw.polygon(self._surface, (68, 68, 72), self._screen_points(road))
        pygame.draw.lines(
            self._surface,
            (235, 235, 235),
            True,
            self._screen_points(self.geometry.left_boundary),
            width=3,
        )
        pygame.draw.lines(
            self._surface,
            (235, 235, 235),
            True,
            self._screen_points(self.geometry.right_boundary),
            width=3,
        )
        centerline = np.column_stack((self.geometry.track.x, self.geometry.track.y))
        pygame.draw.lines(
            self._surface,
            (218, 180, 38),
            True,
            self._screen_points(centerline),
            width=1,
        )

    def _draw_finish_gate(self) -> None:
        """
        Draw the canonical finish-gate segment.
        """
        start_index = self.geometry.track.start_index
        pygame.draw.line(
            self._surface,
            (221, 65, 65),
            self._screen_point(self.geometry.right_boundary[start_index]),
            self._screen_point(self.geometry.left_boundary[start_index]),
            width=4,
        )

    def _draw_car(self, state: VehicleState) -> None:
        """
        Draw a triangular marker whose nose follows the vehicle heading.
        """
        center = np.asarray(self._screen_point(np.asarray([state.x, state.y])))
        forward = np.asarray([cos(state.heading), -sin(state.heading)])
        sideways = np.asarray([-forward[1], forward[0]])
        triangle = [
            center + 12.0 * forward,
            center - 8.0 * forward + 6.0 * sideways,
            center - 8.0 * forward - 6.0 * sideways,
        ]
        pygame.draw.polygon(
            self._surface,
            (46, 156, 230),
            [tuple(np.rint(point).astype(int)) for point in triangle],
        )

    def _ensure_window(self) -> None:
        """
        Create the human-mode Pygame window on first use.
        """
        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode(self.image_size)
            pygame.display.set_caption("Racing environment")
