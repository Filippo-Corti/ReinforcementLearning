"""What a renderer is told about the moment it is drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...tracks import TrackWithGeometry
from ...vehicle import VehicleState


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """
    Carry one instant of the episode to whichever renderer is drawing it.

    Rendering reads this and nothing else, so no drawing code can reach into the
    environment and no drawing choice can affect what the environment does.

    Fields:
        * state: The current point-car pose, speed and steering angle.
        * gate_s: Arc length of *this episode's* finish line.
        * elapsed_time: Simulated seconds since the episode began.
        * progress: Laps completed so far, as a fraction of one circuit.
        * throttle: The last applied throttle/brake action in [-1, 1].
        * steering: The last applied steering action in [-1, 1].
    """

    state: VehicleState
    gate_s: float
    elapsed_time: float = 0.0
    progress: float = 0.0
    throttle: float = 0.0
    steering: float = 0.0


class TrackCamera:
    """
    Map track coordinates onto a rectangle of pixels, preserving shape.

    The fit is computed once from the immutable boundaries, so the view never
    drifts as the car moves and two frames of the same circuit are always
    directly comparable.

    Fields:
        * origin: Track-space point drawn at the centre of the rectangle.
        * scale: Pixels per metre.
        * size: Width and height of the rectangle in pixels.
        * offset: Pixel position of the rectangle's top-left corner.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        size: tuple[int, int],
        *,
        margin: float = 40.0,
        offset: tuple[int, int] = (0, 0),
    ) -> None:
        boundaries = np.vstack((track.left_boundary, track.right_boundary))
        minimum = np.min(boundaries, axis=0)
        maximum = np.max(boundaries, axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        available = np.maximum(np.asarray(size, dtype=np.float64) - margin, 1.0)
        self.origin = (minimum + maximum) / 2.0
        self.scale = float(np.min(available / span))
        self.size = size
        self.offset = offset

    def point(self, point: np.ndarray) -> tuple[int, int]:
        """
        Convert one track-space point to a pixel location.
        """
        centered = (np.asarray(point, dtype=np.float64) - self.origin) * self.scale
        return (
            round(self.offset[0] + self.size[0] / 2.0 + centered[0]),
            round(self.offset[1] + self.size[1] / 2.0 - centered[1]),
        )

    def points(self, points: np.ndarray) -> list[tuple[int, int]]:
        """
        Convert a track-space polyline to pixel locations.
        """
        return [self.point(row) for row in points]

    def road_quads(self, track: TrackWithGeometry) -> list[list[tuple[int, int]]]:
        """
        Return the road surface as one quad per pair of boundary samples.

        Filling the road as a single ring polygon leaves a visible chord where
        the outline closes on itself, because the boundary is one loop and the
        fill has to cross it somewhere. Quads have no seam to cross.
        """
        left = self.points(track.left_boundary)
        right = self.points(track.right_boundary)
        count = len(left)
        return [
            [
                left[index],
                left[(index + 1) % count],
                right[(index + 1) % count],
                right[index],
            ]
            for index in range(count)
        ]
