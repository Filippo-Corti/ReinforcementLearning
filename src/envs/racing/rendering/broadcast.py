"""A driver's-eye view of the circuit with a broadcast-style overlay."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, degrees, radians, sin, tan
from typing import ClassVar

import numpy as np
import pygame

from configs import CarConfig

from ...tracks import TrackWithGeometry
from .frame import RenderFrame, TrackCamera

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class _RoadSample:
    """
    One cross-section of the road ahead, already projected onto the image.

    Fields:
        * arc_length: Position along the centerline, which keeps painted
          markings attached to the track instead of to the sampling.
        * distance: Metres ahead of the car, which sets haze and line widths.
        * left: Projected left edge.
        * right: Projected right edge.
    """

    arc_length: float
    distance: float
    left: Point
    right: Point


# The camera looks level down the road from just behind and above the car, the
# way an onboard broadcast shot does. A level camera keeps the horizon fixed,
# which is what makes a corner read as a corner rather than as the whole world
# tilting.
#
# The eye height matters more than it looks: ground distance maps to screen
# height through it alone, so a camera at head height compresses the whole road
# into a band a hundred pixels tall. The near plane has to stay small even so,
# or the road stops short of the bottom of the frame and meets the grass at a
# cliff edge. Road within the near plane is off-lens and simply clipped away.
_EYE_HEIGHT = 3.2
_FIELD_OF_VIEW = 72.0
_NEAR_PLANE = 1.5
_LOOKAHEAD = 260.0
_ROAD_SAMPLES = 120
_FINISH_LINE_DEPTH = 2.5
# The strip also runs backwards from the car. Sampling only forwards means
# "ahead along the track", which is not the same as "in front of the camera": a
# car turned across the road has all the road it can see beside and behind its
# own arc length, and drawing none of it leaves the car floating on grass.
_LOOKBEHIND = 60.0
_BEHIND_SAMPLES = 24

_SKY_HIGH = (12, 18, 38)
_SKY_LOW = (86, 122, 170)
_GRASS_FAR = (38, 74, 46)
_GRASS_NEAR = (46, 96, 54)
_ROAD_FAR = (52, 54, 60)
_ROAD_NEAR = (38, 40, 45)
_CURB_RED = (196, 52, 48)
_CURB_WHITE = (232, 232, 232)
_CENTER_DASH = (206, 206, 206)
# Distant road is blended toward the horizon, which both reads as depth and
# hides the far end of the drawn strip: the lookahead stops in open air, and
# without haze it stops visibly.
_HAZE = (104, 134, 176)

_PANEL = (10, 12, 18)
_PANEL_EDGE = (74, 80, 96)
_TEXT = (238, 240, 246)
_DIM_TEXT = (150, 156, 172)
_THROTTLE = (74, 214, 122)
_BRAKE = (226, 74, 66)
_ACCENT = (240, 196, 62)
_CAR_DOT = (240, 96, 72)


class BroadcastRacingRenderer:
    """
    Draw the circuit as the driver sees it, with the telemetry a viewer wants.

    The main image is a perspective projection of the road ahead from the car's
    own pose. A corner inset carries the circuit from above so the shape of the
    lap and the car's place in it stay legible, which a forward view alone
    cannot show. Everything else is read from the frame and drawn as an overlay.

    Fields:
        * track: The sampled circuit being drawn.
        * image_size: Output width and height in pixels.
    """

    def __init__(
        self,
        track: TrackWithGeometry,
        *,
        image_size: tuple[int, int],
        vehicle_config: CarConfig | None = None,
    ) -> None:
        self.track = track
        self.image_size = image_size
        self.vehicle = vehicle_config or CarConfig()
        width, height = image_size
        self._horizon = height * 0.42
        self._focal = (width / 2.0) / tan(radians(_FIELD_OF_VIEW) / 2.0)
        self._fonts = _Fonts()

        inset = max(150, min(round(width * 0.26), round(height * 0.34)))
        self._inset_size = (inset, inset)
        self._inset_origin = (width - inset - 18, 18)
        self._inset_camera = TrackCamera(
            track,
            self._inset_size,
            margin=26.0,
            offset=self._inset_origin,
        )
        self._inset_road = self._inset_camera.road_quads(track)
        # Sampling is dense near the car and sparse far away, because a fixed
        # spacing spends most of its samples on road that is a few pixels tall.
        self._offsets = np.concatenate(
            (
                np.linspace(-_LOOKBEHIND, 0.0, _BEHIND_SAMPLES, endpoint=False),
                np.linspace(0.0, 1.0, _ROAD_SAMPLES) ** 2.2 * _LOOKAHEAD,
            )
        )

    def draw(self, surface: pygame.Surface, frame: RenderFrame) -> None:
        """
        Draw the driver's view and the overlay that explains it.
        """
        self._draw_sky_and_ground(surface)
        self._draw_road(surface, frame)
        self._draw_cockpit(surface, frame)
        self._draw_minimap(surface, frame)
        self._draw_telemetry(surface, frame)

    def _draw_sky_and_ground(self, surface: pygame.Surface) -> None:
        """
        Fill the frame with a graded sky above the horizon and grass below it.
        """
        width, height = self.image_size
        horizon = int(self._horizon)
        for row in range(max(horizon, 0)):
            blend = row / max(horizon, 1)
            surface.fill(_mix(_SKY_HIGH, _SKY_LOW, blend), (0, row, width, 1))
        for row in range(max(horizon, 0), height):
            blend = (row - horizon) / max(height - horizon, 1)
            surface.fill(_mix(_GRASS_FAR, _GRASS_NEAR, blend), (0, row, width, 1))

    def _project(
        self, forward: float, lateral: float, *, height: float = 0.0
    ) -> Point | None:
        """
        Project a point given in car-relative metres onto the image.

        Returns nothing for anything behind the near plane, which has no image
        and whose projection would otherwise diverge. A point exactly *on* the
        plane is kept: that is where clipping puts one, and rejecting it would
        throw away every section the clipping exists to save.
        """
        if forward < _NEAR_PLANE:
            return None
        return (
            self.image_size[0] / 2.0 - self._focal * lateral / forward,
            self._horizon + self._focal * (_EYE_HEIGHT - height) / forward,
        )

    def _to_car_frame(
        self, points: np.ndarray, frame: RenderFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Express track-space points as forward and leftward metres from the car.
        """
        offset = points - np.asarray([frame.state.x, frame.state.y])
        heading = frame.state.heading
        forward = offset[:, 0] * cos(heading) + offset[:, 1] * sin(heading)
        lateral = -offset[:, 0] * sin(heading) + offset[:, 1] * cos(heading)
        return forward, lateral

    def _road_samples(self, frame: RenderFrame) -> list[_RoadSample]:
        """
        Project the road edges at each sampled distance ahead of the car.
        """
        arc = self.track.centerline_projector.project(
            np.asarray([frame.state.x, frame.state.y])
        )
        spacing = self.track.track.sample_spacing
        current_s = (arc.segment_index + arc.fraction) * spacing

        lengths = current_s + self._offsets
        left = np.asarray([self.track.left_boundary_position(s) for s in lengths])
        right = np.asarray([self.track.right_boundary_position(s) for s in lengths])
        left_forward, left_lateral = self._to_car_frame(left, frame)
        right_forward, right_lateral = self._to_car_frame(right, frame)

        samples: list[_RoadSample] = []
        for index in range(len(lengths)):
            edges = _clip_to_near_plane(
                (float(left_forward[index]), float(left_lateral[index])),
                (float(right_forward[index]), float(right_lateral[index])),
            )
            if edges is None:
                continue
            near_left, near_right = edges
            left_point = self._project(*near_left)
            right_point = self._project(*near_right)
            if left_point is None or right_point is None:
                continue
            samples.append(
                _RoadSample(
                    arc_length=float(lengths[index]),
                    # Depth is how far the road is from the camera, not how far
                    # along the track it is: behind the car those disagree.
                    distance=(near_left[0] + near_right[0]) / 2.0,
                    left=left_point,
                    right=right_point,
                )
            )
        return samples

    def _draw_road(self, surface: pygame.Surface, frame: RenderFrame) -> None:
        """
        Draw the road surface, its curbs, the centre dashes and the finish line.
        """
        samples = self._road_samples(frame)
        if len(samples) < 2:
            return

        for index in range(len(samples) - 1):
            near, far = samples[index], samples[index + 1]
            depth = _depth(near.distance)
            quad = [near.left, far.left, far.right, near.right]
            surfaced = _mix(_ROAD_NEAR, _ROAD_FAR, depth)
            pygame.draw.polygon(surface, _mix(surfaced, _HAZE, depth**2), quad)

        self._draw_curbs(surface, samples)
        self._draw_centre_dashes(surface, samples, frame)
        self._draw_finish_line(surface, samples, frame)

    def _draw_curbs(self, surface: pygame.Surface, samples: list[_RoadSample]) -> None:
        """
        Stripe both edges in alternating blocks, the way a real kerb is painted.
        """
        for index in range(len(samples) - 1):
            near, far = samples[index], samples[index + 1]
            # Blocks are keyed to arc length, so they stay put on the track and
            # stream past the car instead of crawling with the sampling.
            block = int(near.arc_length // 4.0) % 2
            depth = _depth(near.distance)
            colour = _mix(_CURB_RED if block else _CURB_WHITE, _HAZE, depth**2)
            edges = ((near.left, far.left), (near.right, far.right))
            for near_edge, far_edge in edges:
                pygame.draw.line(
                    surface,
                    colour,
                    _round(near_edge),
                    _round(far_edge),
                    width=max(1, round(7 - 6 * depth)),
                )

    def _draw_centre_dashes(
        self,
        surface: pygame.Surface,
        samples: list[_RoadSample],
        frame: RenderFrame,
    ) -> None:
        """
        Draw a dashed centre line, keyed to arc length so the dashes hold still.
        """
        del frame
        for index in range(len(samples) - 1):
            near, far = samples[index], samples[index + 1]
            span = far.arc_length - near.arc_length
            # Once the samples are further apart than the dash period, every
            # other one lands on a gap and the line breaks into flicker. Beyond
            # that distance the dashes are sub-pixel anyway, so they stop.
            if span > 4.0 or int(near.arc_length // 8.0) % 2:
                continue
            depth = _depth(near.distance)
            pygame.draw.line(
                surface,
                _mix(_CENTER_DASH, _HAZE, depth**2),
                _round(_midpoint(near.left, near.right)),
                _round(_midpoint(far.left, far.right)),
                width=max(1, round(5 - 4 * depth)),
            )

    def _draw_finish_line(
        self,
        surface: pygame.Surface,
        samples: list[_RoadSample],
        frame: RenderFrame,
    ) -> None:
        """
        Draw a chequered band where this episode's lap actually ends.
        """
        if not samples:
            return
        length = self.track.track.track_length
        # Measured from the furthest sample behind the car, since that is where
        # the strip starts. Anything genuinely behind the camera is rejected by
        # the projection below rather than by this bound.
        distance = (frame.gate_s - samples[0].arc_length) % length
        if distance > _LOOKAHEAD + _LOOKBEHIND:
            return

        # The band is built from the gate's own arc length rather than from the
        # sample that happens to bracket it: far apart samples would otherwise
        # give the line a depth of ten metres or more.
        edges: list[tuple[Point, Point]] = []
        for offset in (0.0, _FINISH_LINE_DEPTH):
            arc = frame.gate_s + offset
            pair = np.asarray(
                [
                    self.track.left_boundary_position(arc),
                    self.track.right_boundary_position(arc),
                ]
            )
            forward, lateral = self._to_car_frame(pair, frame)
            left = self._project(forward[0], lateral[0])
            right = self._project(forward[1], lateral[1])
            if left is None or right is None:
                return
            edges.append((left, right))

        for square in range(10):
            first, second = square / 10.0, (square + 1) / 10.0
            pygame.draw.polygon(
                surface,
                _CURB_WHITE if square % 2 else (22, 22, 26),
                [
                    _round(_lerp(edges[0][0], edges[0][1], first)),
                    _round(_lerp(edges[0][0], edges[0][1], second)),
                    _round(_lerp(edges[1][0], edges[1][1], second)),
                    _round(_lerp(edges[1][0], edges[1][1], first)),
                ],
            )

    def _draw_cockpit(self, surface: pygame.Surface, frame: RenderFrame) -> None:
        """
        Draw the nose of the car the view is taken from.

        Without it the camera is a floating eye and the shot reads as a fly-by.
        The nose gives the frame something that belongs to the car, and leaning
        it with the steering shows what the front wheels are doing without
        having to read the number for it.
        """
        width, height = self.image_size
        centre = width / 2.0
        base = height - 148
        limit = max(radians(self.vehicle.max_steering_angle), 1e-9)
        lean = min(max(frame.state.steering_angle / limit, -1.0), 1.0) * 30.0

        # Front wheels, set wide and turned with the steering. They sit behind
        # the nose so the nose's own edge cuts across them.
        for side in (-1.0, 1.0):
            hub = (centre + side * 208 + lean * 0.6, base + 44)
            wheel = pygame.Rect(0, 0, 54, 96)
            wheel.center = (round(hub[0]), round(hub[1]))
            pygame.draw.rect(surface, (16, 16, 20), wheel, border_radius=12)
            pygame.draw.rect(surface, (54, 58, 68), wheel, width=2, border_radius=12)

        nose = [
            (centre - 132 + lean * 0.5, base + 62),
            (centre - 34 + lean, base - 92),
            (centre + 34 + lean, base - 92),
            (centre + 132 + lean * 0.5, base + 62),
        ]
        pygame.draw.polygon(surface, (24, 26, 32), [_round(point) for point in nose])
        pygame.draw.polygon(
            surface, (72, 78, 92), [_round(point) for point in nose], width=3
        )
        # A stripe down the nose makes the lean legible; a flat shape would not
        # show which way the front wheels are pointing.
        pygame.draw.line(
            surface,
            _ACCENT,
            _round((centre + lean, base - 88)),
            _round((centre + lean * 0.5, base + 58)),
            width=5,
        )

    def _draw_minimap(self, surface: pygame.Surface, frame: RenderFrame) -> None:
        """
        Draw the circuit from above, with the finish line and the car on it.
        """
        panel = pygame.Rect(self._inset_origin, self._inset_size)
        _panel(surface, panel)
        for quad in self._inset_road:
            pygame.draw.polygon(surface, (58, 60, 68), quad)
        pygame.draw.lines(
            surface,
            (120, 124, 136),
            True,
            self._inset_camera.points(self.track.left_boundary),
            width=1,
        )
        pygame.draw.lines(
            surface,
            (120, 124, 136),
            True,
            self._inset_camera.points(self.track.right_boundary),
            width=1,
        )

        # The finish line is drawn where *this episode* started, which is where
        # the lap it is timing will actually end.
        gate = self.track.position(frame.gate_s)
        normal = self.track.normal(frame.gate_s)
        half_width = self.track.track.width / 2.0
        pygame.draw.line(
            surface,
            _CURB_WHITE,
            self._inset_camera.point(gate + half_width * normal),
            self._inset_camera.point(gate - half_width * normal),
            width=2,
        )
        pygame.draw.circle(
            surface,
            _CAR_DOT,
            self._inset_camera.point(np.asarray([frame.state.x, frame.state.y])),
            4,
        )
        self._fonts.write(
            surface, "CIRCUIT", (panel.left + 10, panel.top + 6), "small", _DIM_TEXT
        )

    def _draw_telemetry(self, surface: pygame.Surface, frame: RenderFrame) -> None:
        """
        Draw speed, lap time, lap progress and the controls being applied.
        """
        width, height = self.image_size
        panel = pygame.Rect(16, height - 132, width - 32, 116)
        _panel(surface, panel)

        speed = frame.state.speed * 3.6
        self._fonts.write(
            surface, f"{speed:5.1f}", (panel.left + 18, panel.top + 10), "huge", _TEXT
        )
        self._fonts.write(
            surface, "km/h", (panel.left + 20, panel.top + 66), "small", _DIM_TEXT
        )
        _bar(
            surface,
            pygame.Rect(panel.left + 18, panel.top + 88, 168, 8),
            min(frame.state.speed / max(self.vehicle.max_speed, 1e-9), 1.0),
            _ACCENT,
        )

        column = panel.left + 232
        self._fonts.write(
            surface, "LAP TIME", (column, panel.top + 12), "small", _DIM_TEXT
        )
        self._fonts.write(
            surface,
            _clock(frame.elapsed_time),
            (column, panel.top + 30),
            "large",
            _TEXT,
        )
        self._fonts.write(surface, "LAP", (column, panel.top + 72), "small", _DIM_TEXT)
        self._fonts.write(
            surface,
            f"{frame.progress * 100:5.1f}%",
            (column + 44, panel.top + 70),
            "medium",
            _TEXT,
        )
        _bar(
            surface,
            pygame.Rect(column, panel.top + 96, 168, 6),
            min(max(frame.progress, 0.0), 1.0),
            _THROTTLE,
        )

        self._draw_controls(
            surface, frame, pygame.Rect(panel.right - 300, panel.top + 8, 284, 100)
        )

    def _draw_controls(
        self, surface: pygame.Surface, frame: RenderFrame, area: pygame.Rect
    ) -> None:
        """
        Draw the throttle, the brake, and where the front wheels are pointing.
        """
        self._fonts.write(
            surface, "THROTTLE", (area.left, area.top), "small", _DIM_TEXT
        )
        _bar(
            surface,
            pygame.Rect(area.left, area.top + 18, 120, 10),
            max(frame.throttle, 0.0),
            _THROTTLE,
        )
        self._fonts.write(
            surface, "BRAKE", (area.left, area.top + 36), "small", _DIM_TEXT
        )
        _bar(
            surface,
            pygame.Rect(area.left, area.top + 54, 120, 10),
            max(-frame.throttle, 0.0),
            _BRAKE,
        )

        # Steering reads as an angle rather than a number, which is how a driver
        # experiences it: the marker is where the wheels actually are.
        self._fonts.write(
            surface, "STEERING", (area.left + 150, area.top), "small", _DIM_TEXT
        )
        track_rect = pygame.Rect(area.left + 150, area.top + 20, 130, 8)
        pygame.draw.rect(surface, (44, 48, 60), track_rect, border_radius=4)
        pygame.draw.line(
            surface,
            (90, 96, 112),
            (track_rect.centerx, track_rect.top - 3),
            (track_rect.centerx, track_rect.bottom + 3),
            width=1,
        )
        limit = max(radians(self.vehicle.max_steering_angle), 1e-9)
        fraction = min(max(frame.state.steering_angle / limit, -1.0), 1.0)
        marker = int(track_rect.centerx + fraction * track_rect.width / 2.0)
        pygame.draw.circle(surface, _ACCENT, (marker, track_rect.centery), 6)
        self._fonts.write(
            surface,
            f"{degrees(frame.state.steering_angle):+5.1f} deg",
            (area.left + 150, area.top + 36),
            "small",
            _TEXT,
        )


class _Fonts:
    """
    Provide the overlay's fonts, and cope with a build that has none.

    Pygame can be installed without working font support. Text is the least
    important thing on the frame, so a missing font degrades the overlay to its
    bars rather than stopping the render.
    """

    _SIZES: ClassVar[dict[str, int]] = {
        "small": 14,
        "medium": 20,
        "large": 30,
        "huge": 52,
    }

    def __init__(self) -> None:
        self._fonts: dict[str, pygame.font.Font] = {}
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            for name, size in self._SIZES.items():
                self._fonts[name] = pygame.font.SysFont("consolas,dejavusansmono", size)
        except Exception:  # noqa: BLE001
            self._fonts = {}

    def write(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        size: str,
        colour: tuple[int, int, int],
    ) -> None:
        """
        Draw one line of text, or nothing at all when no font is available.
        """
        font = self._fonts.get(size)
        if font is None:
            return
        surface.blit(font.render(text, True, colour), position)


def _panel(surface: pygame.Surface, rect: pygame.Rect) -> None:
    """
    Draw a translucent overlay panel with a thin edge.
    """
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    layer.fill((*_PANEL, 186))
    surface.blit(layer, rect.topleft)
    pygame.draw.rect(surface, _PANEL_EDGE, rect, width=1, border_radius=6)


def _bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fraction: float,
    colour: tuple[int, int, int],
) -> None:
    """
    Draw a filled proportion of a rounded track.
    """
    pygame.draw.rect(surface, (44, 48, 60), rect, border_radius=4)
    filled = int(rect.width * min(max(fraction, 0.0), 1.0))
    if filled > 0:
        pygame.draw.rect(
            surface,
            colour,
            pygame.Rect(rect.left, rect.top, filled, rect.height),
            border_radius=4,
        )


def _clock(seconds: float) -> str:
    """
    Format simulated seconds the way a timing screen would.
    """
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes)}:{remainder:06.3f}"


def _clip_to_near_plane(left: Point, right: Point) -> tuple[Point, Point] | None:
    """
    Trim one cross-section of road to the part the camera can actually see.

    Each endpoint is `(forward, lateral)` in metres from the car. A section with
    one end behind the camera used to be discarded whole, which is why a car
    turned across the road lost its road entirely: every section it could see
    had one end behind it. Moving that end up to the near plane keeps the
    section, and the strip stays continuous.
    """
    if left[0] <= _NEAR_PLANE and right[0] <= _NEAR_PLANE:
        return None
    if left[0] > _NEAR_PLANE and right[0] > _NEAR_PLANE:
        return left, right
    behind, ahead = (left, right) if left[0] <= _NEAR_PLANE else (right, left)
    weight = (_NEAR_PLANE - behind[0]) / (ahead[0] - behind[0])
    clipped = (
        _NEAR_PLANE,
        behind[1] + (ahead[1] - behind[1]) * weight,
    )
    return (clipped, ahead) if left[0] <= _NEAR_PLANE else (ahead, clipped)


def _depth(distance: float) -> float:
    """
    Return how far away something is, as a fraction of the drawn lookahead.
    """
    return min(max(distance / _LOOKAHEAD, 0.0), 1.0)


def _mix(
    first: tuple[int, int, int], second: tuple[int, int, int], blend: float
) -> tuple[int, int, int]:
    """
    Blend two colours, with zero returning the first.
    """
    weight = min(max(blend, 0.0), 1.0)
    return (
        round(first[0] + (second[0] - first[0]) * weight),
        round(first[1] + (second[1] - first[1]) * weight),
        round(first[2] + (second[2] - first[2]) * weight),
    )


def _midpoint(first: Point, second: Point) -> Point:
    """
    Return the point halfway between two projected points.
    """
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def _lerp(first: Point, second: Point, weight: float) -> Point:
    """
    Return a point a fraction of the way from one projected point to another.
    """
    return (
        first[0] + (second[0] - first[0]) * weight,
        first[1] + (second[1] - first[1]) * weight,
    )


def _round(point: Point) -> tuple[int, int]:
    """
    Snap a projected point to whole pixels.
    """
    return (round(point[0]), round(point[1]))
