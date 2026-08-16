"""Draw, stage by stage, how one racing circuit is generated.

``envs.tracks.generation`` turns a seed into a finished circuit in a single
call, keeping none of the intermediate geometry. This script re-walks the same
pipeline and keeps everything: the jittered polygon, the deflection at each
vertex, the tangent construction that rounds it into a corner, the segment list,
the uniform samples, and the seam. Each stage is then drawn, in the order
``docs/TRACK.md`` describes it.

The generator code below is a deliberate duplicate of the real one. Duplication
is only useful while the copy stays honest, so the copy is drawn and the
*original* is what it is checked against: ``main`` regenerates the same seed
through ``envs.tracks.generate_track`` and reports the largest disagreement,
which is expected to be exactly zero.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from math import cos, degrees, isclose, pi, radians, sin, tan
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arc
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from configs import CarConfig, TrackGenerationConfig
from envs.geometry import wrap_angle
from envs.tracks import generate_track
from envs.tracks.errors import TrackValidationError
from envs.tracks.validation import validate_track_geometry
from envs.types import FloatArray

# One palette for the whole figure, so the same idea keeps the same colour
# across panels: what the polygon contributes, what a corner contributes, and
# what is left over as a straight.
POLYGON_COLOR = "#98a2b3"
GUIDE_COLOR = "#c9ced6"
ARC_COLOR = "#d1495b"
STRAIGHT_COLOR = "#00798c"
ROAD_COLOR = "#33363d"
ACCENT_COLOR = "#e8a33d"


@dataclass(frozen=True, slots=True)
class Segment:
    """
    One straight or constant-radius piece of the centerline.

    Fields:
        * length: Arc length spanned by the segment.
        * curvature: Signed curvature, zero on a straight and positive to the left.
    """

    length: float
    curvature: float


@dataclass(frozen=True, slots=True)
class Corner:
    """
    Everything the generator computes at one polygon vertex, kept for drawing.

    The real generator keeps only ``radius`` and ``tangent`` long enough to emit
    a segment. The rest is what makes the choice legible: how sharp the vertex
    is, how large a corner its two edges could have accepted, and which fraction
    of that was drawn.

    Fields:
        * index: Position of the vertex in the polygon.
        * vertex: The polygon vertex being rounded off.
        * deflection: Signed turn at the vertex, positive to the left.
        * fit_radius: Largest radius the two half-edges admit, before clipping.
        * largest_radius: The same after the configured maximum is applied.
        * fraction: Fraction of ``largest_radius`` drawn for this corner.
        * radius: The corner radius actually used.
        * tangent: Distance from the vertex to each tangent point.
        * entry_heading: Heading of the edge arriving at the vertex.
        * entry_point: Where the corner leaves that edge.
        * exit_point: Where the corner rejoins the outgoing edge.
        * centre: Centre of the corner's circle.
        * straight_after: What survives of the outgoing edge.
    """

    index: int
    vertex: FloatArray
    deflection: float
    fit_radius: float
    largest_radius: float
    fraction: float
    radius: float
    tangent: float
    entry_heading: float
    entry_point: FloatArray
    exit_point: FloatArray
    centre: FloatArray
    straight_after: float

    @property
    def arc_length(self) -> float:
        """
        Return the arc length the corner sweeps.
        """
        return self.radius * abs(self.deflection)

    @property
    def curvature(self) -> float:
        """
        Return the signed curvature of the corner.
        """
        return float(np.sign(self.deflection)) / self.radius


@dataclass(frozen=True, slots=True)
class GenerationStory:
    """
    Every intermediate stage of one successful generation, in pipeline order.

    Fields:
        * seed: The seed the circuit was generated from.
        * config: Generation settings in force.
        * rejections: Why each earlier attempt was rejected, oldest first.
        * ideal_vertices: Where the polygon vertices would sit without jitter.
        * vertices: The sampled polygon.
        * corners: One record per vertex, in polygon order.
        * segments: The emitted arc-and-straight sequence, before rescaling.
        * start_position: Where the circuit begins, at the first corner's entry.
        * start_heading: Heading there.
        * raw_length: Total length of the segment list as emitted.
        * track_length: The same after rescaling to a whole number of samples.
        * longest_straight: Index into ``segments`` of the straight holding the seam.
        * seam: Arc length of the seam before it is rolled to index zero.
        * x, y, heading, curvature: The finished sampled table.
    """

    seed: int
    config: TrackGenerationConfig
    rejections: tuple[str, ...]
    ideal_vertices: FloatArray
    vertices: FloatArray
    corners: tuple[Corner, ...]
    segments: tuple[Segment, ...]
    start_position: FloatArray
    start_heading: float
    raw_length: float
    track_length: float
    longest_straight: int
    seam: float
    x: FloatArray
    y: FloatArray
    heading: FloatArray
    curvature: FloatArray

    @property
    def scale(self) -> float:
        """
        Return the similarity factor applied to close on a whole sample count.
        """
        return self.track_length / self.raw_length

    @property
    def centerline(self) -> FloatArray:
        """
        Return the sampled centerline as a point array.
        """
        return np.column_stack((self.x, self.y))

    def boundaries(self) -> tuple[FloatArray, FloatArray]:
        """
        Return the left and right boundaries of the finished circuit.
        """
        normal = np.column_stack((-np.sin(self.heading), np.cos(self.heading)))
        offset = 0.5 * self.config.width * normal
        return self.centerline + offset, self.centerline - offset


# --------------------------------------------------------------------------
# Stage 0: the retry loop
# --------------------------------------------------------------------------


def retrace_generation(
    seed: int,
    *,
    track_config: TrackGenerationConfig,
    vehicle_config: CarConfig,
) -> GenerationStory:
    """
    Re-run generation for one seed, keeping every stage of the accepted attempt.

    Attempts are independent draws from streams spawned off the seed, so a
    rejected candidate is discarded whole rather than repaired. That is what
    makes the seed alone sufficient to name a circuit: whichever attempt
    succeeds, it succeeds in the same place every time.
    """
    attempt_sequences = np.random.SeedSequence(seed).spawn(track_config.max_attempts)
    rejections: list[str] = []

    for attempt_sequence in attempt_sequences:
        random = np.random.default_rng(attempt_sequence)
        try:
            story = _build_candidate(seed, random, track_config, tuple(rejections))
            validate_track_geometry(
                generate_track_stub(story),
                vehicle_config=vehicle_config,
                track_config=track_config,
            )
        except TrackValidationError as error:
            rejections.append(str(error))
            continue
        return story

    raise SystemExit(
        f"no valid circuit for seed {seed} in {track_config.max_attempts} attempts"
    )


def generate_track_stub(story: GenerationStory):  # type: ignore[no-untyped-def]
    """
    Wrap a retraced story in the ``Track`` object validation expects.

    Import is local so the rest of the file reads as geometry rather than as
    plumbing: this is the one place the duplicate has to speak the real
    generator's data type.
    """
    from envs.tracks.track import Track, TrackGenerationMetadata

    return Track(
        generation=TrackGenerationMetadata(
            seed=story.seed,
            n_corners=story.config.n_corners,
            base_radius=story.config.base_radius,
            radial_jitter=story.config.radial_jitter,
            angular_jitter=story.config.angular_jitter,
            max_attempts=story.config.max_attempts,
        ),
        width=story.config.width,
        sample_spacing=story.config.sample_spacing,
        track_length=story.track_length,
        start_index=0,
        s=np.arange(story.x.size, dtype=np.float64) * story.config.sample_spacing,
        x=story.x,
        y=story.y,
        heading=story.heading,
        curvature=story.curvature,
    )


def _build_candidate(
    seed: int,
    random: np.random.Generator,
    config: TrackGenerationConfig,
    rejections: tuple[str, ...],
) -> GenerationStory:
    """
    Run one attempt through every stage, keeping what each stage produced.
    """
    ideal_vertices, vertices = sample_polygon(random, config)
    corners, segments, start_position, start_heading = fit_corners(
        vertices, random, config
    )

    # Stage 5: rescale so the length is an exact multiple of the sample spacing.
    # Scaling every radius and every straight by one factor is a similarity
    # transform, so a closed polygon stays closed and only the size moves.
    raw_length = sum(segment.length for segment in segments)
    sample_count = max(3, round(raw_length / config.sample_spacing))
    track_length = sample_count * config.sample_spacing
    scale = track_length / raw_length
    scaled = tuple(
        Segment(segment.length * scale, segment.curvature / scale)
        for segment in segments
    )

    s = np.arange(sample_count, dtype=np.float64) * config.sample_spacing
    x, y, heading, curvature = sample_segments(
        scaled, start_position * scale, start_heading, s
    )

    # Stage 6: move the seam to the middle of the longest straight.
    longest_straight, seam = find_seam(scaled)
    shift = round(seam / config.sample_spacing) % x.size

    return GenerationStory(
        seed=seed,
        config=config,
        rejections=rejections,
        ideal_vertices=ideal_vertices,
        vertices=vertices,
        corners=corners,
        segments=segments,
        start_position=start_position,
        start_heading=start_heading,
        raw_length=raw_length,
        track_length=track_length,
        longest_straight=longest_straight,
        seam=seam,
        x=np.roll(x, -shift),
        y=np.roll(y, -shift),
        heading=np.roll(heading, -shift),
        curvature=np.roll(curvature, -shift),
    )


# --------------------------------------------------------------------------
# Stage 1: the polygon
# --------------------------------------------------------------------------


def sample_polygon(
    random: np.random.Generator,
    config: TrackGenerationConfig,
) -> tuple[FloatArray, FloatArray]:
    """
    Sample the closed polygon whose vertices become corners.

    Vertices are placed around one centre at evenly spaced angles with jitter,
    and at ``base_radius`` with radial jitter. Taking them in angular order
    makes the polygon star-shaped, and therefore simple: every ray from the
    centre crosses it exactly once. That is what lets corner fitting assume only
    neighbouring edges can interfere.

    Returns the unjittered vertices alongside the sampled ones, so a drawing can
    show what the jitter moved.
    """
    sector = 2.0 * pi / config.n_corners
    ideal_angles = np.arange(config.n_corners, dtype=np.float64) * sector

    # Draw order matters: the seed names a circuit only while every consumer of
    # the stream asks for the same values in the same order as the real one.
    angles = ideal_angles + random.uniform(
        -config.angular_jitter * sector,
        config.angular_jitter * sector,
        size=config.n_corners,
    )
    radii = config.base_radius * (
        1.0
        + random.uniform(
            -config.radial_jitter,
            config.radial_jitter,
            size=config.n_corners,
        )
    )
    ideal = np.column_stack(
        (
            config.base_radius * np.cos(ideal_angles),
            config.base_radius * np.sin(ideal_angles),
        )
    )
    return ideal, np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


# --------------------------------------------------------------------------
# Stages 2 to 4: rounding each vertex into a corner
# --------------------------------------------------------------------------


def fit_corners(
    vertices: FloatArray,
    random: np.random.Generator,
    config: TrackGenerationConfig,
) -> tuple[tuple[Corner, ...], tuple[Segment, ...], FloatArray, float]:
    """
    Replace every polygon vertex with a constant-radius corner.

    A corner of radius ``R`` turning by ``|d|`` leaves each edge a tangent
    distance ``T = R tan(|d| / 2)`` from the vertex, so radius and straight
    trade off directly: a larger corner eats further back along both edges.
    Capping each corner at half of each edge it touches leaves every straight
    non-negative without any search.

    The radius is drawn as a fraction of the largest one that fits rather than
    as an absolute length, which is what makes the contrast controllable: the
    same fraction gives a tight corner on a small vertex and a sweeper on a
    large one.
    """
    count = len(vertices)
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.hypot(edges[:, 0], edges[:, 1])
    edge_headings = np.arctan2(edges[:, 1], edges[:, 0])

    # Vertex i is entered along edge i-1 and left along edge i, so its turn is
    # the change between those two headings.
    deflections = np.asarray(
        [
            wrap_angle(float(edge_headings[index] - edge_headings[index - 1]))
            for index in range(count)
        ],
        dtype=np.float64,
    )
    fractions = random.uniform(
        config.corner_radius_fraction[0],
        config.corner_radius_fraction[1],
        size=count,
    )

    radii = np.zeros(count, dtype=np.float64)
    tangents = np.zeros(count, dtype=np.float64)
    fit_radii = np.zeros(count, dtype=np.float64)
    largest_radii = np.zeros(count, dtype=np.float64)
    for index in range(count):
        deflection = abs(float(deflections[index]))
        if deflection < 1e-9:
            raise TrackValidationError("polygon vertex does not turn.")
        available = 0.5 * min(
            float(edge_lengths[index - 1]), float(edge_lengths[index])
        )
        fit_radius = available / tan(deflection / 2.0)
        largest_radius = min(fit_radius, config.max_corner_radius)
        if largest_radius < config.min_corner_radius:
            raise TrackValidationError(
                "polygon vertex is too sharp for the edges that meet it."
            )
        radius = min(
            max(float(fractions[index]) * largest_radius, config.min_corner_radius),
            largest_radius,
        )
        fit_radii[index] = fit_radius
        largest_radii[index] = largest_radius
        radii[index] = radius
        tangents[index] = radius * tan(deflection / 2.0)

    corners: list[Corner] = []
    segments: list[Segment] = []
    for index in range(count):
        deflection = float(deflections[index])
        radius = float(radii[index])
        tangent = float(tangents[index])
        entry_heading = float(edge_headings[index - 1])
        curvature = float(np.sign(deflection)) / radius

        entry_point = vertices[index] - tangent * _direction(entry_heading)
        exit_point, _ = advance(
            entry_point, entry_heading, curvature, radius * abs(deflection)
        )
        centre = entry_point + (1.0 / curvature) * _direction(entry_heading + 0.5 * pi)

        segments.append(Segment(radius * abs(deflection), curvature))
        straight = (
            float(edge_lengths[index]) - tangent - float(tangents[(index + 1) % count])
        )
        if straight < 0.0:
            raise TrackValidationError("two corners overlap on one straight.")
        if straight > 0.0:
            segments.append(Segment(straight, 0.0))

        corners.append(
            Corner(
                index=index,
                vertex=vertices[index],
                deflection=deflection,
                fit_radius=float(fit_radii[index]),
                largest_radius=float(largest_radii[index]),
                fraction=float(fractions[index]),
                radius=radius,
                tangent=tangent,
                entry_heading=entry_heading,
                entry_point=entry_point,
                exit_point=exit_point,
                centre=centre,
                straight_after=straight,
            )
        )

    # The circuit begins where the first corner does, a tangent distance back
    # along the edge that arrives at it.
    entry_heading = float(edge_headings[-1])
    start_position = vertices[0] - tangents[0] * _direction(entry_heading)
    return tuple(corners), tuple(segments), start_position, entry_heading


def advance(
    position: FloatArray,
    heading: float,
    curvature: float,
    distance: float,
) -> tuple[FloatArray, float]:
    """
    Move one pose along a straight or a constant-radius arc, in closed form.

    An arc of signed curvature ``k`` turns about a centre offset ``1 / k`` to
    its left, so a single expression covers left-hand and right-hand corners and
    nothing has to be integrated numerically.
    """
    if curvature == 0.0:
        step = np.asarray((cos(heading), sin(heading)), dtype=np.float64)
        return position + distance * step, heading
    signed_radius = 1.0 / curvature
    # (-sin, cos) is the unit normal pointing left of the heading. It is spelled
    # out rather than reused from `_direction` so that this arithmetic is
    # identical, operation for operation, to the generator it stands in for.
    centre = position + signed_radius * np.asarray(
        (-sin(heading), cos(heading)), dtype=np.float64
    )
    final_heading = heading + curvature * distance
    point = centre + signed_radius * np.asarray(
        (sin(final_heading), -cos(final_heading)), dtype=np.float64
    )
    return point, final_heading


def _direction(heading: float) -> FloatArray:
    """
    Return the unit vector pointing along a heading.
    """
    return np.asarray((cos(heading), sin(heading)), dtype=np.float64)


# --------------------------------------------------------------------------
# Stages 5 and 6: sampling and the seam
# --------------------------------------------------------------------------


def sample_segments(
    segments: tuple[Segment, ...],
    start_position: FloatArray,
    start_heading: float,
    s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Evaluate pose and curvature at each requested arc length.

    Because both primitives have a closed-form parametrization, curvature is
    exact rather than a finite difference of neighbouring headings, and the
    closure check below is a statement about the construction rather than about
    accumulated integration error.
    """
    boundaries = np.concatenate(
        ([0.0], np.cumsum([segment.length for segment in segments]))
    )
    poses: list[tuple[FloatArray, float]] = []
    position = np.asarray(start_position, dtype=np.float64)
    heading = start_heading
    for segment in segments:
        poses.append((position, heading))
        position, heading = advance(
            position, heading, segment.curvature, segment.length
        )

    closure = float(np.hypot(*(position - np.asarray(start_position))))
    if not isclose(closure, 0.0, abs_tol=1e-6):
        raise TrackValidationError(
            f"segment sequence does not close: {closure:.6g} m apart."
        )

    indices = np.clip(
        np.searchsorted(boundaries, s, side="right") - 1, 0, len(segments) - 1
    )
    x = np.empty(s.size, dtype=np.float64)
    y = np.empty(s.size, dtype=np.float64)
    headings = np.empty(s.size, dtype=np.float64)
    curvature = np.empty(s.size, dtype=np.float64)
    for sample, index in enumerate(indices):
        segment = segments[index]
        base_position, base_heading = poses[index]
        offset = float(s[sample]) - float(boundaries[index])
        point, pose_heading = advance(
            base_position, base_heading, segment.curvature, offset
        )
        x[sample] = point[0]
        y[sample] = point[1]
        headings[sample] = wrap_angle(pose_heading)
        curvature[sample] = segment.curvature
    return x, y, headings, curvature


def find_seam(segments: tuple[Segment, ...]) -> tuple[int, float]:
    """
    Return the longest straight and the arc length of its midpoint.

    Starting there puts the finish line where a real one sits, and joins two
    samples that both have zero curvature, so the periodic table is continuous
    in position, heading and curvature with no special handling at the seam.
    """
    boundaries = np.concatenate(
        ([0.0], np.cumsum([segment.length for segment in segments]))
    )
    straights = [
        index for index, segment in enumerate(segments) if segment.curvature == 0.0
    ]
    if not straights:
        raise TrackValidationError("circuit has no straight to start from.")
    longest = max(straights, key=lambda index: segments[index].length)
    return longest, float(boundaries[longest]) + 0.5 * segments[longest].length


def walk_segments(
    segments: tuple[Segment, ...],
    start_position: FloatArray,
    start_heading: float,
    step: float = 1.0,
) -> list[tuple[Segment, FloatArray]]:
    """
    Return one polyline per segment, so arcs and straights can be drawn apart.
    """
    polylines: list[tuple[Segment, FloatArray]] = []
    position = np.asarray(start_position, dtype=np.float64)
    heading = start_heading
    for segment in segments:
        count = max(2, int(segment.length / step) + 2)
        offsets = np.linspace(0.0, segment.length, count)
        points = np.asarray(
            [
                advance(position, heading, segment.curvature, float(offset))[0]
                for offset in offsets
            ]
        )
        polylines.append((segment, points))
        position, heading = advance(
            position, heading, segment.curvature, segment.length
        )
    return polylines


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def draw_story(story: GenerationStory, *, zoom_vertex: int) -> Figure:
    """
    Draw all six stages, plus the curvature profile the pipeline produces.
    """
    figure = plt.figure(figsize=(16.5, 14.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.55))

    panels = [
        figure.add_subplot(grid[row, column]) for row in (0, 1) for column in range(3)
    ]
    for panel in panels:
        panel.set_aspect("equal")
        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_color(GUIDE_COLOR)

    _draw_polygon_stage(panels[0], story)
    _draw_deflection_stage(panels[1], story)
    _draw_zoom_stage(panels[2], story, zoom_vertex)
    _draw_segment_stage(panels[3], story)
    _draw_sampling_stage(panels[4], story)
    _draw_seam_stage(panels[5], story)
    _draw_curvature_profile(figure.add_subplot(grid[2, :]), story)

    figure.suptitle(
        f"How circuit seed {story.seed} is generated  "
        f"({story.track_length:.1f} m, {len(story.corners)} corners, "
        f"attempt {len(story.rejections) + 1} of {story.config.max_attempts})",
        fontsize=15,
    )
    return figure


def _draw_polygon_stage(axis: Axes, story: GenerationStory) -> None:
    """
    Stage 1: where the jitter puts the vertices, and what they would be without it.
    """
    config = story.config
    theta = np.linspace(0.0, 2.0 * pi, 361)
    inner = config.base_radius * (1.0 - config.radial_jitter)
    outer = config.base_radius * (1.0 + config.radial_jitter)

    # The band every vertex radius is drawn from.
    axis.fill(
        np.concatenate((outer * np.cos(theta), (inner * np.cos(theta))[::-1])),
        np.concatenate((outer * np.sin(theta), (inner * np.sin(theta))[::-1])),
        color=POLYGON_COLOR,
        alpha=0.13,
        linewidth=0.0,
    )
    for radius, style in (
        (config.base_radius, "--"),
        (inner, ":"),
        (outer, ":"),
    ):
        axis.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            color=GUIDE_COLOR,
            linestyle=style,
            linewidth=1.0,
        )

    # Each vertex is a jittered angle and a jittered radius away from an evenly
    # spaced position on that circle, which the arrows show.
    for index, (ideal, vertex) in enumerate(zip(story.ideal_vertices, story.vertices)):
        axis.plot(
            *ideal,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=POLYGON_COLOR,
            zorder=3,
        )
        axis.plot(
            (0.0, vertex[0]),
            (0.0, vertex[1]),
            color=POLYGON_COLOR,
            linewidth=0.6,
            linestyle=":",
        )
        axis.annotate(
            "",
            xy=vertex,
            xytext=ideal,
            arrowprops={"arrowstyle": "->", "color": ACCENT_COLOR, "linewidth": 1.1},
        )
        outward = vertex / max(float(np.hypot(*vertex)), 1e-9)
        axis.annotate(
            str(index),
            xy=tuple(vertex + 9.0 * outward),
            color=ROAD_COLOR,
            fontsize=8,
            ha="center",
            va="center",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )

    closed = np.vstack((story.vertices, story.vertices[:1]))
    axis.plot(closed[:, 0], closed[:, 1], color=POLYGON_COLOR, linewidth=1.6)
    axis.plot(
        story.vertices[:, 0],
        story.vertices[:, 1],
        linestyle="none",
        marker="o",
        markersize=5,
        color=ROAD_COLOR,
        zorder=4,
    )
    axis.plot(0.0, 0.0, marker="+", markersize=9, color=GUIDE_COLOR)
    axis.annotate(
        "arrows: where the jitter moved each vertex",
        xy=(0.5, 0.015),
        xycoords="axes fraction",
        color=ACCENT_COLOR,
        fontsize=8,
        ha="center",
    )
    axis.set_title(
        f"1. Sample a jittered polar polygon\n"
        f"{config.n_corners} vertices, R = {config.base_radius:.0f} m "
        f"± {config.radial_jitter:.0%}, angle ± {config.angular_jitter:.2f} sector",
        fontsize=10,
    )


def _draw_deflection_stage(axis: Axes, story: GenerationStory) -> None:
    """
    Stage 2: how far the route turns at each vertex, and in which direction.
    """
    closed = np.vstack((story.vertices, story.vertices[:1]))
    axis.plot(closed[:, 0], closed[:, 1], color=POLYGON_COLOR, linewidth=1.6)

    lengths = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    for corner in story.corners:
        # Keep the marker inside the two edges it sits between, so a short edge
        # does not get a marker longer than itself.
        span = 0.30 * min(
            float(lengths[corner.index - 1]), float(lengths[corner.index])
        )
        marker_radius = min(span, 0.18 * story.config.base_radius)
        exit_heading = corner.entry_heading + corner.deflection
        left_turn = corner.deflection > 0.0
        first, second = (
            (corner.entry_heading, exit_heading)
            if left_turn
            else (exit_heading, corner.entry_heading)
        )
        axis.add_patch(
            Arc(
                tuple(corner.vertex),
                2.0 * marker_radius,
                2.0 * marker_radius,
                theta1=degrees(first),
                theta2=degrees(second),
                color=ARC_COLOR if left_turn else STRAIGHT_COLOR,
                linewidth=1.4,
            )
        )
        label_at = corner.vertex + 1.55 * marker_radius * _direction(
            corner.entry_heading + 0.5 * corner.deflection
        )
        axis.annotate(
            f"{degrees(corner.deflection):+.0f}°",
            xy=tuple(label_at),
            color=ARC_COLOR if left_turn else STRAIGHT_COLOR,
            fontsize=8,
            ha="center",
            va="center",
        )

    axis.plot(
        story.vertices[:, 0],
        story.vertices[:, 1],
        linestyle="none",
        marker="o",
        markersize=5,
        color=ROAD_COLOR,
        zorder=4,
    )
    right_handers = sum(1 for corner in story.corners if corner.deflection < 0.0)
    axis.set_title(
        "2. Measure the turn at each vertex\n"
        f"Δ > 0 turns left (red), Δ < 0 turns right (blue) — "
        f"{right_handers} of {len(story.corners)} turn right",
        fontsize=10,
    )


def _draw_zoom_stage(axis: Axes, story: GenerationStory, index: int) -> None:
    """
    Stage 3: the tangent construction that turns one vertex into a corner.
    """
    corner = story.corners[index]
    previous = story.vertices[index - 1]
    following = story.vertices[(index + 1) % len(story.vertices)]

    # Half of each edge is the most this corner may claim, which is what keeps
    # both straights non-negative without searching for a radius. Only that half
    # is drawn, so the budget and the corner fit in one view.
    midpoints = [
        0.5 * (corner.vertex + neighbour) for neighbour in (previous, following)
    ]
    for midpoint in midpoints:
        axis.plot(
            (corner.vertex[0], midpoint[0]),
            (corner.vertex[1], midpoint[1]),
            color=POLYGON_COLOR,
            linewidth=1.4,
        )
        axis.plot(
            *midpoint,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=POLYGON_COLOR,
            zorder=4,
        )
        axis.annotate(
            "½‖e‖",
            xy=tuple(midpoint),
            xytext=(0.0, 9.0),
            textcoords="offset points",
            color=POLYGON_COLOR,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    theta = np.linspace(0.0, 2.0 * pi, 241)
    axis.plot(
        corner.centre[0] + corner.radius * np.cos(theta),
        corner.centre[1] + corner.radius * np.sin(theta),
        color=ARC_COLOR,
        linewidth=0.7,
        linestyle=":",
        alpha=0.7,
    )
    for tangent_point in (corner.entry_point, corner.exit_point):
        axis.plot(
            (corner.centre[0], tangent_point[0]),
            (corner.centre[1], tangent_point[1]),
            color=ARC_COLOR,
            linewidth=0.9,
            linestyle="--",
        )
        axis.plot(*tangent_point, marker="o", markersize=5, color=ARC_COLOR, zorder=4)

    offsets = np.linspace(0.0, corner.arc_length, 120)
    arc = np.asarray(
        [
            advance(
                corner.entry_point,
                corner.entry_heading,
                corner.curvature,
                float(offset),
            )[0]
            for offset in offsets
        ]
    )
    axis.plot(arc[:, 0], arc[:, 1], color=ARC_COLOR, linewidth=3.0)
    axis.plot(*corner.centre, marker="+", markersize=9, color=ARC_COLOR)
    axis.plot(*corner.vertex, marker="o", markersize=6, color=ROAD_COLOR, zorder=5)

    # The tangent distance is the quantity the whole construction turns on: it
    # is what the corner spends, out of the half-edge budget, to exist at all.
    for tangent_point in (corner.entry_point, corner.exit_point):
        axis.annotate(
            "",
            xy=tuple(tangent_point),
            xytext=tuple(corner.vertex),
            arrowprops={
                "arrowstyle": "<->",
                "color": ACCENT_COLOR,
                "linewidth": 1.3,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        # A shallow vertex puts both tangent points almost in line, so each
        # label is pushed off its own arrow rather than onto the other's.
        along = tangent_point - corner.vertex
        sideways = np.asarray((-along[1], along[0]))
        sideways = sideways / max(float(np.hypot(*sideways)), 1e-9)
        axis.annotate(
            f"T = {corner.tangent:.1f} m",
            xy=tuple(0.5 * (corner.vertex + tangent_point)),
            xytext=tuple(11.0 * sideways),
            textcoords="offset points",
            color=ACCENT_COLOR,
            fontsize=8.5,
            ha="center",
            va="center",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )
    # Frame the construction on the polygon. The circle is deliberately not
    # framed: at a shallow vertex it is enormous, and letting it run off the
    # panel says that more directly than shrinking everything else would.
    key_points = np.vstack(
        (corner.vertex, corner.entry_point, corner.exit_point, *midpoints)
    )
    lower = key_points.min(axis=0)
    upper = key_points.max(axis=0)
    span = float(np.max(upper - lower))
    if float(np.max(np.abs(corner.centre - 0.5 * (lower + upper)))) < span:
        key_points = np.vstack((key_points, corner.centre))
        lower = key_points.min(axis=0)
        upper = key_points.max(axis=0)
        span = float(np.max(upper - lower))
    middle = 0.5 * (lower + upper)
    half_span = 0.60 * span + 2.0
    axis.set_xlim(middle[0] - half_span, middle[0] + half_span)
    axis.set_ylim(middle[1] - half_span, middle[1] + half_span)

    # Keep the radius label near the tangent point, so it stays in frame even
    # when the centre does not.
    radius_arm = corner.centre - corner.entry_point
    reach = min(0.5, 0.45 * half_span / max(float(np.hypot(*radius_arm)), 1e-9))
    axis.annotate(
        f"R = {corner.radius:.1f} m",
        xy=tuple(corner.entry_point + reach * radius_arm),
        color=ARC_COLOR,
        fontsize=8.5,
        ha="center",
        va="center",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"},
    )
    axis.set_title(
        f"3. Round vertex {index} off:  T = R tan(|Δ| / 2)\n"
        f"|Δ| = {abs(degrees(corner.deflection)):.0f}°, "
        f"R = {corner.fraction:.2f} × R_max({corner.largest_radius:.0f} m) "
        f"= {corner.radius:.1f} m,  T ≤ ½‖e‖",
        fontsize=10,
    )


def _draw_segment_stage(axis: Axes, story: GenerationStory) -> None:
    """
    Stage 4: the emitted sequence of arcs and the straights left between them.
    """
    closed = np.vstack((story.vertices, story.vertices[:1]))
    axis.plot(
        closed[:, 0],
        closed[:, 1],
        color=POLYGON_COLOR,
        linewidth=0.9,
        linestyle="--",
        alpha=0.6,
    )

    polylines = walk_segments(
        story.segments, story.start_position, story.start_heading, step=0.8
    )
    for segment, points in polylines:
        corner_piece = segment.curvature != 0.0
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=ARC_COLOR if corner_piece else STRAIGHT_COLOR,
            linewidth=2.6 if corner_piece else 2.0,
            solid_capstyle="round",
        )

    for corner in story.corners:
        for tangent_point in (corner.entry_point, corner.exit_point):
            axis.plot(
                *tangent_point, marker="o", markersize=3.0, color=ROAD_COLOR, zorder=4
            )

    straights = [corner.straight_after for corner in story.corners]
    arc_total = sum(corner.arc_length for corner in story.corners)
    axis.set_title(
        "4. Emit each arc, then what survives of the edge\n"
        f"straight = ‖e‖ − Tᵢ − Tᵢ₊₁ ∈ [{min(straights):.0f}, {max(straights):.0f}] m; "
        f"{arc_total / story.raw_length:.0%} of the lap is corner",
        fontsize=10,
    )


def _draw_sampling_stage(axis: Axes, story: GenerationStory) -> None:
    """
    Stage 5: uniform arc-length samples, and the boundaries offset from them.
    """
    left, right = story.boundaries()
    road = np.vstack((left, right[::-1]))
    axis.fill(road[:, 0], road[:, 1], color=ROAD_COLOR, alpha=0.85, linewidth=0.0)

    centerline = story.centerline
    stride = max(1, centerline.shape[0] // 90)
    axis.plot(
        centerline[::stride, 0],
        centerline[::stride, 1],
        linestyle="none",
        marker="o",
        markersize=2.4,
        color="white",
        alpha=0.9,
    )
    for boundary in (left, right):
        closed = np.vstack((boundary, boundary[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color=ROAD_COLOR, linewidth=1.2)

    axis.set_title(
        "5. Rescale, sample every Δs, offset ± w/2\n"
        f"{story.raw_length:.2f} m × {story.scale:.6f} = {story.track_length:.1f} m "
        f"= {story.x.size} × {story.config.sample_spacing} m",
        fontsize=10,
    )


def _draw_seam_stage(axis: Axes, story: GenerationStory) -> None:
    """
    Stage 6: the seam, rolled to the middle of the longest straight.
    """
    left, right = story.boundaries()
    road = np.vstack((left, right[::-1]))
    axis.fill(road[:, 0], road[:, 1], color=ROAD_COLOR, alpha=0.18, linewidth=0.0)
    for boundary in (left, right):
        closed = np.vstack((boundary, boundary[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color=ROAD_COLOR, linewidth=1.1)

    # The straight now holding the seam runs across index zero in both
    # directions, because the roll put its midpoint there.
    indices = _straight_through_start(story.curvature)
    centerline = story.centerline
    axis.plot(
        centerline[indices, 0],
        centerline[indices, 1],
        color=ACCENT_COLOR,
        linewidth=4.0,
        solid_capstyle="butt",
        alpha=0.85,
    )

    start = centerline[0]
    normal = np.asarray(
        (-sin(float(story.heading[0])), cos(float(story.heading[0]))),
        dtype=np.float64,
    )
    gate = 0.5 * story.config.width * normal
    forward = _direction(float(story.heading[0]))
    axis.plot(
        (start[0] - gate[0], start[0] + gate[0]),
        (start[1] - gate[1], start[1] + gate[1]),
        color=ROAD_COLOR,
        linewidth=3.0,
        zorder=5,
    )
    # Set the arrow ahead of the gate and off the centerline, so it reads as the
    # direction of travel rather than as part of either line it would cross.
    arrow_base = start + 8.0 * forward + 0.45 * gate
    axis.annotate(
        "",
        xy=tuple(arrow_base + 26.0 * forward),
        xytext=tuple(arrow_base),
        arrowprops={"arrowstyle": "-|>", "color": ROAD_COLOR, "linewidth": 1.6},
    )
    axis.annotate(
        "s = 0",
        xy=tuple(start - 1.9 * gate),
        color=ROAD_COLOR,
        fontsize=9,
        ha="center",
        va="center",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"},
    )

    axis.set_title(
        "6. Put the seam mid-way along the longest straight\n"
        f"straight #{story.longest_straight} "
        f"({story.segments[story.longest_straight].length:.0f} m), "
        f"seam rolled from s = {story.seam:.1f} m to s = 0",
        fontsize=10,
    )


def _draw_curvature_profile(axis: Axes, story: GenerationStory) -> None:
    """
    Show the piecewise-constant curvature the construction produces.
    """
    s = np.arange(story.x.size, dtype=np.float64) * story.config.sample_spacing
    axis.fill_between(s, 0.0, story.curvature, step="post", color=ARC_COLOR, alpha=0.28)
    axis.step(s, story.curvature, where="post", color=ARC_COLOR, linewidth=1.2)
    axis.axhline(0.0, color=STRAIGHT_COLOR, linewidth=1.4)

    vehicle = CarConfig()
    limit = tan(radians(vehicle.max_steering_angle)) / vehicle.wheelbase
    for sign in (-1.0, 1.0):
        axis.axhline(
            sign * limit, color=ROAD_COLOR, linestyle="--", linewidth=0.9, alpha=0.7
        )
    axis.annotate(
        f"kinematic limit  1/R_min = {limit:.3f} m⁻¹",
        xy=(s[-1], limit),
        xytext=(-6.0, 4.0),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=ROAD_COLOR,
    )

    straight_fraction = float(np.mean(story.curvature == 0.0))
    axis.set_xlim(0.0, float(s[-1]))
    axis.set_xlabel("arc length s (m)")
    axis.set_ylabel("curvature κ (1/m)")
    axis.set_title(
        "Curvature is exact and piecewise constant: zero on a straight, ± 1/R in a "
        f"corner ({straight_fraction:.0%} of the lap is straight)",
        fontsize=10,
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _straight_through_start(curvature: FloatArray) -> NDArray[np.intp]:
    """
    Return the sample indices of the straight containing the seam.
    """
    count = curvature.size
    forward = 0
    while forward < count and curvature[forward % count] == 0.0:
        forward += 1
    backward = 0
    while backward < count and curvature[-(backward + 1) % count] == 0.0:
        backward += 1
    return np.concatenate(
        (
            np.arange(count - backward, count, dtype=np.intp),
            np.arange(0, forward, dtype=np.intp),
        )
    )


# --------------------------------------------------------------------------
# Narration and entry point
# --------------------------------------------------------------------------


def narrate(story: GenerationStory) -> None:
    """
    Print the same six stages as numbers, for what a drawing cannot show.
    """
    config = story.config
    for attempt, reason in enumerate(story.rejections, start=1):
        print(f"attempt {attempt} rejected: {reason}")
    print(f"attempt {len(story.rejections) + 1} accepted\n")

    radii = np.hypot(story.vertices[:, 0], story.vertices[:, 1])
    print(
        f"1. polygon      {config.n_corners} vertices at "
        f"{radii.min():.1f}-{radii.max():.1f} m "
        f"(base {config.base_radius:.0f} m +/- {config.radial_jitter:.0%})"
    )
    print("\n2-4. corners")
    print(
        f"   {'#':>2}  {'turn':>8}  {'R_max':>8}  {'fraction':>8}  "
        f"{'R':>8}  {'T':>7}  {'straight':>9}"
    )
    for corner in story.corners:
        clipped = "*" if corner.largest_radius < corner.fit_radius else " "
        print(
            f"   {corner.index:>2}  {degrees(corner.deflection):>+7.1f}d  "
            f"{corner.largest_radius:>7.1f}{clipped}  {corner.fraction:>8.2f}  "
            f"{corner.radius:>6.1f} m  {corner.tangent:>5.1f} m  "
            f"{corner.straight_after:>7.1f} m"
        )
    if any(corner.largest_radius < corner.fit_radius for corner in story.corners):
        print(f"   * clipped to max_corner_radius = {config.max_corner_radius:.0f} m")

    print(
        f"\n5. rescale      {story.raw_length:.4f} m x {story.scale:.8f} = "
        f"{story.track_length:.1f} m = {story.x.size} samples of "
        f"{config.sample_spacing} m"
    )
    print(
        f"6. seam         longest straight is segment "
        f"#{story.longest_straight} "
        f"({story.segments[story.longest_straight].length:.1f} m); "
        f"its midpoint at s = {story.seam:.1f} m becomes s = 0"
    )


def cross_check(story: GenerationStory, vehicle_config: CarConfig) -> float:
    """
    Regenerate the same seed through the real generator and return the gap.

    This file duplicates the generator to expose its intermediate stages, and a
    duplicate is only worth reading while it still describes the original. The
    two are built from the same seeded draws in the same order, so the expected
    disagreement is exactly zero.
    """
    reference = generate_track(
        story.seed,
        track_config=story.config,
        vehicle_config=vehicle_config,
    )
    return max(
        float(np.max(np.abs(reference.x - story.x))),
        float(np.max(np.abs(reference.y - story.y))),
        float(np.max(np.abs(reference.heading - story.heading))),
        float(np.max(np.abs(reference.curvature - story.curvature))),
        abs(reference.track_length - story.track_length),
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """
    defaults = TrackGenerationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="circuit seed to explain")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the figure (default outputs/track_generation_seed<N>.png)",
    )
    parser.add_argument(
        "--vertex",
        type=int,
        default=None,
        help="vertex to zoom into (default the sharpest turn)",
    )
    parser.add_argument("--corners", type=int, default=defaults.n_corners)
    parser.add_argument("--radius", type=float, default=defaults.base_radius)
    parser.add_argument("--radial-jitter", type=float, default=defaults.radial_jitter)
    parser.add_argument("--angular-jitter", type=float, default=defaults.angular_jitter)
    parser.add_argument(
        "--show", action="store_true", help="open the figure instead of only saving it"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Retrace one seed, narrate it, draw it, and check it against the original.
    """
    arguments = build_parser().parse_args(argv)
    config = TrackGenerationConfig(
        n_corners=arguments.corners,
        base_radius=arguments.radius,
        radial_jitter=arguments.radial_jitter,
        angular_jitter=arguments.angular_jitter,
    )
    vehicle = CarConfig()

    story = retrace_generation(
        arguments.seed, track_config=config, vehicle_config=vehicle
    )
    narrate(story)

    deviation = cross_check(story, vehicle)
    print(
        "\ncross-check     vs envs.tracks.generate_track: "
        f"largest disagreement {deviation:.3g}"
    )
    if deviation > 1e-9:
        raise SystemExit(
            "this script no longer reproduces the real generator; "
            "it must be updated before its drawings can be trusted."
        )

    zoom_vertex = arguments.vertex
    if zoom_vertex is None:
        zoom_vertex = max(
            story.corners, key=lambda corner: abs(corner.deflection)
        ).index
    figure = draw_story(story, zoom_vertex=zoom_vertex % len(story.corners))

    output = arguments.output or Path(
        f"outputs/track_generation_seed{arguments.seed}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    print(f"figure          {output}")
    if arguments.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
