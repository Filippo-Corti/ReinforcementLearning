"""Deterministically generate racing circuits from straights and corners.

The pipeline is specified in ``docs/TRACK.md``::

    seeded RNG
        |
        v
    jittered polar polygon           <- the route the circuit takes
        |
        v
    one constant-radius corner       <- rounds off each polygon vertex
    fitted into each vertex
        |
        v
    straight + arc segment list      <- what is left of each polygon edge
        |
        v
    uniform arc-length samples -> headings + curvature -> geometric validation

A circuit is therefore a sequence of straights and constant-radius corners, the
way a real circuit is described, rather than one smooth closed curve. Curvature
is piecewise constant and every segment has a closed-form arc-length
parametrization, so samples, headings and curvature are evaluated exactly
instead of being integrated numerically and differenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isclose, pi, sin, tan
from os import PathLike

import numpy as np

from configs import CarConfig, TrackGenerationConfig

from ..geometry import wrap_angle
from ..types import FloatArray
from .errors import TrackGenerationError, TrackValidationError
from .track import Track, TrackGenerationMetadata
from .validation import validate_track_geometry


@dataclass(frozen=True, slots=True)
class _Segment:
    """
    One straight or constant-radius piece of the centerline.

    Fields:
        * length: Arc length spanned by the segment.
        * curvature: Signed curvature, zero on a straight and positive to the left.
    """

    length: float
    curvature: float


def generate_track_file(
    path: str | PathLike[str],
    *,
    seed: int,
    track_config: TrackGenerationConfig | None = None,
    vehicle_config: CarConfig | None = None,
) -> Track:
    """
    Generate and save a deterministic track file.
    """
    generation = track_config or TrackGenerationConfig()
    vehicle = vehicle_config or CarConfig()
    track = generate_track(
        seed,
        track_config=generation,
        vehicle_config=vehicle,
    )
    track.save(
        path,
        vehicle_config=vehicle,
        track_config=generation,
    )
    return track


def generate_track(
    seed: int,
    *,
    track_config: TrackGenerationConfig | None = None,
    vehicle_config: CarConfig | None = None,
) -> Track:
    """
    Generate a valid track deterministically from ``seed`` and configuration.
    """
    generation = track_config or TrackGenerationConfig()
    vehicle = vehicle_config or CarConfig()
    attempt_sequences = np.random.SeedSequence(seed).spawn(generation.max_attempts)

    for attempt_sequence in attempt_sequences:
        random = np.random.default_rng(attempt_sequence)
        try:
            candidate = _generate_candidate(seed, random, generation)
            validate_track_geometry(
                candidate,
                vehicle_config=vehicle,
                track_config=generation,
            )
        except TrackValidationError:
            continue
        return candidate

    raise TrackGenerationError(
        f"failed to generate a valid track for seed {seed} after "
        f"{generation.max_attempts} attempts"
    )


def _generate_candidate(
    seed: int,
    random: np.random.Generator,
    config: TrackGenerationConfig,
) -> Track:
    """
    Build one candidate circuit of straights and corners.

    The polygon fixes where the circuit goes; the corner radii fix how much of
    each edge survives as a straight. A candidate may still violate the vehicle
    or separation constraints, which validation rejects.
    """
    vertices = _sample_polygon(random, config)
    segments, start_position, start_heading = _fit_corners(vertices, random, config)

    length = sum(segment.length for segment in segments)
    sample_count = max(3, round(length / config.sample_spacing))
    track_length = sample_count * config.sample_spacing
    # A similarity transform preserves closure, so scaling every radius and
    # straight together makes the length an exact multiple of the sample
    # spacing without reopening the loop.
    scale = track_length / length
    segments = tuple(
        _Segment(segment.length * scale, segment.curvature / scale)
        for segment in segments
    )
    start_position = start_position * scale

    s = np.arange(sample_count, dtype=np.float64) * config.sample_spacing
    x, y, heading, curvature = _sample_segments(
        segments, start_position, start_heading, s
    )
    x, y, heading, curvature = _rotate_to_longest_straight(
        segments, config.sample_spacing, x, y, heading, curvature
    )

    return Track(
        generation=TrackGenerationMetadata(
            seed=seed,
            n_corners=config.n_corners,
            base_radius=config.base_radius,
            radial_jitter=config.radial_jitter,
            angular_jitter=config.angular_jitter,
            max_attempts=config.max_attempts,
        ),
        width=config.width,
        sample_spacing=config.sample_spacing,
        track_length=track_length,
        start_index=0,
        s=s,
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
    )


def _sample_polygon(
    random: np.random.Generator,
    config: TrackGenerationConfig,
) -> FloatArray:
    """
    Sample the closed polygon whose vertices become corners.

    Vertices are placed at jittered angles and radii around one centre. Ordering
    them by angle makes the polygon star-shaped and therefore simple, which is
    what lets the corner fitting below assume that only neighbouring edges can
    interfere.
    """
    sector = 2.0 * pi / config.n_corners
    angles = np.arange(config.n_corners, dtype=np.float64) * sector + random.uniform(
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
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def _fit_corners(
    vertices: FloatArray,
    random: np.random.Generator,
    config: TrackGenerationConfig,
) -> tuple[tuple[_Segment, ...], FloatArray, float]:
    """
    Replace every polygon vertex with a constant-radius corner.

    A corner of radius ``R`` turning by ``|d|`` meets its two edges a tangent
    distance ``T = R * tan(|d| / 2)`` from the vertex, so the corner radius and
    the length of the surrounding straights trade off directly against each
    other. Each corner may claim at most half of each edge it touches, which
    leaves every straight non-negative without any search.

    The radius is drawn as a fraction of the largest one that fits rather than
    as an absolute value. A small fraction yields a tight corner between long
    straights and a large one yields a sweeper, so a single sampled number
    controls the contrast the circuit is generated for.
    """
    count = len(vertices)
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.hypot(edges[:, 0], edges[:, 1])
    edge_headings = np.arctan2(edges[:, 1], edges[:, 0])

    # Vertex i is entered along edge i-1 and left along edge i.
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
    for index in range(count):
        deflection = abs(float(deflections[index]))
        if deflection < 1e-9:
            raise TrackValidationError("polygon vertex does not turn.")
        available = 0.5 * min(
            float(edge_lengths[index - 1]), float(edge_lengths[index])
        )
        largest_radius = min(
            available / tan(deflection / 2.0), config.max_corner_radius
        )
        if largest_radius < config.min_corner_radius:
            raise TrackValidationError(
                "polygon vertex is too sharp for the edges that meet it."
            )
        radius = min(
            max(float(fractions[index]) * largest_radius, config.min_corner_radius),
            largest_radius,
        )
        radii[index] = radius
        tangents[index] = radius * tan(deflection / 2.0)

    segments: list[_Segment] = []
    for index in range(count):
        deflection = float(deflections[index])
        radius = float(radii[index])
        segments.append(
            _Segment(radius * abs(deflection), float(np.sign(deflection)) / radius)
        )
        straight = (
            float(edge_lengths[index])
            - float(tangents[index])
            - float(tangents[(index + 1) % count])
        )
        if straight < 0.0:
            raise TrackValidationError("two corners overlap on one straight.")
        if straight > 0.0:
            segments.append(_Segment(straight, 0.0))

    # The circuit starts where the first corner begins, a tangent distance back
    # along the edge that arrives at it.
    entry_heading = float(edge_headings[-1])
    start_position = vertices[0] - tangents[0] * np.asarray(
        (cos(entry_heading), sin(entry_heading)), dtype=np.float64
    )
    return tuple(segments), start_position, entry_heading


def _sample_segments(
    segments: tuple[_Segment, ...],
    start_position: FloatArray,
    start_heading: float,
    s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Evaluate pose and curvature at each requested arc length.

    Both primitives are integrated in closed form. A straight advances along its
    heading; an arc of signed curvature ``k`` turns about a centre offset
    ``1 / k`` to its left, so one expression covers left and right corners.
    """
    boundaries = np.concatenate(
        ([0.0], np.cumsum([segment.length for segment in segments]))
    )
    poses: list[tuple[FloatArray, float]] = []
    position = np.asarray(start_position, dtype=np.float64)
    heading = start_heading
    for segment in segments:
        poses.append((position, heading))
        position, heading = _advance(position, heading, segment.curvature, segment.length)

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
        point, pose_heading = _advance(
            base_position, base_heading, segment.curvature, offset
        )
        x[sample] = point[0]
        y[sample] = point[1]
        headings[sample] = wrap_angle(pose_heading)
        curvature[sample] = segment.curvature
    return x, y, headings, curvature


def _advance(
    position: FloatArray,
    heading: float,
    curvature: float,
    distance: float,
) -> tuple[FloatArray, float]:
    """
    Move one pose along a straight or a constant-radius arc.
    """
    if curvature == 0.0:
        step = np.asarray((cos(heading), sin(heading)), dtype=np.float64)
        return position + distance * step, heading
    signed_radius = 1.0 / curvature
    centre = position + signed_radius * np.asarray(
        (-sin(heading), cos(heading)), dtype=np.float64
    )
    final_heading = heading + curvature * distance
    point = centre + signed_radius * np.asarray(
        (sin(final_heading), -cos(final_heading)), dtype=np.float64
    )
    return point, final_heading


def _rotate_to_longest_straight(
    segments: tuple[_Segment, ...],
    sample_spacing: float,
    x: FloatArray,
    y: FloatArray,
    heading: FloatArray,
    curvature: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Move the seam to the middle of the longest straight.

    The start and finish line then sits where a real one does, and the seam
    joins two samples of equal zero curvature, so the periodic table is
    continuous in position, heading and curvature without special handling.
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
    seam = float(boundaries[longest]) + 0.5 * segments[longest].length
    shift = round(seam / sample_spacing) % x.size
    return (
        np.roll(x, -shift),
        np.roll(y, -shift),
        np.roll(heading, -shift),
        np.roll(curvature, -shift),
    )


__all__ = ["generate_track", "generate_track_file"]
