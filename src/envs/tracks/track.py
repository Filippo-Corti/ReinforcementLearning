"""Sampled racing tracks, derived geometry, and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import cos, isfinite, sin
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from scipy.interpolate import CubicSpline

from configs import CarConfig, TrackGenerationConfig

from ..geometry import PolylineProjector, ScalarPiecewisePolynomial, wrap_angle
from ..types import FloatArray
from .errors import TrackValidationError


class TrackWithGeometry:
    """
    Combine a sampled track with periodic interpolation, boundaries, and indexes.

    Fields:
        * track: The sampled track that defines the geometry.
        * left_boundary: The sampled boundary on the left of forward travel.
        * right_boundary: The sampled boundary on the right of forward travel.
        * centerline_projector: The spatial index over centerline segments.
        * left_boundary_projector: The spatial index over left-boundary segments.
        * right_boundary_projector: The spatial index over right-boundary segments.
    """

    def __init__(self, track: Track) -> None:
        self.track = track
        extended_s = np.append(track.s, track.track_length)
        extended_x = np.append(track.x, track.x[0])
        extended_y = np.append(track.y, track.y[0])
        extended_curvature = np.append(track.curvature, track.curvature[0])

        curvature_spline = CubicSpline(
            extended_s,
            extended_curvature,
            bc_type="periodic",
        )
        self._position_spline = ScalarPiecewisePolynomial.periodic_spline(
            extended_s,
            np.column_stack((extended_x, extended_y)),
        )
        self._curvature_spline = ScalarPiecewisePolynomial.of(curvature_spline)
        self._curvature_integral = ScalarPiecewisePolynomial.of(
            curvature_spline.antiderivative()
        )
        self._lap_curvature_integral = float(
            self._curvature_integral(track.track_length) - self._curvature_integral(0.0)
        )

        unwrapped_heading = np.unwrap(track.heading)
        closing_turn = wrap_angle(float(track.heading[0] - track.heading[-1]))
        self._heading_s = extended_s
        self._heading_unwrapped = np.append(
            unwrapped_heading,
            unwrapped_heading[-1] + closing_turn,
        )

        normals = np.column_stack((-np.sin(track.heading), np.cos(track.heading)))
        centerline = np.column_stack((track.x, track.y))
        self.left_boundary = centerline + (track.width / 2.0) * normals
        self.right_boundary = centerline - (track.width / 2.0) * normals
        self.left_boundary.setflags(write=False)
        self.right_boundary.setflags(write=False)

        self.centerline_projector = PolylineProjector(centerline)
        self.left_boundary_projector = PolylineProjector(self.left_boundary)
        self.right_boundary_projector = PolylineProjector(self.right_boundary)

    @classmethod
    def generate(
        cls,
        seed: int,
        *,
        track_config: TrackGenerationConfig | None = None,
        vehicle_config: CarConfig | None = None,
    ) -> TrackWithGeometry:
        """
        Generate, validate, and prepare one track outside the environment.
        """
        from .generation import generate_track

        return cls(
            generate_track(
                seed,
                track_config=track_config,
                vehicle_config=vehicle_config,
            )
        )

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        vehicle_config: CarConfig | None = None,
        track_config: TrackGenerationConfig | None = None,
    ) -> TrackWithGeometry:
        """
        Load, validate, and prepare one saved track outside the environment.
        """
        track = Track.load(path, validate_geometry=False)
        from .validation import validate_track_geometry

        return validate_track_geometry(
            track,
            vehicle_config=vehicle_config,
            track_config=track_config,
        )

    def position(self, s: float) -> FloatArray:
        """
        Interpolate centerline position periodically at an arc length.
        """
        return np.asarray(self._position_spline(self._wrapped(s)), dtype=np.float64)

    def heading(self, s: float) -> float:
        """
        Interpolate wrapped centerline heading at an arc length.
        """
        wrapped = self._wrapped(s)
        unwrapped = float(np.interp(wrapped, self._heading_s, self._heading_unwrapped))
        return wrap_angle(unwrapped)

    def normal(self, s: float) -> FloatArray:
        """
        Return the unit normal pointing left of the centerline tangent.
        """
        heading = self.heading(s)
        return np.array([-sin(heading), cos(heading)], dtype=np.float64)

    def curvature(self, s: float) -> float:
        """
        Interpolate local curvature periodically at an arc length.
        """
        return float(self._curvature_spline(self._wrapped(s)))

    def integrated_curvature(self, start_s: float, distance: float) -> float:
        """
        Integrate periodic curvature forward over a non-negative distance.
        """
        start = self._wrapped(start_s)
        if not isfinite(distance) or distance < 0:
            raise ValueError("distance must be finite and non-negative.")
        length = self.track.track_length
        complete_laps, remainder = divmod(distance, length)
        total = complete_laps * self._lap_curvature_integral
        end = start + remainder
        if end <= length:
            total += float(
                self._curvature_integral(end) - self._curvature_integral(start)
            )
        else:
            total += float(
                self._curvature_integral(length)
                - self._curvature_integral(start)
                + self._curvature_integral(end - length)
            )
        return total

    def left_boundary_position(self, s: float) -> FloatArray:
        """
        Interpolate the left boundary at an arc length.
        """
        return self.position(s) + (self.track.width / 2.0) * self.normal(s)

    def right_boundary_position(self, s: float) -> FloatArray:
        """
        Interpolate the right boundary at an arc length.
        """
        return self.position(s) - (self.track.width / 2.0) * self.normal(s)

    def _wrapped(self, s: float) -> float:
        """
        Wrap an arc length to the track's periodic interval.
        """
        return float(s % self.track.track_length)


@dataclass(frozen=True, slots=True, kw_only=True)
class Track:
    """
    Store a closed centerline sampled at uniform arc-length intervals.

    Fields:
        * generation: Metadata describing the configuration and seed that produced the track.
        * width: The constant track width.
        * sample_spacing: The distance between consecutive centerline samples.
        * track_length: The total centerline length.
        * start_index: The sample used as the canonical start and finish point.
        * s: The arc length of each sample.
        * x: The horizontal coordinate of each sample.
        * y: The vertical coordinate of each sample.
        * heading: The centerline heading at each sample.
        * curvature: The centerline curvature at each sample.
    """

    FORMAT_VERSION: ClassVar[int] = 1

    generation: TrackGenerationMetadata
    width: float
    sample_spacing: float
    track_length: float
    start_index: int
    s: FloatArray = field(repr=False, compare=False)
    x: FloatArray = field(repr=False, compare=False)
    y: FloatArray = field(repr=False, compare=False)
    heading: FloatArray = field(repr=False, compare=False)
    curvature: FloatArray = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Track:
        """
        Build a track from a decoded JSON object.
        """
        format_version = data["format_version"]
        if format_version != cls.FORMAT_VERSION:
            raise TrackValidationError(
                f"unsupported format_version {format_version}; "
                f"expected {cls.FORMAT_VERSION}."
            )

        columns: dict[str, list[float]] = {
            "s": [],
            "x": [],
            "y": [],
            "heading": [],
            "curvature": [],
        }
        for sample in data["samples"]:
            for name, values in columns.items():
                values.append(sample[name])

        return cls(
            generation=TrackGenerationMetadata.from_dict(data["generation"]),
            width=data["width"],
            sample_spacing=data["sample_spacing"],
            track_length=data["track_length"],
            start_index=data["start_index"],
            s=np.asarray(columns["s"], dtype=np.float64),
            x=np.asarray(columns["x"], dtype=np.float64),
            y=np.asarray(columns["y"], dtype=np.float64),
            heading=np.asarray(columns["heading"], dtype=np.float64),
            curvature=np.asarray(columns["curvature"], dtype=np.float64),
        )

    def to_dict(self) -> dict[str, object]:
        """
        Return the persistent JSON-compatible representation.
        """
        samples = [
            {
                "s": float(s),
                "x": float(x),
                "y": float(y),
                "heading": float(heading),
                "curvature": float(curvature),
            }
            for s, x, y, heading, curvature in zip(
                self.s, self.x, self.y, self.heading, self.curvature
            )
        ]
        return {
            "format_version": self.FORMAT_VERSION,
            "generation": self.generation.to_dict(),
            "width": self.width,
            "sample_spacing": self.sample_spacing,
            "track_length": self.track_length,
            "start_index": self.start_index,
            "samples": samples,
        }

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        validate_geometry: bool = True,
        vehicle_config: CarConfig | None = None,
        track_config: TrackGenerationConfig | None = None,
    ) -> Track:
        """
        Load and validate a UTF-8 JSON track file.
        """
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise TrackValidationError(f"{source} is not valid JSON: {error.msg}.")

        track = cls.from_dict(data)
        if validate_geometry:
            from .validation import validate_track_geometry

            validate_track_geometry(
                track,
                vehicle_config=vehicle_config,
                track_config=track_config,
            )
        return track

    def save(
        self,
        path: str | PathLike[str],
        *,
        validate_geometry: bool = True,
        vehicle_config: CarConfig | None = None,
        track_config: TrackGenerationConfig | None = None,
    ) -> None:
        """
        Validate and serialize the track deterministically as UTF-8 JSON.
        """
        if validate_geometry:
            from .validation import validate_track_geometry

            validate_track_geometry(
                self,
                vehicle_config=vehicle_config,
                track_config=track_config,
            )

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        destination.write_text(f"{serialized}\n", encoding="utf-8", newline="\n")


@dataclass(frozen=True, slots=True)
class TrackGenerationMetadata:
    """
    Store the configuration and seed that produced a sampled track.

    Fields:
        * seed: The random seed used to generate the track.
        * n_checkpoints: The number of checkpoints used to generate the track.
        * base_radius: The base radius of the track.
        * radial_jitter: The fraction of the base radius used for radial jitter.
        * angular_jitter: The fraction of one checkpoint sector used for angular jitter.
        * max_attempts: The maximum number of generation attempts.
    """

    seed: int
    n_checkpoints: int
    base_radius: float
    radial_jitter: float
    angular_jitter: float
    max_attempts: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackGenerationMetadata:
        """
        Build generation metadata from its JSON representation.
        """
        return cls(
            seed=data["seed"],
            n_checkpoints=data["n_checkpoints"],
            base_radius=data["base_radius"],
            radial_jitter=data["radial_jitter"],
            angular_jitter=data["angular_jitter"],
            max_attempts=data["max_attempts"],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return JSON-compatible generation metadata.
        """
        return {
            "seed": self.seed,
            "n_checkpoints": self.n_checkpoints,
            "base_radius": self.base_radius,
            "radial_jitter": self.radial_jitter,
            "angular_jitter": self.angular_jitter,
            "max_attempts": self.max_attempts,
        }
