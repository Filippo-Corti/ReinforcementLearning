"""Sampled racing-track data and deterministic JSON persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar
import numpy as np
from numpy.typing import NDArray

from ..configs import TrackGenerationConfig, CarConfig


class TrackValidationError(ValueError):
    """Raised when track data does not satisfy the persistent schema."""

    pass


@dataclass(frozen=True, slots=True)
class TrackGenerationMetadata:
    """
    Configuration and seed that produced a sampled track.

    Fields:
        * seed: The random seed used to generate the track.
        * n_checkpoints: The number of checkpoints used to generate the track.
        * base_radius: The base radius of the track in meters.
        * radial_jitter: The fraction of the base radius used to jitter the checkpoints radially.
        * angular_jitter: The fraction of the circle used to jitter the checkpoints angularly.
        * max_attempts: The maximum number of attempts to generate a valid track.
    """

    seed: int
    n_checkpoints: int
    base_radius: float
    radial_jitter: float
    angular_jitter: float
    max_attempts: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackGenerationMetadata:
        """Build generation metadata from its JSON representation."""
        return cls(
            seed=data["seed"],
            n_checkpoints=data["n_checkpoints"],
            base_radius=data["base_radius"],
            radial_jitter=data["radial_jitter"],
            angular_jitter=data["angular_jitter"],
            max_attempts=data["max_attempts"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible generation metadata."""
        return {
            "seed": self.seed,
            "n_checkpoints": self.n_checkpoints,
            "base_radius": self.base_radius,
            "radial_jitter": self.radial_jitter,
            "angular_jitter": self.angular_jitter,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Track:
    """
    Discrete, immutable representation of a closed racing track.
    It stores samples of the track's centerline at uniform arc-length intervals, along with
    the track's width and metadata describing how it was generated.

    Fields:
        * generation: Metadata describing the configuration and seed that produced the track.
        * width: The width of the track in meters (everywhere).
        * sample_spacing: The distance between consecutive samples along the track in meters.
        * track_length: The total length of the track in meters.
        * start_index: The index of the sample that is considered the starting point of the track.
        * s: The arc length of each sample along the track in meters.
        * x: The x-coordinate of each sample in meters.
        * y: The y-coordinate of each sample in meters.
        * heading: The heading angle of each sample, in radians.
        * curvature: The curvature of the track at each sample, in 1/meters.
    """

    FORMAT_VERSION: ClassVar[int] = 1

    generation: TrackGenerationMetadata
    width: float
    sample_spacing: float
    track_length: float
    start_index: int
    s: NDArray[np.float64] = field(repr=False, compare=False)
    x: NDArray[np.float64] = field(repr=False, compare=False)
    y: NDArray[np.float64] = field(repr=False, compare=False)
    heading: NDArray[np.float64] = field(repr=False, compare=False)
    curvature: NDArray[np.float64] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Track:
        """Build a track from a decoded JSON object."""
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
        samples = data["samples"]
        for _, sample in enumerate(samples):
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
        """Return the persistent JSON-compatible representation."""
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
        """Load and validate a UTF-8 JSON track file."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise TrackValidationError(f"{source} is not valid JSON: {error.msg}.")

        track = cls.from_dict(data)
        if validate_geometry:
            from .geometry import validate_track_geometry

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
        """Validate and serialize the track deterministically as UTF-8 JSON."""
        if validate_geometry:
            from .geometry import validate_track_geometry

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
