"""Sampled racing-track data and deterministic JSON persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import hypot, isfinite
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from configs import TrackGenerationConfig, VehicleConfig


class TrackValidationError(ValueError):
    """Raised when track data does not satisfy the persistent schema."""


class UnsupportedTrackFormatError(TrackValidationError):
    """Raised when a track file uses an unsupported schema version."""


@dataclass(frozen=True, slots=True)
class TrackUnits:
    """Units attached to persisted track values."""

    length: str = "m"
    angle: str = "rad"
    curvature: str = "1/m"

    def __post_init__(self) -> None:
        expected = ("m", "rad", "1/m")
        actual = (self.length, self.angle, self.curvature)
        if actual != expected:
            raise TrackValidationError(
                "units must be length='m', angle='rad', curvature='1/m'."
            )

    @classmethod
    def from_dict(cls, data: object) -> TrackUnits:
        """Build units from their JSON representation."""
        mapping = _require_mapping(data, "units")
        _require_exact_keys(
            mapping,
            {"length", "angle", "curvature"},
            "units",
        )
        return cls(
            length=_require_string(mapping["length"], "units.length"),
            angle=_require_string(mapping["angle"], "units.angle"),
            curvature=_require_string(
                mapping["curvature"],
                "units.curvature",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible units mapping."""
        return {
            "length": self.length,
            "angle": self.angle,
            "curvature": self.curvature,
        }


@dataclass(frozen=True, slots=True)
class TrackGenerationMetadata:
    """Configuration and seed that produced a sampled track."""

    seed: int
    n_checkpoints: int
    base_radius_m: float
    radial_jitter_fraction: float
    angular_jitter_sectors: float
    max_attempts: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise TrackValidationError(
                "generation.seed must be a non-negative integer."
            )
        if type(self.n_checkpoints) is not int or self.n_checkpoints < 3:
            raise TrackValidationError(
                "generation.n_checkpoints must be an integer of at least 3."
            )
        if not isfinite(self.base_radius_m) or self.base_radius_m <= 0:
            raise TrackValidationError(
                "generation.base_radius must be finite and positive."
            )
        if (
            not isfinite(self.radial_jitter_fraction)
            or not 0 <= self.radial_jitter_fraction < 1
        ):
            raise TrackValidationError(
                "generation.radial_jitter must be finite and in [0, 1)."
            )
        if (
            not isfinite(self.angular_jitter_sectors)
            or not 0 <= self.angular_jitter_sectors < 0.5
        ):
            raise TrackValidationError(
                "generation.angular_jitter_sectors must be finite and in [0, 0.5)."
            )
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise TrackValidationError(
                "generation.max_attempts must be a positive integer."
            )

    @classmethod
    def from_dict(cls, data: object) -> TrackGenerationMetadata:
        """Build generation metadata from its JSON representation."""
        mapping = _require_mapping(data, "generation")
        _require_exact_keys(
            mapping,
            {
                "seed",
                "n_checkpoints",
                "base_radius",
                "radial_jitter",
                "angular_jitter_sectors",
                "max_attempts",
            },
            "generation",
        )
        return cls(
            seed=_require_int(mapping["seed"], "generation.seed"),
            n_checkpoints=_require_int(
                mapping["n_checkpoints"],
                "generation.n_checkpoints",
            ),
            base_radius_m=_require_float(
                mapping["base_radius"],
                "generation.base_radius",
            ),
            radial_jitter_fraction=_require_float(
                mapping["radial_jitter"],
                "generation.radial_jitter",
            ),
            angular_jitter_sectors=_require_float(
                mapping["angular_jitter_sectors"],
                "generation.angular_jitter_sectors",
            ),
            max_attempts=_require_int(
                mapping["max_attempts"],
                "generation.max_attempts",
            ),
        )

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-compatible generation metadata."""
        return {
            "seed": self.seed,
            "n_checkpoints": self.n_checkpoints,
            "base_radius": self.base_radius_m,
            "radial_jitter": self.radial_jitter_fraction,
            "angular_jitter_sectors": self.angular_jitter_sectors,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Track:
    """Validated, uniformly sampled representation of a closed racing track."""

    FORMAT_VERSION: ClassVar[int] = 1

    generation: TrackGenerationMetadata
    width_m: float
    sample_spacing_m: float
    track_length_m: float
    start_index: int
    s_m: NDArray[np.float64] = field(repr=False, compare=False)
    x_m: NDArray[np.float64] = field(repr=False, compare=False)
    y_m: NDArray[np.float64] = field(repr=False, compare=False)
    heading_rad: NDArray[np.float64] = field(repr=False, compare=False)
    curvature_per_m: NDArray[np.float64] = field(repr=False, compare=False)
    units: TrackUnits = field(default_factory=TrackUnits)

    def __post_init__(self) -> None:
        if not isfinite(self.width_m) or self.width_m <= 0:
            raise TrackValidationError("width must be finite and positive.")
        if not isfinite(self.sample_spacing_m) or self.sample_spacing_m <= 0:
            raise TrackValidationError("sample_spacing must be finite and positive.")
        if not isfinite(self.track_length_m) or self.track_length_m <= 0:
            raise TrackValidationError("track_length must be finite and positive.")
        if type(self.start_index) is not int:
            raise TrackValidationError("start_index must be an integer.")

        array_fields = (
            "s_m",
            "x_m",
            "y_m",
            "heading_rad",
            "curvature_per_m",
        )
        arrays: dict[str, NDArray[np.float64]] = {}
        for name in array_fields:
            try:
                array = np.array(
                    getattr(self, name),
                    dtype=np.float64,
                    copy=True,
                )
            except (TypeError, ValueError) as error:
                raise TrackValidationError(
                    f"{name} must contain numeric values."
                ) from error
            if array.ndim != 1:
                raise TrackValidationError(f"{name} must be one-dimensional.")
            if not np.all(np.isfinite(array)):
                raise TrackValidationError(f"{name} must contain only finite values.")
            array.setflags(write=False)
            arrays[name] = array
            object.__setattr__(self, name, array)

        sample_count = arrays["s_m"].size
        if sample_count < 3:
            raise TrackValidationError("samples must contain at least 3 entries.")
        if any(array.size != sample_count for array in arrays.values()):
            raise TrackValidationError("all sample arrays must have the same length.")
        if not 0 <= self.start_index < sample_count:
            raise TrackValidationError("start_index must reference an existing sample.")

        s_m = arrays["s_m"]
        if np.any(np.diff(s_m) <= 0):
            raise TrackValidationError(
                "sample arc lengths must be strictly increasing."
            )

        tolerance = 1e-9 * max(1.0, abs(self.track_length_m))
        expected_s = np.arange(sample_count, dtype=np.float64) * self.sample_spacing_m
        if not np.allclose(s_m, expected_s, rtol=0.0, atol=tolerance):
            raise TrackValidationError(
                "sample arc lengths must equal index * sample_spacing."
            )

        expected_length = sample_count * self.sample_spacing_m
        if not np.isclose(
            self.track_length_m,
            expected_length,
            rtol=0.0,
            atol=tolerance,
        ):
            raise TrackValidationError(
                "track_length must include one sample_spacing closing segment."
            )

        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(arrays["x_m"]))),
            float(np.max(np.abs(arrays["y_m"]))),
        )
        duplicate_tolerance = np.finfo(np.float64).eps * coordinate_scale * 16
        closing_chord = hypot(
            arrays["x_m"][-1] - arrays["x_m"][0],
            arrays["y_m"][-1] - arrays["y_m"][0],
        )
        if closing_chord <= duplicate_tolerance:
            raise TrackValidationError(
                "the final sample must not duplicate the first; "
                "the closing segment is implicit."
            )

    @classmethod
    def from_dict(cls, data: object) -> Track:
        """Build a track from a decoded JSON object."""
        mapping = _require_mapping(data, "track")

        if "format_version" not in mapping:
            raise TrackValidationError("track is missing key: format_version.")
        format_version = _require_int(
            mapping["format_version"],
            "format_version",
        )
        if format_version != cls.FORMAT_VERSION:
            raise UnsupportedTrackFormatError(
                f"unsupported format_version {format_version}; "
                f"expected {cls.FORMAT_VERSION}."
            )

        _require_exact_keys(
            mapping,
            {
                "format_version",
                "units",
                "generation",
                "width",
                "sample_spacing",
                "track_length",
                "start_index",
                "samples",
            },
            "track",
        )

        samples = _require_sequence(mapping["samples"], "samples")
        columns: dict[str, list[float]] = {
            "s": [],
            "x": [],
            "y": [],
            "heading": [],
            "curvature": [],
        }
        for index, sample_data in enumerate(samples):
            sample = _require_mapping(sample_data, f"samples[{index}]")
            _require_exact_keys(
                sample,
                set(columns),
                f"samples[{index}]",
            )
            for name, values in columns.items():
                values.append(
                    _require_float(
                        sample[name],
                        f"samples[{index}].{name}",
                    )
                )

        return cls(
            generation=TrackGenerationMetadata.from_dict(mapping["generation"]),
            units=TrackUnits.from_dict(mapping["units"]),
            width_m=_require_float(mapping["width"], "width"),
            sample_spacing_m=_require_float(
                mapping["sample_spacing"],
                "sample_spacing",
            ),
            track_length_m=_require_float(
                mapping["track_length"],
                "track_length",
            ),
            start_index=_require_int(mapping["start_index"], "start_index"),
            s_m=np.asarray(columns["s"], dtype=np.float64),
            x_m=np.asarray(columns["x"], dtype=np.float64),
            y_m=np.asarray(columns["y"], dtype=np.float64),
            heading_rad=np.asarray(columns["heading"], dtype=np.float64),
            curvature_per_m=np.asarray(columns["curvature"], dtype=np.float64),
        )

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        validate_geometry: bool = True,
        vehicle_config: VehicleConfig | None = None,
        track_config: TrackGenerationConfig | None = None,
    ) -> Track:
        """Load and validate a UTF-8 JSON track file."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise TrackValidationError(
                f"{source} is not valid JSON: {error.msg}."
            ) from error
        track = cls.from_dict(data)
        if validate_geometry:
            from .geometry import validate_track_geometry

            validate_track_geometry(
                track,
                vehicle_config=vehicle_config,
                track_config=track_config,
            )
        return track

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
                self.s_m,
                self.x_m,
                self.y_m,
                self.heading_rad,
                self.curvature_per_m,
                strict=True,
            )
        ]
        return {
            "format_version": self.FORMAT_VERSION,
            "units": self.units.to_dict(),
            "generation": self.generation.to_dict(),
            "width": self.width_m,
            "sample_spacing": self.sample_spacing_m,
            "track_length": self.track_length_m,
            "start_index": self.start_index,
            "samples": samples,
        }

    def save(
        self,
        path: str | PathLike[str],
        *,
        validate_geometry: bool = True,
        vehicle_config: VehicleConfig | None = None,
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


""" Helper functions for validating JSON track data. """


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require that a value is a mapping with string keys."""
    if not isinstance(value, Mapping):
        raise TrackValidationError(f"{name} must be an object.")
    if not all(isinstance(key, str) for key in value):  # type: ignore
        raise TrackValidationError(f"{name} keys must be strings.")
    return value  # type: ignore


def _require_sequence(value: object, name: str) -> Sequence[object]:
    """Require that a value is a sequence (list or tuple)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrackValidationError(f"{name} must be an array.")
    return value  # type: ignore


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing keys: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected keys: {', '.join(unexpected)}")
    if details:
        raise TrackValidationError(f"{name} has {'; '.join(details)}.")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TrackValidationError(f"{name} must be a string.")
    return value


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TrackValidationError(f"{name} must be an integer.")
    return value


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackValidationError(f"{name} must be numeric.")
    converted = float(value)
    if not isfinite(converted):
        raise TrackValidationError(f"{name} must be finite.")
    return converted
