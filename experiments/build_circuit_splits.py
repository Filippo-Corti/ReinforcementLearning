"""Freeze the Experiment 2 circuit splits and the bins used to stratify them.

A circuit is named by a logical identity inside its split's seed namespace, and
its geometry is rebuilt from the frozen generator rather than stored. What this
writes is therefore not the circuits but the *commitment*: which identities each
split contains, what geometry they had when the split was fixed, and the bin
edges that later stratify held-out results.

The stratification edges come from the development circuits, which exist to be
looked at before the experiment. Deriving them from validation or test circuits
would choose the bins after seeing the data they are meant to describe.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from configs import EnvironmentConfig, Experiment2MatrixConfig
from configs.serialization import to_plain_dict
from envs.tracks import TrackWithGeometry
from training import CircuitSplit, circuit_track_seed
from training.circuits import SPLIT_NAMESPACES, circuit_geometry_checksum

DEFAULT_OUTPUT = Path("tracks/experiment_2_splits.json")


def build_splits(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    environment_config: EnvironmentConfig | None = None,
    matrix: Experiment2MatrixConfig | None = None,
) -> dict[str, Any]:
    """
    Generate every split, summarize its geometry, and write the commitment.
    """
    configuration = environment_config or EnvironmentConfig()
    sizes = _split_sizes(matrix or Experiment2MatrixConfig())

    splits: dict[str, Any] = {}
    summaries: dict[CircuitSplit, list[dict[str, Any]]] = {}
    for split, count in sizes.items():
        circuits = [
            _describe(split, identity, configuration) for identity in range(count)
        ]
        summaries[split] = circuits
        splits[split.value] = {
            "namespace": SPLIT_NAMESPACES[split].name,
            "circuit_count": count,
            "circuits": circuits,
        }

    manifest = {
        "schema_version": 1,
        # Recorded because the identities mean nothing without the generator
        # settings that turn them into geometry.
        "track_generation": to_plain_dict(configuration.track),
        "splits": splits,
        "geometry_strata": _strata(summaries[CircuitSplit.DEVELOPMENT]),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _split_sizes(matrix: Experiment2MatrixConfig) -> dict[CircuitSplit, int]:
    """
    Read each split's size from the frozen experiment matrix.
    """
    return {
        CircuitSplit.DEVELOPMENT: matrix.development_circuit_count,
        CircuitSplit.VALIDATION: matrix.validation_circuit_count,
        CircuitSplit.TEST: matrix.test_circuit_count,
    }


def _describe(
    split: CircuitSplit,
    identity: int,
    environment_config: EnvironmentConfig,
) -> dict[str, Any]:
    """
    Generate one circuit and record what identifies and characterizes it.
    """
    seed = circuit_track_seed(SPLIT_NAMESPACES[split], identity)
    track = TrackWithGeometry.generate(
        seed,
        track_config=environment_config.track,
        vehicle_config=environment_config.vehicle,
    )
    absolute_curvature = np.abs(track.track.curvature)
    straight_fraction = float(np.mean(absolute_curvature < 1.0 / 500.0))
    return {
        "identity": identity,
        "track_seed": seed,
        "geometry_checksum": circuit_geometry_checksum(track),
        "track_length": float(track.track.track_length),
        "straight_fraction": straight_fraction,
        "curvature_q50": float(np.quantile(absolute_curvature, 0.50)),
        "curvature_q90": float(np.quantile(absolute_curvature, 0.90)),
        "tightest_radius": float(1.0 / absolute_curvature.max()),
    }


def _strata(development: list[dict[str, Any]]) -> dict[str, list[float]]:
    """
    Choose bin edges from the development circuits alone.

    Tertiles give three bins with development circuits spread across them. They
    are recorded as explicit numbers, so held-out results are binned by a rule
    fixed before those results existed rather than by their own distribution.
    """
    lengths = np.asarray([row["track_length"] for row in development])
    curvature = np.asarray([row["curvature_q90"] for row in development])
    return {
        "length_edges": [
            round(float(value), 1) for value in np.quantile(lengths, (1 / 3, 2 / 3))
        ],
        "curvature_edges": [
            round(float(value), 5) for value in np.quantile(curvature, (1 / 3, 2 / 3))
        ],
    }


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse the output location of the split commitment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path of the split manifest to write.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Write the split manifest and report what it contains.
    """
    parsed = parse_arguments(arguments)
    manifest = build_splits(output_path=parsed.output)
    for name, split in manifest["splits"].items():
        lengths = [circuit["track_length"] for circuit in split["circuits"]]
        straight = [circuit["straight_fraction"] for circuit in split["circuits"]]
        radii = [circuit["tightest_radius"] for circuit in split["circuits"]]
        print(
            f"{name:<12} {split['circuit_count']:>3} circuits  "
            f"length {min(lengths):.0f}-{max(lengths):.0f} m  "
            f"straight {min(straight):.0%}-{max(straight):.0%}  "
            f"tightest radius {min(radii):.1f} m"
        )
    print(f"strata: {manifest['geometry_strata']}")
    print(f"written to {parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
