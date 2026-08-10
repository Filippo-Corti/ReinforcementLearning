from __future__ import annotations

from pathlib import Path

import pytest

from utils.checkpointing import CheckpointError, load_checkpoint, save_checkpoint


def test_checkpoint_atomically_round_trips_versioned_state(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"

    save_checkpoint(destination, {"counter": 3, "nested": {"value": "saved"}})

    assert load_checkpoint(destination) == {
        "counter": 3,
        "nested": {"value": "saved"},
    }


def test_checkpoint_rejects_wrong_schema(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(destination, {"counter": 1})
    import torch

    torch.save({"schema_version": 999, "state": {}}, destination)

    with pytest.raises(CheckpointError, match="unsupported"):
        load_checkpoint(destination)


def test_checkpoint_reports_a_truncated_file_as_invalid(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"not a complete torch checkpoint")

    with pytest.raises(CheckpointError, match="could not load"):
        load_checkpoint(destination)
