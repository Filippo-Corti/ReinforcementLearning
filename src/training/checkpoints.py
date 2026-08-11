"""Versioned atomic checkpoints for resumable on-policy training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

CHECKPOINT_SCHEMA_VERSION = 2


class CheckpointError(RuntimeError):
    """
    Report an invalid, incompatible, or incomplete training checkpoint.
    """


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """
    Atomically replace one versioned checkpoint after writing and syncing it.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state": state,
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """
    Load one checkpoint only when its explicit schema and structure are valid.
    """
    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise CheckpointError(f"could not load checkpoint: {source}") from error
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a dictionary.")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "unsupported checkpoint schema "
            f"{payload.get('schema_version')!r}; expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint state must be a dictionary.")
    return state
