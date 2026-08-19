"""Versioned atomic checkpoints, and the engine identity guarding a resume.

Two layers, deliberately kept apart. `save_checkpoint`/`load_checkpoint` know
only how to put a dictionary on disk without ever leaving a half-written file
behind. `EngineCheckpoint` knows what a training run *is*, and refuses to
restore one onto a different one.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

CHECKPOINT_SCHEMA_VERSION = 2
ENGINE_STATE_VERSION = 4


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
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        # ImportError belongs here because a checkpoint stores pickled records,
        # and moving the module one of them was defined in makes the file
        # unreadable long before its schema version can be checked. That is a
        # checkpoint this code cannot load, which is what CheckpointError means;
        # the chained cause still names the module for whoever has to fix it.
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


class EngineCheckpoint:
    """
    Save and restore one engine's state, guarding what it is restored onto.

    Resuming is the operation with the quietest failure mode in the project: a
    checkpoint written by a different configuration will load happily and
    produce a run that is neither the old one nor a new one. Two guards prevent
    that. The state version rejects a layout this code no longer understands,
    and the configuration fingerprint rejects a run that differs in anything the
    engine's identity depends on.

    Fields:
        * configuration: Fingerprint every restore must match exactly.
    """

    def __init__(self, configuration: dict[str, Any]) -> None:
        self.configuration = configuration

    def save(self, path: str | Path, sections: dict[str, Any]) -> None:
        """
        Write the engine's state atomically, stamped with its identity.
        """
        save_checkpoint(
            path,
            {
                "engine_state_version": ENGINE_STATE_VERSION,
                "engine_configuration": self.configuration,
                **sections,
            },
        )

    def load(self, path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
        """
        Read a checkpoint, refusing one written for a different run.
        """
        state = load_checkpoint(path, map_location=map_location)
        if state.get("engine_state_version") != ENGINE_STATE_VERSION:
            raise ValueError(
                "checkpoint has an incompatible training-engine state version."
            )
        stored = state.get("engine_configuration")
        if not isinstance(stored, dict):
            raise TypeError(
                "checkpoint field 'engine_configuration' must be a mapping."
            )
        if stored != self.configuration:
            raise ValueError(
                "checkpoint configuration does not match this training engine."
            )
        return state


def mapping(state: dict[str, Any], name: str) -> dict[str, Any]:
    """
    Read one required mapping from a checkpoint with a concise failure mode.
    """
    value = state.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint field {name!r} must be a dictionary.")
    return value


def typed_list[T](state: dict[str, Any], name: str, item_type: type[T]) -> list[T]:
    """
    Restore one required list only when every semantic record has the expected type.
    """
    value = state.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, item_type) for item in value
    ):
        raise TypeError(f"checkpoint field {name!r} has invalid record types.")
    return value
