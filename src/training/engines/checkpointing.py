"""Versioned engine checkpoints that refuse to resume onto the wrong run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..checkpoints import load_checkpoint, save_checkpoint

ENGINE_STATE_VERSION = 3


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
