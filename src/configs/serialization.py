"""Stable conversion of configuration dataclasses to plain data."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


class SerializableConfig:
    """
    Mixin that gives immutable configuration dataclasses stable serialization.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic plain-dictionary representation.
        """
        return to_plain_dict(self)


def to_plain_dict(configuration: object) -> dict[str, Any]:
    """
    Convert a configuration dataclass to JSON-compatible deterministic data.
    """
    plain_value = _to_plain_value(configuration)
    if not isinstance(plain_value, dict):
        raise TypeError("A configuration must serialize to a dictionary.")
    return plain_value


def _to_plain_value(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_plain_value(item) for item in value]
    return value
