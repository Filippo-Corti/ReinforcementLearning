from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_ROOT / "docs" / "EXPERIMENT.md"

# A row of the freeze table: a path in backticks or a link, then a SHA-256.
_ROW = re.compile(
    r"^\|\s*(?:\[)?`([^`]+)`(?:\]\([^)]*\))?\s*\|[^|]*\|\s*`([0-9a-f]{64})`\s*\|$"
)


def _frozen_rows() -> list[tuple[str, str]]:
    """
    Read the frozen-document table out of the protocol.
    """
    text = PROTOCOL.read_text(encoding="utf-8")
    start = text.index("## Frozen protocol")
    section = text[start : text.index("\n## ", start + 1)]
    return [
        (match.group(1), match.group(2))
        for line in section.splitlines()
        if (match := _ROW.match(line))
    ]


def _checksum(path: Path) -> str:
    """
    Hash a file's newline-normalized bytes, as the protocol says it does.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_freeze_table_lists_every_contract_document() -> None:
    """
    The freeze is only a constraint if it covers what defines the experiments.
    """
    named = {name for name, _ in _frozen_rows()}

    assert named == {
        "MDP.md",
        "LEARNING.md",
        "TRACK.md",
        "tracks/experiment_1.json",
        "tracks/experiment_2_splits.json",
    }


@pytest.mark.parametrize("name, expected", _frozen_rows())
def test_a_frozen_document_still_matches_its_recorded_checksum(
    name: str, expected: str
) -> None:
    """
    Fail when a frozen document changes without the freeze being re-issued.

    This is the whole point of recording the checksums: a protocol that says it
    is frozen while its own definitions drift underneath it is worse than one
    that never claimed to be. Editing a document here is not forbidden — it
    requires a dated amendment and a new checksum, which is a deliberate act
    rather than an accident.
    """
    path = PROJECT_ROOT / ("docs" / Path(name) if name.endswith(".md") else Path(name))

    assert path.is_file(), f"{name} is frozen but missing"
    assert _checksum(path) == expected, (
        f"{name} changed since the protocol was frozen. If that was deliberate, "
        f"amend docs/EXPERIMENT.md with a dated revision and update the checksum "
        f"to {_checksum(path)}."
    )
