"""Show recorded analysis tables and figures inside a notebook.

The analysis writes every table as JSON and CSV and every figure as a PNG, and
those files are the authoritative result. This only reads them back and renders
them, so a notebook never recomputes a number it is reporting and cannot
disagree with the files it was generated from.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import HTML, Image, display

TableRow = Mapping[str, Any]


def read_table(directory: str | Path, name: str) -> list[dict[str, Any]]:
    """
    Read the rows of one analysis table, by name and without its extension.

    Each table file wraps its rows alongside a schema version, so the version
    is checked on the way in rather than left for a reader to notice: a table
    written by a different contract is a different table.
    """
    path = Path(directory) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"the analysis produced no table named {name!r}.")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "rows" not in document:
        raise ValueError(f"table {name!r} is not a recorded analysis table.")
    return list(document["rows"])


def show_table(
    rows: Sequence[TableRow],
    *,
    columns: Sequence[str] | None = None,
    title: str | None = None,
    sort_by: str | Sequence[str] | None = None,
    where: Callable[[TableRow], bool] | None = None,
    limit: int | None = None,
    precision: int = 3,
) -> None:
    """
    Render rows as a table, keeping only the columns worth looking at.

    Analysis tables carry every field any consumer might need, which is far
    more than a reader can take in at once. Naming the columns is how each
    section of a notebook says what its own question is about.
    """
    selected = [row for row in rows if where is None or where(row)]
    if sort_by is not None:
        keys = (sort_by,) if isinstance(sort_by, str) else tuple(sort_by)
        selected = sorted(
            selected, key=lambda row: tuple(_sortable(row.get(key)) for key in keys)
        )
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        display(HTML(f"<p><em>{escape(title or 'table')}: no rows.</em></p>"))
        return
    names = list(columns) if columns is not None else list(selected[0])
    header = "".join(
        f"<th style='text-align:left'>{escape(name)}</th>" for name in names
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td style='text-align:left'>{escape(_format(row.get(name), precision))}</td>"
            for name in names
        )
        + "</tr>"
        for row in selected
    )
    caption = (
        f"<caption style='text-align:left;font-weight:600;padding:4px 0'>{escape(title)}</caption>"
        if title
        else ""
    )
    display(
        HTML(
            "<table style='border-collapse:collapse;font-size:0.9em'>"
            f"{caption}<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        )
    )


def show_figure(directory: str | Path, name: str, *, width: int = 980) -> None:
    """
    Display one analysis figure by name, without its extension.
    """
    path = Path(directory) / f"{name}.png"
    if not path.is_file():
        display(HTML(f"<p><em>no figure named {escape(name)}.</em></p>"))
        return
    display(Image(filename=str(path), width=width))


def describe(values: Iterable[float], *, precision: int = 3) -> str:
    """
    Summarize a handful of root values as text, sample size included.

    Five roots cannot support a confident interval, so the count travels with
    the numbers rather than being left for a reader to assume.
    """
    numbers = [float(value) for value in values]
    if not numbers:
        return "no values"
    mean = sum(numbers) / len(numbers)
    if len(numbers) > 1:
        variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
    else:
        variance = 0.0
    return (
        f"n={len(numbers)}  mean={mean:.{precision}f}  sd={variance ** 0.5:.{precision}f}  "
        f"min={min(numbers):.{precision}f}  max={max(numbers):.{precision}f}"
    )


def _format(value: Any, precision: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:,.{precision}f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format(item, precision) for item in value)
    return str(value)


def _sortable(value: Any) -> tuple[int, float, str]:
    if value is None:
        return (2, 0.0, "")
    if isinstance(value, bool):
        return (0, float(value), "")
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (1, 0.0, str(value))
