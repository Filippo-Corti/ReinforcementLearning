"""Run a design matrix of training runs, resumably.

An experiment here is dozens of independent runs that together take hours, so
the loop that drives them has to survive being interrupted. It does that by
treating the results tree as the record of what has been done: a run whose
`completion.json` exists is finished and is skipped, and everything else is
started from scratch.

Nothing about the experiments themselves lives here. The matrices, their
configurations and their meaning belong to the notebooks that define them; this
only knows how to execute a list of specifications and report what happened.
"""

from __future__ import annotations

import shutil
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True, slots=True)
class RunSpecification:
    """
    Name one run of a design matrix and say how to start it.

    Fields:
        * run_id: Identity of the run, unique within its matrix.
        * path: Directory the run writes its records into.
        * launch: Starts the run; called with no arguments.
    """

    run_id: str
    path: Path
    launch: Callable[[], object]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """
    Report what happened to one specification.

    Fields:
        * run_id: Identity of the run this describes.
        * status: One of `completed`, `skipped` or `failed`.
        * duration: Wall seconds spent, zero for a skipped run.
        * error: Formatted traceback when the status is `failed`.
    """

    run_id: str
    status: str
    duration: float
    error: str | None = None


def is_complete(path: Path) -> bool:
    """
    Return whether a run directory holds a finished run.

    Completion is the last thing a run writes, so its presence means every
    record before it was written too. A directory without it is the debris of
    an interrupted run rather than a result.
    """
    return (Path(path) / "completion.json").is_file()


def execute(
    specifications: Sequence[RunSpecification],
    *,
    skip_complete: bool = True,
    report: Callable[[str], None] = print,
) -> list[RunOutcome]:
    """
    Run every specification that is not already finished.

    A failing run does not stop the matrix. Overnight, one bad specification
    ending a nine-hour queue would be worse than finishing the rest and looking
    at the failure in the morning, and because finished runs are skipped, the
    repaired specification can simply be run again. The caller is handed every
    outcome and decides what a failure means.

    An incomplete directory is deleted before its run restarts. The recorder
    refuses to write into a non-empty directory, and that debris is by
    definition a run that produced no result.
    """
    outcomes: list[RunOutcome] = []
    total = len(specifications)
    for index, specification in enumerate(specifications, start=1):
        prefix = f"[{index}/{total}] {specification.run_id}"
        if skip_complete and is_complete(specification.path):
            report(f"{prefix}: already complete, skipped")
            outcomes.append(RunOutcome(specification.run_id, "skipped", 0.0))
            continue
        if specification.path.exists():
            report(f"{prefix}: clearing an incomplete directory")
            shutil.rmtree(specification.path)
        report(f"{prefix}: running")
        started = perf_counter()
        try:
            specification.launch()
        # Deliberately blind: a matrix runner cannot know what a run might
        # raise, and the whole point is that one failure does not end the
        # queue. The traceback is kept and reported rather than swallowed.
        except Exception:  # noqa: BLE001
            duration = perf_counter() - started
            report(f"{prefix}: FAILED after {duration / 60:.1f} min")
            outcomes.append(
                RunOutcome(
                    specification.run_id, "failed", duration, traceback.format_exc()
                )
            )
            continue
        duration = perf_counter() - started
        report(f"{prefix}: done in {duration / 60:.1f} min")
        outcomes.append(RunOutcome(specification.run_id, "completed", duration))
    return outcomes


def summarize(
    outcomes: Iterable[RunOutcome], report: Callable[[str], None] = print
) -> int:
    """
    Print one line per status and return how many runs failed.
    """
    outcomes = list(outcomes)
    counts = {status: 0 for status in ("completed", "skipped", "failed")}
    for outcome in outcomes:
        counts[outcome.status] += 1
    spent = sum(outcome.duration for outcome in outcomes)
    report(
        f"{counts['completed']} completed, {counts['skipped']} skipped, "
        f"{counts['failed']} failed, {spent / 60:.1f} min spent"
    )
    for outcome in outcomes:
        if outcome.status == "failed":
            report(f"\n--- {outcome.run_id} ---\n{outcome.error}")
    return counts["failed"]
