"""Non-overlapping component timing shared by every training engine."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from recording.records import MetricScope, RunCategory, TimingRecord


class TrainingTimer:
    """
    Timer that keeps track of how long a training run spends in each phase.
    The categories are mutually exclusive by construction: a phase is timed by
    the block that performs it, and blocks do not nest.

    Fields:
        * collection: Seconds spent stepping environments and choosing actions.
        * optimization: Seconds spent inside agent updates.
        * evaluation: Seconds spent on deterministic evaluation episodes.
        * persistence: Seconds spent writing checkpoints and records.
    """

    def __init__(self) -> None:
        self.collection = 0.0
        self.optimization = 0.0
        self.evaluation = 0.0
        self.persistence = 0.0
        self._started_at = perf_counter()

    @contextmanager
    def collecting(self) -> Iterator[None]:
        """
        Time one block of environment interaction.
        """
        started = perf_counter()
        try:
            yield
        finally:
            self.collection += perf_counter() - started

    @contextmanager
    def optimizing(self) -> Iterator[ElapsedTimeRecorder]:
        """
        Time one agent update, yielding a handle that reports its own duration.

        The duration is yielded because an update is recorded individually as
        well as accumulated, and measuring it twice would be measuring two
        different things.
        """
        elapsed = ElapsedTimeRecorder(perf_counter())
        try:
            yield elapsed
        finally:
            self.optimization += elapsed.seconds

    @contextmanager
    def evaluating(self) -> Iterator[None]:
        """
        Time one block of deterministic evaluation.
        """
        started = perf_counter()
        try:
            yield
        finally:
            self.evaluation += perf_counter() - started

    @contextmanager
    def persisting(self) -> Iterator[None]:
        """
        Time one block of checkpoint or record writing.
        """
        started = perf_counter()
        try:
            yield
        finally:
            self.persistence += perf_counter() - started

    def record(self, run_category: RunCategory) -> TimingRecord:
        """
        Return the accumulated categories beside the elapsed end-to-end time.
        """
        return TimingRecord(
            run_category=run_category,
            scope=MetricScope.TRAINING,
            collection=self.collection,
            optimization=self.optimization,
            evaluation=self.evaluation,
            persistence=self.persistence,
            end_to_end=perf_counter() - self._started_at,
        )

    def state(self) -> dict[str, float]:
        """
        Return the accumulated categories for a checkpoint.
        """
        return {
            "collection": self.collection,
            "optimization": self.optimization,
            "evaluation": self.evaluation,
            "persistence": self.persistence,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """
        Resume accumulating from a checkpoint's totals.
        """
        self.collection = float(state["collection"])
        self.optimization = float(state["optimization"])
        self.evaluation = float(state["evaluation"])
        self.persistence = float(state["persistence"])


class TrainingProgress:
    """
    Progress bar that reports progress over the training interactions.
    A two-million-step run gives no sign of life for tens of minutes otherwise.

    The bar is opt-in because the same engines run under scripts that capture
    output, where a redrawing bar is noise rather than information.

    Fields:
        * enabled: Whether a bar is being drawn at all.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._bar: object | None = None
        self._position = 0

    def start(self, position: int, budget: int) -> None:
        """
        Open a bar covering the interactions still to be collected.
        """
        self.close()
        self._position = position
        if not self.enabled or budget <= position:
            return
        from tqdm.auto import tqdm

        self._bar = tqdm(
            total=budget - position,
            desc="Training interactions",
            unit="interaction",
        )

    def advance(self, position: int) -> None:
        """
        Move the bar to a new absolute interaction count.
        """
        if self._bar is None:
            return
        self._bar.update(position - self._position)  # type: ignore[attr-defined]
        self._position = position

    def close(self) -> None:
        """
        Finish the bar, leaving the completed line in place.
        """
        if self._bar is not None:
            self._bar.close()  # type: ignore[attr-defined]
            self._bar = None


class ElapsedTimeRecorder:
    """
    Utility class for the TrainingTimer.
    It reports a duration since a given start time, and then holds that value for later reads.

    Fields:
        * seconds: Duration of the block, fixed at first read so both uses agree.
    """

    def __init__(self, started: float) -> None:
        self._started = started
        self._seconds: float | None = None

    @property
    def seconds(self) -> float:
        """
        Return the duration, measured once and then held.
        """
        if self._seconds is None:
            self._seconds = perf_counter() - self._started
        return self._seconds
