"""How much memory a run actually needed, where the platform will say.

Peak memory is a reported experiment outcome, so it is measured rather than
estimated from what the configuration implies a run should cost.
"""

from __future__ import annotations

import sys

import psutil


def peak_process_memory() -> int | None:
    """
    Return this process's peak resident memory in bytes, or `None` if unknown.

    This is the training process: the one holding the networks, the optimizers
    and the collected experience. Environment workers are separate processes
    and are deliberately not included. Their peak cannot be observed from here
    once they exit, and substituting their *current* usage would report a
    different quantity under a name that says peak.

    `None` is returned rather than a zero or a guess when no platform interface
    supplies the number, so a missing measurement is visible in the record as a
    missing measurement.
    """
    # Windows reports the peak directly; `resource` does not exist there, and
    # the check is written against `sys.platform` so a type checker can see
    # that the POSIX-only import below is unreachable on it.
    peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
    if peak is not None:
        return int(peak)
    if sys.platform == "win32":
        return None

    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes here; the BSDs, including macOS, report bytes.
    return int(maximum) * 1024 if sys.platform.startswith("linux") else int(maximum)
