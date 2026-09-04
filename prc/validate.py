"""Validation split and scoring.

The ranking set is **January and July 2026**. Two months, six months apart:
one deep-winter (de-icing, low-visibility procedures, holiday traffic) and one
peak-summer (highest movement counts, ATFM regulation season). A random split
of 2025 measures neither, and a plain chronological tail measures only December.

So the honest local analogue is to hold out **January and July 2025** and train
on the other ten months. It reproduces the seasonal composition of the target
and the "predict a month you did not see" structure at the same time.

Known limitation, worth stating rather than papering over: it does *not*
reproduce the year gap. The real task extrapolates 2025 → 2026 through a year of
traffic growth and schedule change, and no split of 2025 alone can measure that.
Expect the leaderboard to sit worse than local validation, and treat the offset
as roughly constant rather than trying to correct it.

    from prc.validate import HOLDOUT_MONTHS, rmse, split_by_month
"""

from __future__ import annotations

from typing import Iterable

# (year-agnostic) months held out locally, mirroring the ranking set.
HOLDOUT_MONTHS: tuple[int, ...] = (1, 7)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root mean squared error — the competition metric, in seconds."""
    import numpy as np

    a = np.asarray(list(actual), dtype="float64")
    p = np.asarray(list(predicted), dtype="float64")
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    if a.size == 0:
        raise ValueError("empty input")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def split_by_month(frame, time_col: str, months: tuple[int, ...] = HOLDOUT_MONTHS):
    """Split a polars DataFrame into (train, holdout) on the month of ``time_col``."""
    import polars as pl

    month = pl.col(time_col).dt.month()
    return frame.filter(~month.is_in(months)), frame.filter(month.is_in(months))


def report(actual, predicted, label: str = "holdout") -> str:
    """One line per evaluation, so runs can be diffed against each other."""
    import numpy as np

    a = np.asarray(list(actual), dtype="float64")
    p = np.asarray(list(predicted), dtype="float64")
    residual = p - a
    return (
        f"{label}: n={a.size:,}  RMSE={rmse(a, p):.4f}s  "
        f"MAE={np.mean(np.abs(residual)):.2f}s  bias={np.mean(residual):+.2f}s  "
        f"p95|err|={np.percentile(np.abs(residual), 95):.1f}s"
    )
