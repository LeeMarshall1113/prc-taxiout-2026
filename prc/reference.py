"""EUROCONTROL reference (unimpeded) taxi-out time.

Their published recipe, from `Additional taxi-out time performance indicator
document`, Edition 01.00, 16-Mar-2023 (PRU/AIU):

- reference taxi-out is the **10th percentile** of taxi-out time within a
  (departure stand, departure runway) combination,
- computed over a rolling 12-month window,
- and used only where at least **10 flights** in that window have a taxi-out at
  or below the percentile. Combinations failing that are excluded from their
  indicator entirely.

We keep the percentile and the validity floor and replace the exclusion with a
back-off, because a model has to predict every row: stand+runway, else runway,
else airport. ``ref_level`` records which one was used, so the model can learn
how much to trust it.

Aircraft type is deliberately NOT part of the grouping. That is EUROCONTROL's
choice and the reason is sample size per combination; we follow it.

**This is a target-derived feature, so it is fitted, never computed in place.**
Fit on training rows only and apply outward — otherwise a row's own taxi time
leaks into its own predictor. ``fit`` takes the training split; ``apply`` takes
anything.
"""

from __future__ import annotations

import polars as pl

PERCENTILE = 0.10
MIN_AT_OR_BELOW = 10
TARGET = "TAXITIME_SEC_mvt"


def _level(frame: pl.DataFrame, keys: list[str], name: str) -> pl.DataFrame:
    """Percentile plus the count at or below it, per group."""
    ref = frame.group_by(keys).agg(
        pl.col(TARGET).quantile(PERCENTILE, interpolation="linear").alias(name),
        pl.len().alias(f"{name}_rows"),
    )
    at_or_below = (
        frame.join(ref, on=keys, how="left")
        .filter(pl.col(TARGET) <= pl.col(name))
        .group_by(keys)
        .agg(pl.len().alias(f"{name}_n"))
    )
    return ref.join(at_or_below, on=keys, how="left").with_columns(
        pl.col(f"{name}_n").fill_null(0)
    )


def fit(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Build the three back-off levels from rows that carry a target."""
    frame = frame.filter(pl.col(TARGET).is_not_null())
    return {
        "pair": _level(frame, ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], "ref_pair"),
        "runway": _level(frame, ["ADEP_mvt", "RUNWAY_mvt"], "ref_runway"),
        "airport": _level(frame, ["ADEP_mvt"], "ref_airport"),
    }


def apply(frame: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Attach ``ref_taxi_s`` and ``ref_level``, backing off where sparse."""
    out = (
        frame.join(tables["pair"], on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
        .join(tables["runway"], on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        .join(tables["airport"], on=["ADEP_mvt"], how="left")
    )
    # A level counts only if it clears EUROCONTROL's validity floor.
    ok = lambda name: pl.col(f"{name}_n") >= MIN_AT_OR_BELOW  # noqa: E731
    return out.with_columns(
        pl.when(ok("ref_pair")).then(pl.col("ref_pair"))
        .when(ok("ref_runway")).then(pl.col("ref_runway"))
        .otherwise(pl.col("ref_airport"))
        .alias("ref_taxi_s"),
        pl.when(ok("ref_pair")).then(pl.lit("pair"))
        .when(ok("ref_runway")).then(pl.lit("runway"))
        .otherwise(pl.lit("airport"))
        .alias("ref_level"),
    ).drop([c for c in out.columns if c.startswith(("ref_pair", "ref_runway", "ref_airport"))])


def attach(train: pl.DataFrame, *frames: pl.DataFrame) -> list[pl.DataFrame]:
    """Fit on ``train`` and apply to every frame given, ``train`` included.

    Training rows do get a reference partly derived from themselves, which is
    what EUROCONTROL's own rolling window does too. It is tolerable here because
    the validity floor means a pair-level reference is backed by roughly a
    hundred flights, so any single row moves its own percentile very little.
    Rows that would be backed by fewer than that fall through to the runway or
    airport level, which is coarser still.
    """
    tables = fit(train)
    return [apply(f, tables) for f in (train, *frames)]
