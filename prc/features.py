"""Feature construction.

Built against the real schema (see notes/2026-09-04-data-findings.md), not the
website's description of it.

The one structural fact that shapes everything here: **1.08% of departures have
no matching flight-plan row**, and those rows hold 41.8% of the target's total
variance. Their conditional distribution is a different animal — mean 1384s and
sd 3398s, against 987s and 418s for the rest. ``has_flight_plan`` is therefore
not a nuisance indicator, it is the most important feature in the set, and the
missingness of every ``_flt`` column is left as missingness rather than imputed.

Nothing here uses ``BLOCK_TIME_UTC_mvt`` or ``TAXITIME_SEC_mvt``: both are
blanked for departures in the ranking file. ``MVT_TIME_UTC_mvt`` (wheels-up) is
given for every row and is fair game.
"""

from __future__ import annotations

import numpy as np
import polars as pl

CATEGORICAL = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
    "FLIGHT_RULE_mvt",
]

NUMERIC = [
    "hour",
    "minute_of_day",
    "dow",
    "month",
    "doy",
    "is_weekend",
    "has_flight_plan",
    "gap_aobt",
    "gap_eobt",
    "gap_sched",
    "gap_lobt",
    "dep_delay",
    "sched_vs_eobt",
    "dep_30min",
    "arr_30min",
    "dep_60min",
    "arr_60min",
    "stand_runway_pair_n",
]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "TAXITIME_SEC_mvt"


def _window_counts(times: np.ndarray, airports: np.ndarray, half_window_s: int) -> np.ndarray:
    """For each row, how many movements share its airport within +/- the window.

    Counted over whichever movements are passed in, so callers control whether
    that means departures, arrivals or both. Uses searchsorted per airport, which
    is exact and cheap; a rolling join would need the frame sorted globally.
    """
    out = np.zeros(len(times), dtype=np.int32)
    order = np.argsort(airports, kind="stable")
    sorted_airports = airports[order]
    bounds = np.searchsorted(sorted_airports, np.unique(sorted_airports), side="left")
    bounds = np.append(bounds, len(sorted_airports))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        idx = order[start:stop]
        t = times[idx]
        t_order = np.argsort(t, kind="stable")
        t_sorted = t[t_order]
        lo = np.searchsorted(t_sorted, t_sorted - half_window_s, side="left")
        hi = np.searchsorted(t_sorted, t_sorted + half_window_s, side="right")
        counts = (hi - lo - 1).astype(np.int32)  # exclude self
        out[idx[t_order]] = counts
    return out


def _congestion(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Departure and arrival pressure around each movement's wheels-up time.

    Both are knowable at prediction time: the ranking file keeps MVT_TIME_UTC_mvt
    for departures and the whole arrival side untouched.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    # A movement's own airport: ADEP for departures, ADES for arrivals.
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())

    out = {}
    for half, label in ((900, "30min"), (1800, "60min")):
        for want, name in (("DEP", "dep"), ("ARR", "arr")):
            mask = phase == want
            counts = np.zeros(len(epoch), dtype=np.int32)
            if mask.any():
                sub = _window_counts(epoch[mask], airport[mask], half)
                # Every row gets the count of same-airport movements of this
                # phase; rows of the other phase are matched by interval search
                # against the same arrays.
                full = np.zeros(len(epoch), dtype=np.int32)
                full[mask] = sub
                if (~mask).any():
                    for apt in np.unique(airport):
                        src = mask & (airport == apt)
                        dst = (~mask) & (airport == apt)
                        if not src.any() or not dst.any():
                            continue
                        t_src = np.sort(epoch[src])
                        t_dst = epoch[dst]
                        full[dst] = (
                            np.searchsorted(t_src, t_dst + half, side="right")
                            - np.searchsorted(t_src, t_dst - half, side="left")
                        ).astype(np.int32)
                counts = full
            out[f"{name}_{label}"] = counts
    return out


def build(frame: pl.DataFrame, with_target: bool = True) -> pl.DataFrame:
    """Turn a raw movements frame into the model matrix (departures only).

    ``frame`` must contain both phases: arrivals are dropped from the output but
    are needed first to compute arrival pressure.
    """
    congestion = _congestion(frame)
    frame = frame.with_columns(
        [pl.Series(name, values) for name, values in congestion.items()]
    )

    mvt = pl.col("MVT_TIME_UTC_mvt")
    frame = frame.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        mvt.dt.hour().alias("hour"),
        (mvt.dt.hour() * 60 + mvt.dt.minute()).alias("minute_of_day"),
        mvt.dt.weekday().alias("dow"),
        mvt.dt.month().alias("month"),
        mvt.dt.ordinal_day().alias("doy"),
        (mvt.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("AOBT_3_flt").is_not_null().cast(pl.Int8).alias("has_flight_plan"),
        (mvt - pl.col("AOBT_3_flt")).dt.total_seconds().alias("gap_aobt"),
        (mvt - pl.col("EOBT_1_flt")).dt.total_seconds().alias("gap_eobt"),
        (mvt - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap_sched"),
        (mvt - pl.col("LOBT_flt")).dt.total_seconds().alias("gap_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("dep_delay"),
        (pl.col("SCHED_TIME_UTC_mvt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("sched_vs_eobt"),
    )

    # How busy this stand/runway combination is -- a crude proxy for apron
    # layout and taxi distance, which we have no geometry for.
    #
    # Expressed as movements per operating day, NOT a raw count. build() is
    # called per monthly file in training but on the whole ranking file at
    # prediction time, and the ranking file holds two months, so a raw count
    # would arrive about twice as large at serve time than at fit time. Dividing
    # by the number of distinct dates present makes the two agree.
    n_days = max(frame.select(pl.col("MVT_TIME_UTC_mvt").dt.date().n_unique()).item(), 1)
    pair = frame.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(
        (pl.len() / n_days).alias("stand_runway_pair_n")
    )
    frame = frame.join(pair, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")

    keep = ["MVT_ID_mvt", *FEATURES] + ([TARGET] if with_target else [])
    frame = frame.select(keep)
    return frame.with_columns([pl.col(c).cast(pl.Utf8).fill_null("__NA__") for c in CATEGORICAL])
