"""Feature construction.

Built against the real schema (see notes/2026-09-04-data-findings.md), not the
website's description of it.

The one structural fact that shapes everything here: a missing flight-plan row
matters enormously, but **only at one airport**. `LIRF + no flight plan` is 1,488
rows, 0.071% of the data, holding 37.4% of the target's total variance with a
conditional mean of 6,531s. Away from Rome a missing flight plan is
unremarkable — 1,019s against a global 991s. So the signal is the *interaction*,
carried by ``airport_plan``, not ``has_flight_plan`` on its own; the flag alone
averages Rome together with 21,000 harmless rows.

It is left as a general per-airport crossing rather than a hard-coded LIRF flag:
it is the same information without hand-picking an airport, the other nine get
to contribute, and it does not silently break if the fault moves or a second
airport develops one. The missingness of every ``_flt`` column is preserved
rather than imputed.

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
    "airport_plan",
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
    "dep_rwy_30min",
    "dep_rwy_headway",
    "arr_taxi_60min",
    "sched_demand_30min",
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


_MAX_HEADWAY_S = 6 * 3600


def _prev_gap(times: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Seconds since the previous movement sharing the same key. -1 if first."""
    out = np.full(len(times), -1.0)
    order = np.lexsort((times, keys))
    k, t = keys[order], times[order]
    same = np.empty(len(order), dtype=bool)
    same[0] = False
    same[1:] = k[1:] == k[:-1]
    gaps = np.empty(len(order))
    gaps[0] = -1.0
    gaps[1:] = np.where(same[1:], t[1:] - t[:-1], -1.0)
    out[order] = gaps
    return out


def _trailing_mean(
    query_times: np.ndarray,
    query_keys: np.ndarray,
    src_times: np.ndarray,
    src_keys: np.ndarray,
    src_values: np.ndarray,
    window_s: int,
) -> np.ndarray:
    """Mean of src_values over the ``window_s`` seconds before each query time.

    Strictly backward-looking, so it stays honest: only movements that had
    already happened can inform a prediction. NaN where the window is empty.
    """
    out = np.full(len(query_times), np.nan)
    for key in np.unique(query_keys):
        q = np.flatnonzero(query_keys == key)
        s = np.flatnonzero(src_keys == key)
        if not len(q) or not len(s):
            continue
        order = np.argsort(src_times[s], kind="stable")
        st, sv = src_times[s][order], src_values[s][order]
        csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(sv))])
        ccnt = np.concatenate([[0.0], np.cumsum(~np.isnan(sv))])
        qt = query_times[q]
        hi = np.searchsorted(st, qt, side="right")
        lo = np.searchsorted(st, qt - window_s, side="left")
        n = ccnt[hi] - ccnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[q] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    return out


def _wave2(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Congestion the earlier features missed: runway-level, and ground state.

    ``arr_taxi_60min`` is the mean taxi-in of arrivals that landed at this
    airport in the previous hour. It is the closest thing available to a direct
    reading of how congested the surface actually is right now, and it is honest
    at prediction time: ranking.parquet blanks the departure taxi times but
    leaves the whole arrival side intact.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())
    runway = frame["RUNWAY_mvt"].to_numpy().astype(str)
    apt_rwy = np.char.add(np.char.add(airport.astype(str), "/"), runway)

    dep = phase == "DEP"
    arr = phase == "ARR"
    taxi = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()

    out = {}
    # Same-runway departure pressure, and headway to the previous departure.
    rwy_counts = np.zeros(len(epoch), dtype=np.int32)
    if dep.any():
        rwy_counts[dep] = _window_counts(epoch[dep], apt_rwy[dep], 900)
    out["dep_rwy_30min"] = rwy_counts

    # Anything beyond a few hours is not a headway, it is "nothing recent", and
    # the distinction matters at prediction time: the ranking file holds January
    # and July in one frame, so the first July departure on a runway would
    # otherwise measure back five months. 85 ranking rows exceeded seven days
    # against 10 in a training month. NaN says "no recent departure" in a way
    # that means the same thing in both frames.
    headway = np.full(len(epoch), np.nan)
    if dep.any():
        gaps = _prev_gap(epoch[dep].astype(float), apt_rwy[dep])
        gaps[(gaps < 0) | (gaps > _MAX_HEADWAY_S)] = np.nan
        headway[dep] = gaps
    out["dep_rwy_headway"] = headway

    # Ground state, read off the arrivals that have already landed.
    out["arr_taxi_60min"] = _trailing_mean(
        epoch.astype(float), airport, epoch[arr].astype(float), airport[arr], taxi[arr], 3600
    )

    # Planned demand, from scheduled times rather than achieved ones.
    sched = frame["SCHED_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    sched_counts = np.zeros(len(epoch), dtype=np.int32)
    if dep.any():
        sched_counts[dep] = _window_counts(sched[dep], airport[dep], 900)
    out["sched_demand_30min"] = sched_counts
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
    congestion.update(_wave2(frame))
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

    # Per-airport crossing of the flight-plan flag. See the module docstring:
    # a missing flight plan is only dangerous at some airports, so the model
    # needs the pair, not the flag.
    frame = frame.with_columns(
        (pl.col("ADEP_mvt") + pl.lit("|") + pl.col("has_flight_plan").cast(pl.Utf8)).alias(
            "airport_plan"
        )
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
