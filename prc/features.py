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
    "callsign_op",
    "airport_plan",
    "ref_level",
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
    "mvt_sec_00",
    "dep_30min",
    "arr_30min",
    "dep_60min",
    "arr_60min",
    "dep_rwy_30min",
    "dep_rwy_headway",
    "arr_taxi_60min",
    "sched_demand_30min",
    "rwy_dep_active",
    "rwy_dep_share",
    "rwy_arr_active",
    "rwy_mixed_mode",
    "prev_stand_gap",
    "prev_stand_headway",
    "prev_rwy_gap",
    "stand_runway_pair_n",
    "ref_taxi_s",
    *__import__("prc.weather", fromlist=["FEATURES"]).FEATURES,
]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "TAXITIME_SEC_mvt"

# Fitted on training rows by prc.reference and attached afterwards, so build()
# cannot produce them and must not try to select them.
REFERENCE = ["ref_taxi_s", "ref_level"]


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


def _sequence_proxy(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """What the PREVIOUS departure from this stand, and this runway, experienced.

    A flight's own (MVT - AOBT_3) gap is its best single predictor at r=0.536,
    but it is null for the rows that hurt us most. The neighbouring flights'
    gaps are not: they come from other rows, and every input is present in
    ranking.parquet -- MVT_TIME 0% null, AOBT_3 1.53%, STAND and RUNWAY ~0%.

    Measured: the previous same-stand departure's gap correlates 0.27 with this
    row's target overall and 0.38 on flight-plan-present non-LIRF rows, against
    0.11 for the shipped arrival-taxi feature. Inside the no-flight-plan group,
    which has no timestamp features at all because every _flt column is null on
    the row itself, it still carries 0.15 to 0.18.

    Strictly backward-looking: only departures that have already taken off.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())
    gap = (frame["MVT_TIME_UTC_mvt"] - frame["AOBT_3_flt"]).dt.total_seconds().to_numpy().astype(float)

    n = len(epoch)
    out = {k: np.full(n, np.nan) for k in
           ("prev_stand_gap", "prev_stand_headway", "prev_rwy_gap")}
    dep = phase == "DEP"

    for field, gap_key, head_key in (
        ("STAND_mvt", "prev_stand_gap", "prev_stand_headway"),
        ("RUNWAY_mvt", "prev_rwy_gap", None),
    ):
        key = np.char.add(np.char.add(airport.astype(str), "|"),
                          frame[field].to_numpy().astype(str))
        for k in np.unique(key[dep]):
            i = np.flatnonzero(dep & (key == k))
            if len(i) < 2:
                continue
            o = i[np.argsort(epoch[i], kind="stable")]
            # shift by one: each row sees only its predecessor
            head = (epoch[o[1:]] - epoch[o[:-1]]).astype(float)
            # ...but only if the predecessor is recent enough to mean anything.
            # ranking.parquet holds January and July of 2026 in one frame, so
            # July's first departure from a stand otherwise inherits January's,
            # five months back, and a headway no training month can produce.
            # _wave2 caps for exactly this reason; this must too.
            stale = head > _MAX_HEADWAY_S
            prev = gap[o[:-1]].copy()
            prev[stale] = np.nan
            head[stale] = np.nan
            out[gap_key][o[1:]] = prev
            if head_key:
                out[head_key][o[1:]] = head
    return out


def _runway_config(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Which runways the airport is actually working, around each departure.

    An airport's configuration -- how many runways are open for departures, how
    the traffic is split across them, and whether departures and arrivals share
    a runway -- changes with wind and time of day and rewrites taxi routing for
    everyone on the field. We know each flight's own assigned runway but have
    had no view of the configuration it sits inside.

    Unlike the rejected reference-taxi feature, this is time-varying and
    airport-wide rather than a static function of a row's own (stand, runway),
    so CatBoost cannot already be recovering it from the raw categoricals.

    Reconstructed from movements we already hold, on both sides: RUNWAY_mvt is
    100% populated on arrival rows in ranking.parquet as well as in training.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    # ARR rows are keyed on ADES: only 16% of them have ADEP among the ten.
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())
    runway = frame["RUNWAY_mvt"].to_numpy().astype(str)

    n = len(epoch)
    out = {k: np.full(n, np.nan) for k in
           ("rwy_dep_active", "rwy_dep_share", "rwy_arr_active", "rwy_mixed_mode")}
    half = 1800  # +/- 30 minutes

    for apt in np.unique(airport):
        at_apt = airport == apt
        rwys = np.unique(runway[at_apt])
        index = {r: i for i, r in enumerate(rwys)}

        for want, prefix in (("DEP", "dep"), ("ARR", "arr")):
            side = at_apt & (phase == want)
            if not side.any():
                continue
            order = np.argsort(epoch[side], kind="stable")
            times = epoch[side][order]
            codes = np.array([index[r] for r in runway[side][order]])
            # cumulative count per runway, so any window is one subtraction
            onehot = np.zeros((len(times) + 1, len(rwys)), dtype=np.int32)
            onehot[np.arange(1, len(times) + 1), codes] = 1
            cum = np.cumsum(onehot, axis=0)

            q = np.flatnonzero(at_apt & (phase == "DEP"))
            qt = epoch[q]
            lo = np.searchsorted(times, qt - half, side="left")
            hi = np.searchsorted(times, qt + half, side="right")
            counts = cum[hi] - cum[lo]                      # (len(q), len(rwys))
            active = (counts > 0).sum(axis=1)

            if want == "DEP":
                total = counts.sum(axis=1)
                own = np.array([index[r] for r in runway[q]])
                mine = counts[np.arange(len(q)), own]
                out["rwy_dep_active"][q] = active
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["rwy_dep_share"][q] = np.where(total > 0, mine / np.maximum(total, 1), np.nan)
            else:
                out["rwy_arr_active"][q] = active
                own = np.array([index[r] for r in runway[q]])
                # Is this departure's runway also taking arrivals? Mixed mode
                # costs departures time that segregated operation does not.
                out["rwy_mixed_mode"][q] = (counts[np.arange(len(q)), own] > 0).astype(float)
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
    congestion.update(_runway_config(frame))
    congestion.update(_sequence_proxy(frame))
    frame = frame.with_columns(
        [pl.Series(name, values) for name, values in congestion.items()]
    )

    mvt = pl.col("MVT_TIME_UTC_mvt")
    frame = frame.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        mvt.dt.hour().alias("hour"),
        # .dt.hour() is Int8 and polars keeps that dtype through the multiply,
        # so hour*60 overflows above 127 and wraps: 21:10 became -10, not 1270.
        # 99.7% of rows carried a wrong value; every model so far trained on it.
        (mvt.dt.hour().cast(pl.Int32) * 60 + mvt.dt.minute()).alias("minute_of_day"),
        mvt.dt.weekday().alias("dow"),
        mvt.dt.month().alias("month"),
        mvt.dt.ordinal_day().alias("doy"),
        (mvt.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("AOBT_3_flt").is_not_null().cast(pl.Int8).alias("has_flight_plan"),
        # Operator identity, recovered from the callsign.
        #
        # AIRCRAFT_OPERATOR_flt is 100% null for the no-flight-plan rows -- the
        # group that carries most of our squared error has no operator column at
        # all. But FLIGHT_mvt sits on the movement side and is 99.99% populated
        # in training and 99.98% in ranking.parquet, and its leading letters are
        # the operator: matched against AIRCRAFT_OPERATOR_flt on the 2,046,656
        # rows where both exist, the prefix agrees with the modal operator
        # 99.48% of the time. 99.88% of ranking prefixes were seen in training.
        #
        # Within the no-plan rows this is worth eta^2 0.069 on the residual
        # after removing airport x aircraft type, so it is not a restatement of
        # either. Handed to CatBoost raw, as a categorical: a hand-rolled target
        # encoding is the mistake ref_taxi_s made twice.
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{2,3})", 1).alias("callsign_op"),
        (mvt - pl.col("AOBT_3_flt")).dt.total_seconds().alias("gap_aobt"),
        (mvt - pl.col("EOBT_1_flt")).dt.total_seconds().alias("gap_eobt"),
        (mvt - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap_sched"),
        (mvt - pl.col("LOBT_flt")).dt.total_seconds().alias("gap_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("dep_delay"),
        (pl.col("SCHED_TIME_UTC_mvt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("sched_vs_eobt"),
        # A sensor reading has uniform seconds; a hand-entered or schedule-copied
        # one piles up on :00. BLOCK_TIME would say more here but is withheld at
        # serve time, so only the movement side is usable.
        (mvt.dt.second() == 0).cast(pl.Int8).alias("mvt_sec_00"),
    )

    from . import weather

    frame = weather.attach(frame)

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
    # Grouped by calendar month as well as by key. Training calls build() once
    # per monthly file, but the real serve-time call passes the whole of
    # ranking.parquet, which is January AND July 2026 -- 62 days in one frame.
    # Dividing by 62 while training divided by 31 halves every key that only
    # appears in one of the two months and biases the rest; measured, only 3.3%
    # of served rows came within 1% of what a single-month build gives, and
    # 13,446 were served exactly half. Grouping by month makes the two agree.
    frame = frame.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.truncate("1mo").alias("_ym")
    )
    days = frame.group_by("_ym").agg(
        pl.col("MVT_TIME_UTC_mvt").dt.date().n_unique().alias("_days")
    )
    pair = frame.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "_ym").agg(
        pl.len().alias("_n")
    ).join(days, on="_ym", how="left").with_columns(
        (pl.col("_n") / pl.max_horizontal(pl.col("_days"), pl.lit(1)))
        .alias("stand_runway_pair_n")
    ).select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "_ym", "stand_runway_pair_n")
    frame = frame.join(
        pair, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "_ym"], how="left"
    ).drop("_ym")


    built = [f for f in FEATURES if f not in REFERENCE]
    keep = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", *built]
    keep += [TARGET] if with_target else []
    frame = frame.select(list(dict.fromkeys(keep)))
    cats = [c for c in CATEGORICAL if c not in REFERENCE]
    return frame.with_columns([pl.col(c).cast(pl.Utf8).fill_null("__NA__") for c in cats])
