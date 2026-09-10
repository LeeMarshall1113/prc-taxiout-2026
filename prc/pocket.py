"""The LIRF no-flight-plan pocket: a three-process mixture, fitted on 2025.

383 of the 344,841 scored rows are LIRF departures with no flight plan, and
they carry an estimated 57% of our board squared error. See
`notes/2026-09-10-the-lirf-pocket.md` for the derivation.

Inside that pocket the target is a mixture of three processes:

  S  the airport had no off-block reading and wrote SCHED into BLOCK, so the
     target is `gap_sched` EXACTLY -- not approximately, identically
  R  BLOCK was sensed but dated a day early, so the target is ~86,400 + a
     normal taxi
  N  an ordinary departure that happens to be missing its flight plan

Which one you are looking at is predictable from `gap_sched`, which is
serve-available. P(S) climbs 0.37 -> 0.59 -> 0.81 -> 0.92 across the bins below,
then falls to 0.27 in the top bin where R takes over.

Under squared loss the right prediction is the posterior mean, so for a row with
gap g in a bin with substitution rate p and non-substituted mean m:

    yhat = p * g + (1 - p) * m

which beats a per-bin constant because it uses the row's own g rather than the
bin's average.

This is a post-hoc patch to a finished submission, deliberately: it costs no
fit, it touches only rows selected by a rule fixed in advance, and it can be
measured against 2025's own conditional distributions before anything is sent.

    python -m prc.pocket --table
    python -m prc.pocket --measure submissions/jolly-lobster_v8.parquet
    python -m prc.pocket --patch submissions/jolly-lobster_v8.parquet \\
                         --out submissions/jolly-lobster_v9.parquet
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import RAW_DIR

# Fixed before any result was seen. The lowest bin starts at 3,600s because
# below that P(S) is low and the dedicated model already wins those rows; the
# default patch threshold is higher still.
BINS = [(3600, 7200), (7200, 14400), (14400, 28800), (28800, 60000), (60000, 10**9)]
MIN_GAP = 7200


def _pocket(path: str) -> pl.DataFrame:
    """LIRF departures with no flight plan, with gap_sched attached."""
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col("PHASE_mvt") == "DEP")
            & (pl.col("ADEP_mvt") == "LIRF")
            & pl.col("AOBT_3_flt").is_null()
        )
        .select("MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                *(["TAXITIME_SEC_mvt"] if "training" in path or "ranking" not in path else []))
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds().alias("gs")
        )
        .collect()
    )


def fit_table() -> dict[tuple[int, int], tuple[float, float, float, int]]:
    """Per-bin (P(substituted), mean of the rest, mean target, n) from 2025."""
    tr = (
        pl.scan_parquet(str(RAW_DIR / "training_*.parquet"))
        .filter(
            (pl.col("PHASE_mvt") == "DEP")
            & (pl.col("ADEP_mvt") == "LIRF")
            & pl.col("AOBT_3_flt").is_null()
        )
        .select("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds().alias("gs")
        )
        .collect()
    )
    y = tr["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    g = tr["gs"].to_numpy().astype(float)
    sub = np.abs(y - g) <= 60

    table = {}
    for lo, hi in BINS:
        m = (g >= lo) & (g < hi)
        if m.sum() < 5:
            continue
        rest = y[m & ~sub]
        table[(lo, hi)] = (
            float(sub[m].mean()),
            float(rest.mean()) if len(rest) else float(y[m].mean()),
            float(y[m].mean()),
            int(m.sum()),
        )
    return table


def _variance_by_bin() -> dict[tuple[int, int], float]:
    tr = (
        pl.scan_parquet(str(RAW_DIR / "training_*.parquet"))
        .filter(
            (pl.col("PHASE_mvt") == "DEP")
            & (pl.col("ADEP_mvt") == "LIRF")
            & pl.col("AOBT_3_flt").is_null()
        )
        .select("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds().alias("gs")
        )
        .collect()
    )
    y = tr["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    g = tr["gs"].to_numpy().astype(float)
    out = {}
    for lo, hi in BINS:
        m = (g >= lo) & (g < hi)
        if m.sum() >= 5:
            out[(lo, hi)] = float(y[m].var())
    return out


def mixture_prediction(gs: np.ndarray, table) -> np.ndarray:
    """p*g + (1-p)*m per bin; NaN where the row falls in no fitted bin."""
    out = np.full(len(gs), np.nan)
    for (lo, hi), (p, m, _ey, _n) in table.items():
        sel = (gs >= lo) & (gs < hi)
        out[sel] = p * gs[sel] + (1.0 - p) * m
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="store_true", help="print the fitted 2025 table")
    ap.add_argument("--measure", help="report the expected change to this submission")
    ap.add_argument("--patch", help="submission parquet to patch")
    ap.add_argument("--out", help="where to write the patched submission")
    ap.add_argument("--min-gap", type=float, default=MIN_GAP,
                    help=f"only patch rows with gap_sched at or above this (default {MIN_GAP})")
    args = ap.parse_args()

    table = fit_table()
    if args.table or not (args.measure or args.patch):
        print(f"{'gap_sched bin':>24s} {'n':>5s} {'P(sub)':>7s} {'E[y|~sub]':>11s} {'E[y]':>10s}")
        for (lo, hi), (p, m, ey, n) in table.items():
            print(f"  [{lo:>8,.0f},{hi:>10,.0f}) {n:5d} {p:7.2f} {m:11,.0f} {ey:10,.0f}")
        if args.table:
            return

    src = args.measure or args.patch
    rk = _pocket(str(RAW_DIR / "ranking.parquet"))
    sub_df = pl.read_parquet(src)
    target = [c for c in sub_df.columns if c != "MVT_ID_mvt"]
    if len(target) != 1:
        raise SystemExit(f"expected one prediction column in {src}, found {target}")
    tcol = target[0]

    j = rk.join(sub_df, on="MVT_ID_mvt", how="inner")
    gs = j["gs"].to_numpy().astype(float)
    cur = j[tcol].to_numpy().astype(float)
    new = mixture_prediction(gs, table)
    touch = (gs >= args.min_gap) & ~np.isnan(new)

    print(f"\npocket rows in the scored set: {j.height:,}   "
          f"patching {int(touch.sum()):,} at gap_sched >= {args.min_gap:,.0f}")

    var = _variance_by_bin()
    n_all = pl.scan_parquet(str(RAW_DIR / "ranking.parquet")).filter(
        pl.col("PHASE_mvt") == "DEP").select(pl.len()).collect().item()
    d_cur = d_new = 0.0
    print(f"\n{'bin':>24s} {'n':>4s} {'now':>10s} {'patched':>10s} "
          f"{'E[SSE] now':>12s} {'E[SSE] new':>12s}")
    for (lo, hi), (p, m, ey, _n) in table.items():
        sel = touch & (gs >= lo) & (gs < hi)
        if not sel.any():
            continue
        v = var[(lo, hi)]
        a = float(np.sum(v + (ey - cur[sel]) ** 2))
        b = float(np.sum(v + (ey - new[sel]) ** 2))
        d_cur += a
        d_new += b
        print(f"  [{lo:>8,.0f},{hi:>10,.0f}) {int(sel.sum()):4d} "
              f"{cur[sel].mean():10,.0f} {new[sel].mean():10,.0f} {a:12.3e} {b:12.3e}")
    print(f"\npatched-rows expected SSE  {d_cur:.4e} -> {d_new:.4e}")
    print("NOTE: E[y] and Var come from 2025; this assumes the per-bin")
    print("distribution transfers to 2026. 22 rows carry most of the movement.")

    if args.patch:
        if not args.out:
            raise SystemExit("--patch needs --out")
        patched = dict(zip(j["MVT_ID_mvt"].to_list(),
                           np.where(touch, new, cur).tolist()))
        ids = sub_df["MVT_ID_mvt"].to_list()
        vals = sub_df[tcol].to_numpy().astype(float).copy()
        changed = 0
        for i, mid in enumerate(ids):
            if mid in patched and patched[mid] != vals[i]:
                vals[i] = patched[mid]
                changed += 1
        out = sub_df.with_columns(pl.Series(tcol, np.clip(vals, 0.0, None)))
        assert out.height == sub_df.height, "row count changed"
        out.write_parquet(args.out)
        print(f"\nwrote {args.out}: {out.height:,} rows, {changed:,} changed "
              f"({changed / out.height * 100:.3f}%)")


if __name__ == "__main__":
    main()
