"""Predict, per row, whether the off-block timestamp is a schedule substitution.

This is the largest remaining lever and it is a CLASSIFICATION problem, not a
regression one. See `notes/2026-09-10-the-lirf-pocket.md`.

Inside the LIRF no-flight-plan pocket -- 348 of 344,841 scored rows -- the target
is a mixture of three processes:

    S  substituted: the airport wrote SCHED into BLOCK, so y IS gap_sched,
       exactly. 722 of 1,488 rows in 2025.
    R  day rollover: BLOCK dated 24h early. 12 rows, mean 87,505, sd 379 --
       and separated from everything else by a 55,900-second gap in the sorted
       values, so the class is not a judgement call.
    N  ordinary: 754 rows, mean 1,557.

Under squared loss the prediction is the posterior mean,

    yhat = p * gap_sched + (1 - p) * m

where m is the non-substituted mean for the row's gap_sched band -- which is
~1,500 at low gaps and ~87,400 above 28,800s, because R takes over there. So
band-level m is enough, and the whole question is how sharply p can be resolved.

Measured: with p at the BAND rate the pocket holds 9.31e9 of expected squared
error, 32% of the whole board. With p known per row it holds 1.6e7. That gap is
worth roughly 51 RMSE seconds -- 288.9 down to 237.8.

The key design choice: train on ALL 2,085,047 departures, not on the 1,488
pocket rows. The substitution mechanism is the same everywhere, and the wider
set carries 199,634 positives against the pocket's 722. A model that learns the
identity from with-plan rows can then apply it where the flight plan is missing,
which is exactly where it pays.

    python -m prc.substitution --fit --folds 1,3,5
    python -m prc.substitution --fit --save models/subclf.cbm
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import INTERIM_DIR, ensure_dirs
from .features import CATEGORICAL, TARGET

# Serve-available only. Every one of these is present in ranking.parquet;
# nothing here reads BLOCK_TIME_UTC_mvt, which is withheld there.
FEATURES = [
    "gap_sched", "ADEP_mvt", "has_flight_plan", "callsign_op",
    "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
    "hour", "minute_of_day", "dow", "month", "doy", "is_weekend",
    "dep_30min", "arr_30min",
]
TOL = 60.0            # a substitution matches SCHED to within a minute
ROLLOVER_MIN = 40000  # the 55,900s gap in the sorted values sits well below this


def label(frame: pl.DataFrame) -> np.ndarray:
    y = frame[TARGET].to_numpy().astype(float)
    gs = np.nan_to_num(frame["gap_sched"].to_numpy(dtype=float), nan=1e18)
    return (np.abs(y - gs) <= TOL).astype(int)


def band_table(frame: pl.DataFrame, bins) -> dict:
    """Per-band mean of the NON-substituted rows, inside the pocket."""
    sub = frame.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("has_flight_plan") == 0))
    y = sub[TARGET].to_numpy().astype(float)
    gs = sub["gap_sched"].to_numpy().astype(float)
    s = np.abs(y - gs) <= TOL
    out = {}
    for lo, hi in bins:
        m = (gs >= lo) & (gs < hi)
        rest = y[m & ~s]
        if m.sum() >= 5:
            out[(lo, hi)] = float(rest.mean()) if len(rest) else float(y[m].mean())
    return out


def fit(train: pl.DataFrame, iterations: int = 600, depth: int = 6, threads: int = 5):
    from catboost import CatBoostClassifier, Pool

    feats = [f for f in FEATURES if f in train.columns]
    cats = [feats.index(c) for c in feats if c in CATEGORICAL]
    y = label(train)
    x = train.select(feats).to_pandas()
    clf = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=0.08,
        thread_count=threads, random_seed=1113, verbose=200,
        loss_function="Logloss", eval_metric="AUC",
    )
    clf.fit(Pool(x, y, cat_features=cats))
    return clf, feats


def pocket_sse(frame: pl.DataFrame, p: np.ndarray, bands: dict) -> tuple[float, int]:
    """Actual squared error on pocket rows under yhat = p*gs + (1-p)*m."""
    gs = frame["gap_sched"].to_numpy().astype(float)
    y = frame[TARGET].to_numpy().astype(float)
    m = np.full(len(gs), np.nan)
    for (lo, hi), val in bands.items():
        m[(gs >= lo) & (gs < hi)] = val
    ok = ~np.isnan(m)
    yhat = p[ok] * gs[ok] + (1.0 - p[ok]) * m[ok]
    return float(np.sum((y[ok] - yhat) ** 2)), int(ok.sum())


def main() -> None:
    import argparse

    from .pocket import BINS
    from .train import load_training_features

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--folds", default="1,3,5",
                    help="month-parity folds to hold out, as in prc.crossval")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=7200.0)
    ap.add_argument("--save", help="write the classifier here once fitted on all of 2025")
    args = ap.parse_args()

    ensure_dirs()
    frame = load_training_features()
    print(f"loaded {frame.height:,} training rows")

    pocket = (pl.col("ADEP_mvt") == "LIRF") & (pl.col("has_flight_plan") == 0)
    folds = [int(f) for f in args.folds.split(",") if f.strip()]

    rows = []
    for f in folds:
        test_months = (f, f + 6)
        te = frame.filter(pl.col("month").is_in(test_months))
        tr = frame.filter(~pl.col("month").is_in(test_months))
        clf, feats = fit(tr, args.iterations, args.depth, args.threads)
        bands = band_table(tr, BINS)

        te_pocket = te.filter(pocket & (pl.col("gap_sched") >= args.min_gap))
        if te_pocket.height == 0:
            continue
        p = clf.predict_proba(te_pocket.select(feats).to_pandas())[:, 1]

        # the same rows under the BAND rate, which is what prc.pocket ships
        band_p = np.zeros(len(p))
        gs = te_pocket["gap_sched"].to_numpy().astype(float)
        trp = tr.filter(pocket)
        ty, tg = trp[TARGET].to_numpy().astype(float), trp["gap_sched"].to_numpy().astype(float)
        ts = np.abs(ty - tg) <= TOL
        for lo, hi in BINS:
            m25 = (tg >= lo) & (tg < hi)
            if m25.sum() >= 5:
                band_p[(gs >= lo) & (gs < hi)] = ts[m25].mean()

        sse_clf, n = pocket_sse(te_pocket, p, bands)
        sse_band, _ = pocket_sse(te_pocket, band_p, bands)
        y = te_pocket[TARGET].to_numpy().astype(float)
        oracle_p = (np.abs(y - gs) <= TOL).astype(float)
        sse_orc, _ = pocket_sse(te_pocket, oracle_p, bands)

        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        truth = (np.abs(y - gs) <= TOL).astype(int)
        auc = roc_auc_score(truth, p) if 0 < truth.sum() < len(truth) else float("nan")

        captured = (sse_band - sse_clf) / max(sse_band - sse_orc, 1e-9)
        rows.append((test_months, n, auc, sse_band, sse_clf, sse_orc, captured))
        print(f"\nfold {test_months}: {n} pocket rows, AUC {auc:.4f}")
        print(f"  band rate  SSE {sse_band:.4e}")
        print(f"  classifier SSE {sse_clf:.4e}   captured {captured*100:5.1f}% of the gap")
        print(f"  oracle     SSE {sse_orc:.4e}")

    if rows:
        print(f"\n{'fold':>10s} {'n':>5s} {'AUC':>7s} {'band SSE':>11s} "
              f"{'clf SSE':>11s} {'captured':>9s}")
        for tm, n, auc, sb, sc, so, cap in rows:
            print(f"  {str(tm):>8s} {n:5d} {auc:7.4f} {sb:11.3e} {sc:11.3e} {cap*100:8.1f}%")
        mean_cap = float(np.mean([r[-1] for r in rows]))
        wins = sum(1 for r in rows if r[4] < r[3])
        print(f"\nmean captured {mean_cap*100:.1f}%   beats the band rate on {wins}/{len(rows)} folds")
        print("Board translation, from notes/2026-09-10-the-lirf-pocket.md:")
        print("  0% -> 288.9   50% -> 264.6   75% -> 251.5   100% -> 237.8")

    if args.save:
        clf, feats = fit(frame, args.iterations, args.depth, args.threads)
        path = args.save if str(args.save).startswith(("/", "C:")) else INTERIM_DIR / args.save
        clf.save_model(str(path))
        print(f"\nsaved classifier fitted on all of 2025 to {path}")


if __name__ == "__main__":
    main()
