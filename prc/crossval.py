"""Multi-fold validation across month-pairs.

Why this exists: three submissions in a row improved on the Jan+Jul 2025
holdout and got worse on the 2026 leaderboard. Month 11 agreed with the
holdout, so it is not a bad fold -- but one fold cannot tell a real improvement
from a fold-specific one, and this metric is decided by a handful of rows that
differ between any two periods. A change worth shipping should win on most
folds, not on one.

Six folds, each holding out one winter month and one summer month so every fold
has the seasonal mix of the ranking set (January and July 2026):

    fold 0: 1, 7      fold 2: 3, 9      fold 4: 5, 11
    fold 1: 2, 8      fold 3: 4, 10     fold 5: 6, 12

Iterations are FIXED rather than early-stopped. Early stopping needs a
validation month, which would either eat a fold or vary between folds, and the
stopping point itself becomes a source of fold-to-fold variance that has nothing
to do with the change being tested.

    python -m prc.crossval --tag base --drop airport_plan,dep_rwy_30min,...
    python -m prc.crossval --tag wave2 --drop airport_plan
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .features import CATEGORICAL, FEATURES, TARGET
from .train import load_training_features
from .validate import rmse

FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
RESULTS = Path("results/crossval.jsonl")


def run_fold(frame, held: tuple[int, int], feats, cat_idx, args) -> dict:
    from catboost import CatBoostRegressor, Pool

    month = pl.col("month")
    train = frame.filter(~month.is_in(held))
    test = frame.filter(month.is_in(held))

    y_train = train[TARGET].to_numpy().astype(float)
    if args.winsor:
        y_train = np.minimum(y_train, args.winsor)

    model = CatBoostRegressor(
        iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
        l2_leaf_reg=args.l2, loss_function="RMSE", thread_count=args.threads,
        random_seed=1113, verbose=False,
    )
    started = time.time()
    model.fit(Pool(train.select(feats).to_pandas(), y_train, cat_features=cat_idx))

    y = test[TARGET].to_numpy().astype(float)
    pred = model.predict(test.select(feats).to_pandas())
    plan = test["has_flight_plan"].to_numpy() == 1
    out = {
        "fold": list(held),
        "n": int(len(y)),
        "rmse": rmse(y, pred),
        "rmse_bulk": rmse(y[plan], pred[plan]),
        "rmse_noplan": rmse(y[~plan], pred[~plan]) if (~plan).any() else None,
        "seconds": round(time.time() - started),
    }
    print(
        f"  fold {held}: RMSE {out['rmse']:8.2f}  bulk {out['rmse_bulk']:7.2f}  "
        f"no-plan {out['rmse_noplan']:8.0f}  ({out['seconds']}s)"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--winsor", type=float, default=0.0)
    parser.add_argument("--drop", default="")
    args = parser.parse_args()

    dropped = [c.strip() for c in args.drop.split(",") if c.strip()]
    feats = [f for f in FEATURES if f not in dropped]
    cat_idx = [feats.index(c) for c in CATEGORICAL if c not in dropped]

    frame = load_training_features()
    print(f"[{args.tag}] {len(feats)} features, {len(FOLDS)} folds, "
          f"{args.iterations} iterations" + (f", winsor {args.winsor:,.0f}s" if args.winsor else ""))

    folds = [run_fold(frame, held, feats, cat_idx, args) for held in FOLDS]
    scores = np.array([f["rmse"] for f in folds])
    bulk = np.array([f["rmse_bulk"] for f in folds])
    print(f"\n  mean RMSE {scores.mean():8.2f}  sd {scores.std(ddof=1):6.2f}  "
          f"range {scores.min():.1f}..{scores.max():.1f}")
    print(f"  mean bulk {bulk.mean():8.2f}  sd {bulk.std(ddof=1):6.2f}")

    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "tag": args.tag, "dropped": dropped, "winsor": args.winsor,
            "iterations": args.iterations, "depth": args.depth, "lr": args.lr,
            "mean_rmse": float(scores.mean()), "sd_rmse": float(scores.std(ddof=1)),
            "mean_bulk": float(bulk.mean()), "folds": folds,
        }) + "\n")
    print(f"appended to {RESULTS}")


if __name__ == "__main__":
    main()
