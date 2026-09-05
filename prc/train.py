"""CatBoost training, with ablation support.

Split discipline (see prc/validate.py for why Jan + Jul):

    train           months 2,3,4,5,6,8,9,10,12
    inner val       month 11        -- early stopping only
    holdout         months 1 and 7  -- reported, touches no decision

**Selection is on trimmed RMSE**, not plain RMSE. See validate.trimmed_rmse:
untrimmed RMSE on this target has a 159s-wide 95% interval and got the v1-vs-v2
comparison backwards, while the trimmed version matched the leaderboard.

Label winsorising (``--winsor``) caps the TRAINING labels only. 584 of 2.08M
rows sit above 7,200s, and a squared loss gives each of them on the order of a
thousand times the gradient weight of an ordinary row -- so a rounding error's
worth of data steers a large share of the fit. Capping trades our ability to
predict extremes we cannot predict anyway for a better fit on the 99.97% that we
can. The holdout and the reported metric are never capped.

    python -m prc.train --tag base
    python -m prc.train --tag winsor20k --winsor 20000
    python -m prc.train --tag v1feats --drop airport_plan
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .config import INTERIM_DIR, RAW_DIR, ensure_dirs
from .features import CATEGORICAL, FEATURES, TARGET, build
from .validate import rmse, trimmed_rmse

TRAIN_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 12)
INNER_VAL_MONTH = 11
HOLDOUT_MONTHS = (1, 7)
RESULTS = Path("results/ablation.jsonl")


def load_training_features(cache: bool = True) -> pl.DataFrame:
    """Build features month by month, so peak memory is one month not twelve."""
    cached = INTERIM_DIR / "train_features.parquet"
    if cache and cached.exists():
        return pl.read_parquet(cached)
    ensure_dirs()
    parts = []
    for path in sorted(RAW_DIR.glob("training_*.parquet")):
        part = build(pl.read_parquet(path))
        parts.append(part)
        print(f"  {path.name}: {part.height:,} dep rows")
    frame = pl.concat(parts)
    if cache:
        frame.write_parquet(cached)
    return frame


def report(y: np.ndarray, pred: np.ndarray, has_plan: np.ndarray, label: str) -> dict:
    print(f"\n== {label} ==")
    out = {
        "rmse": rmse(y, pred),
        "trimmed_rmse": trimmed_rmse(y, pred, 100),
        "n": int(len(y)),
    }
    print(f"  n = {len(y):,}")
    print(f"  RMSE          = {out['rmse']:.4f}s   (reported)")
    print(f"  trimmed RMSE  = {out['trimmed_rmse']:.4f}s   (selection metric)")
    for name, mask in (("flight plan present", has_plan == 1), ("flight plan MISSING", has_plan == 0)):
        if mask.any():
            key = "bulk" if name.endswith("present") else "noplan"
            out[f"rmse_{key}"] = rmse(y[mask], pred[mask])
            share = ((y[mask] - pred[mask]) ** 2).sum() / ((y - pred) ** 2).sum()
            print(
                f"  {name:<21} n={mask.sum():>8,}  RMSE={out[f'rmse_{key}']:8.1f}s"
                f"   {100 * share:5.1f}% of squared error"
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="names the model file and the results row")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=6, help="box is 8C/16T and shared")
    parser.add_argument("--winsor", type=float, default=0.0, help="cap TRAINING labels, seconds")
    parser.add_argument("--drop", default="", help="comma-separated features to exclude")
    parser.add_argument("--od-wait", type=int, default=200)
    args = parser.parse_args()

    from catboost import CatBoostRegressor, Pool

    dropped = [c.strip() for c in args.drop.split(",") if c.strip()]
    feats = [f for f in FEATURES if f not in dropped]
    cats = [c for c in CATEGORICAL if c not in dropped]
    cat_idx = [feats.index(c) for c in cats]

    frame = load_training_features()
    month = pl.col("month")
    splits = {
        "train": frame.filter(month.is_in(TRAIN_MONTHS)),
        "inner": frame.filter(month == INNER_VAL_MONTH),
        "holdout": frame.filter(month.is_in(HOLDOUT_MONTHS)),
    }
    print(f"\n[{args.tag}] {len(feats)} features"
          + (f", dropped {dropped}" if dropped else "")
          + (f", winsor at {args.winsor:,.0f}s" if args.winsor else ""))

    def labels(name: str) -> np.ndarray:
        y = splits[name][TARGET].to_numpy().astype(float)
        # Never cap what we measure on.
        if args.winsor and name in ("train", "inner"):
            capped = int((y > args.winsor).sum())
            if name == "train":
                print(f"  capped {capped:,} training labels ({100 * capped / len(y):.4f}%)")
            return np.minimum(y, args.winsor)
        return y

    pools = {
        name: Pool(part.select(feats).to_pandas(), labels(name), cat_features=cat_idx)
        for name, part in splits.items()
    }

    model = CatBoostRegressor(
        iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
        l2_leaf_reg=args.l2, loss_function="RMSE", eval_metric="RMSE",
        thread_count=args.threads, random_seed=1113,
        od_type="Iter", od_wait=args.od_wait, verbose=500,
    )
    started = time.time()
    model.fit(pools["train"], eval_set=pools["inner"], use_best_model=True)
    elapsed = time.time() - started
    print(f"\nfitted in {elapsed:.0f}s, best iteration {model.get_best_iteration()}")

    part = splits["holdout"]
    result = report(
        part[TARGET].to_numpy().astype(float),
        model.predict(pools["holdout"]),
        part["has_flight_plan"].to_numpy(),
        f"holdout (months 1 and 7) [{args.tag}]",
    )

    Path("models").mkdir(exist_ok=True)
    model.save_model(f"models/{args.tag}.cbm")
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "tag": args.tag, "features": len(feats), "dropped": dropped,
            "winsor": args.winsor, "depth": args.depth, "lr": args.lr, "l2": args.l2,
            "best_iteration": model.get_best_iteration(), "fit_seconds": round(elapsed),
            **result,
        }) + "\n")
    print(f"saved models/{args.tag}.cbm and appended to {RESULTS}")


if __name__ == "__main__":
    main()
