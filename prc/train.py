"""CatBoost baseline.

Split discipline (see prc/validate.py for why Jan + Jul):

    train           months 2,3,4,5,6,8,9,10,12
    inner val       month 11        -- early stopping only
    holdout         months 1 and 7  -- reported, touches no decision

Early stopping on the holdout would quietly fit it, so it gets its own month.
The holdout is scored once, at the end, and nothing is tuned against it.

    python -m prc.train                    # fit, score, save model
    python -m prc.train --iterations 500   # quicker pass
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from .config import INTERIM_DIR, RAW_DIR, ensure_dirs
from .features import CATEGORICAL, FEATURES, TARGET, build
from .validate import rmse

TRAIN_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 12)
INNER_VAL_MONTH = 11
HOLDOUT_MONTHS = (1, 7)
MODEL_PATH = Path("models/catboost_baseline.cbm")


def load_training_features(cache: bool = True) -> pl.DataFrame:
    """Build features month by month, so peak memory is one month not twelve."""
    cached = INTERIM_DIR / "train_features.parquet"
    if cache and cached.exists():
        print(f"loading cached features from {cached}")
        return pl.read_parquet(cached)

    ensure_dirs()
    parts = []
    for path in sorted(RAW_DIR.glob("training_*.parquet")):
        started = time.time()
        part = build(pl.read_parquet(path))
        parts.append(part)
        print(f"  {path.name}: {part.height:,} dep rows in {time.time() - started:.1f}s")
    frame = pl.concat(parts)
    if cache:
        frame.write_parquet(cached)
        print(f"cached {frame.height:,} rows to {cached}")
    return frame


def report(y: np.ndarray, pred: np.ndarray, has_plan: np.ndarray, label: str) -> None:
    """RMSE overall, then split by the thing that actually drives it."""
    print(f"\n== {label} ==")
    print(f"  n = {len(y):,}    RMSE = {rmse(y, pred):.4f}s")
    for name, mask in (("flight plan present", has_plan == 1), ("flight plan MISSING", has_plan == 0)):
        if mask.any():
            share = ((y[mask] - pred[mask]) ** 2).sum() / ((y - pred) ** 2).sum()
            print(
                f"  {name:<21} n={mask.sum():>8,} ({100 * mask.mean():5.2f}%)  "
                f"RMSE={rmse(y[mask], pred[mask]):8.1f}s   {100 * share:5.1f}% of total squared error"
            )
    se = (y - pred) ** 2
    order = np.argsort(-se)
    for k in (10, 100, 1000):
        print(f"  worst {k:>5,} rows carry {100 * se[order[:k]].sum() / se.sum():5.1f}% of squared error")
    print(f"  RMSE excluding the worst 100 rows: {np.sqrt(se[order[100:]].mean()):.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=6, help="box is 8C/16T and shared")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    from catboost import CatBoostRegressor, Pool

    frame = load_training_features(cache=not args.no_cache)
    print(f"\n{frame.height:,} departure rows, {len(FEATURES)} features")

    month = pl.col("month")
    splits = {
        "train": frame.filter(month.is_in(TRAIN_MONTHS)),
        "inner": frame.filter(month == INNER_VAL_MONTH),
        "holdout": frame.filter(month.is_in(HOLDOUT_MONTHS)),
    }
    for name, part in splits.items():
        print(f"  {name:<8} {part.height:>9,} rows")

    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    pools = {
        name: Pool(
            part.select(FEATURES).to_pandas(),
            part[TARGET].to_numpy().astype(float),
            cat_features=cat_idx,
        )
        for name, part in splits.items()
    }

    model = CatBoostRegressor(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        loss_function="RMSE",
        eval_metric="RMSE",
        thread_count=args.threads,
        random_seed=1113,
        od_type="Iter",
        od_wait=200,
        verbose=250,
    )
    started = time.time()
    model.fit(pools["train"], eval_set=pools["inner"], use_best_model=True)
    print(f"\nfitted in {time.time() - started:.1f}s, best iteration {model.get_best_iteration()}")

    for name in ("inner", "holdout"):
        part = splits[name]
        report(
            part[TARGET].to_numpy().astype(float),
            model.predict(pools[name]),
            part["has_flight_plan"].to_numpy(),
            f"{name} ({'month 11' if name == 'inner' else 'months 1 and 7'})",
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"\nsaved {MODEL_PATH}")

    imp = sorted(zip(FEATURES, model.get_feature_importance()), key=lambda kv: -kv[1])
    print("\n-- feature importance, top 15 --")
    for name, score in imp[:15]:
        print(f"  {score:7.2f}  {name}")


if __name__ == "__main__":
    main()
