"""Tune the no-flight-plan model against cached global predictions.

The split model won 3/3 folds but its second half was never tuned: depth 6,
lr 0.05, 600 iterations, 18 features, all picked without a single measurement.
Its group still carries most of the remaining error (RMSE 1,338-2,154 against a
bulk of ~210-280), so this is where the headroom is.

The global half is expensive (2,500 iterations on 1.7M rows) and the no-plan
half is cheap (22k rows), so the global model is fitted ONCE per fold and its
predictions cached, then every no-plan variant is scored against them. Dozens of
configurations for the price of three fits.

**Multiple comparisons are the hazard here.** A 2024 team lost 2,066 -> 2,276
picking among per-subgroup models by validation score. With N configurations and
3 folds, roughly N/8 will win all three folds by chance alone, so a 3/3 record is
necessary and not sufficient. The grid is kept deliberately small and the
selected configuration has to win by a margin that is large next to the spread
between folds, not merely positive.

    python -m prc.tune_noplan --folds 1,3,5 --iterations 2500
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .crossval import FOLDS, NOPLAN_DROP
from .features import CATEGORICAL, FEATURES, TARGET
from .final import DROP
from .train import load_training_features
from .validate import rmse

WAVE2 = ["dep_rwy_30min", "dep_rwy_headway", "arr_taxi_60min", "sched_demand_30min"]


def global_predictions(train, test, feats, cat_idx, args):
    from catboost import CatBoostRegressor, Pool

    model = CatBoostRegressor(
        iterations=args.iterations, depth=8, learning_rate=0.08, loss_function="RMSE",
        thread_count=args.threads, random_seed=1113, verbose=False,
    )
    model.fit(Pool(train.select(feats).to_pandas(),
                   train[TARGET].to_numpy().astype(float), cat_features=cat_idx))
    return model.predict(test.select(feats).to_pandas())


def noplan_predictions(train, test, nfeats, ncats, depth, iters, lr, threads):
    from catboost import CatBoostRegressor, Pool

    sub = train.filter(pl.col("has_flight_plan") == 0)
    model = CatBoostRegressor(
        iterations=iters, depth=depth, learning_rate=lr, loss_function="RMSE",
        thread_count=threads, random_seed=1113, verbose=False,
    )
    model.fit(Pool(sub.select(nfeats).to_pandas(),
                   sub[TARGET].to_numpy().astype(float), cat_features=ncats))
    return model.predict(test.select(nfeats).to_pandas())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="1,3,5")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    from . import reference  # noqa: F401  (attach lives there)

    chosen = [FOLDS[int(i) - 1] for i in args.folds.split(",")]
    frame = load_training_features()
    base_feats = [f for f in FEATURES if f not in DROP]
    # wave2 is dropped from the global model but is movement-side, so it exists
    # for no-flight-plan rows -- the one place feature scarcity actually binds.
    plus_feats = [f for f in FEATURES if f not in DROP or f in WAVE2]

    grid = list(itertools.product(["base", "wave2"], [4, 6, 8], [600, 2000]))
    print(f"{len(chosen)} folds x {len(grid)} no-plan configurations "
          f"(expect ~{len(grid) / 8:.1f} spurious 3/3 records by chance)\n")

    results: dict[tuple, list[float]] = {}
    baseline: list[float] = []
    for held in chosen:
        month = pl.col("month")
        train = frame.filter(~month.is_in(held))
        test = frame.filter(month.is_in(held))
        y = test[TARGET].to_numpy().astype(float)
        plan = test["has_flight_plan"].to_numpy() == 1

        started = time.time()
        cat_idx = [base_feats.index(c) for c in CATEGORICAL if c in base_feats]
        gpred = global_predictions(train, test, base_feats, cat_idx, args)
        print(f"fold {held}: global fitted in {time.time() - started:.0f}s, "
              f"bulk {rmse(y[plan], gpred[plan]):.2f}")

        for featset, depth, iters in grid:
            feats = base_feats if featset == "base" else plus_feats
            nfeats = [f for f in feats if f not in NOPLAN_DROP]
            ncats = [nfeats.index(c) for c in CATEGORICAL if c in nfeats]
            npred = noplan_predictions(train, test, nfeats, ncats, depth, iters, 0.05, args.threads)
            combined = np.where(plan, gpred, npred)
            key = (featset, depth, iters)
            results.setdefault(key, []).append(rmse(y, combined))
            if key == ("base", 6, 600):
                baseline.append(rmse(y, combined))
        print(f"  swept {len(grid)} configurations in {time.time() - started:.0f}s total")

    print(f"\n{'config':<22} {'mean':>9} {'vs shipped':>11} {'per-fold':>28} {'wins':>6}")
    base = np.array(baseline)
    rows = []
    for key, scores in results.items():
        d = np.array(scores) - base
        rows.append((key, float(np.mean(scores)), d, int((d < 0).sum())))
    rows.sort(key=lambda r: r[1])
    for key, mean, d, wins in rows:
        name = f"{key[0]}/d{key[1]}/i{key[2]}"
        marker = "  <- shipped in v5" if key == ("base", 6, 600) else ""
        print(f"{name:<22} {mean:>9.2f} {mean - base.mean():>+11.2f}   "
              f"{'  '.join(f'{x:+7.2f}' for x in d):>28} {wins:>4}/{len(base)}{marker}")

    Path("results").mkdir(exist_ok=True)
    Path("results/tune_noplan.json").write_text(json.dumps(
        {f"{k[0]}/d{k[1]}/i{k[2]}": v for k, v in results.items()}, indent=1), encoding="utf-8")
    print("\nwrote results/tune_noplan.json")


if __name__ == "__main__":
    main()
