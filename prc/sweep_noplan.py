"""Sweep the no-flight-plan specialist without fitting a global model at all.

`rmse_noplan` is computed only on rows where `has_flight_plan == 0`, and every
one of those predictions comes from the specialist — the global model's output
for them is discarded by `np.where(plan, bagged, npred)`. So the global fit,
which is 2,500 iterations on 2.06M rows and the expensive half of a fold, has no
bearing on this metric and can be skipped. `prc.tune_noplan` still fits it once
per fold to score combined RMSE; this trades that for dozens of configurations
in minutes on a small lease.

WHY THIS SWEEP EXISTS. A residual autopsy on real out-of-fold predictions found
the no-plan group's error is **92% instability and 7% bias** — 92% of squared
error sits where the three seeds disagree with each other, only 7% where they
agree and are jointly wrong. `log-corr(seed variance, squared error)` is
0.37-0.44 in this group across all three folds, two to three times the bulk's.
Variance, not missing signal, is what is costing us here.

And nothing has ever been aimed at it. The 2026-09-06 grid varied depth,
iterations and feature set — twelve configurations, no winner — and never
touched regularisation. It also ran *before* airport routing existed, so it
tuned a single model over 22,470 rows. Today LIRF has its own specialist fit on
about 1,200, with `min_data_in_leaf` at CatBoost's default of 1, which permits
single-row leaves.

MULTIPLE COMPARISONS. With N configurations over 3 folds, about N/8 win all
three by chance. A 3/3 record is necessary and not sufficient. Keep the grid
small, and require a margin that is large against the spread between folds.
A 2024 team lost 2,066 -> 2,276 selecting a per-subgroup model this way.

    python -m prc.sweep_noplan --folds 1,3,5
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from .crossval import FOLDS, NOPLAN_SETTINGS, fit_noplan
from .features import CATEGORICAL, FEATURES, TARGET
from .final import DROP
from .train import load_training_features
from .validate import rmse

# v8's shipped feature set, so the control here is the model actually running.
EXTRA_DROP = [
    "prev_stand_gap", "prev_stand_headway", "prev_rwy_gap", "mvt_sec_00",
    "callsign_op", "arr_sub_day", "arr_sub_stand", "lobt_minus_aobt",
]


def settings(**over) -> SimpleNamespace:
    """An args-like object carrying every setting fit_noplan reads."""
    base = dict(
        noplan_iterations=600, noplan_depth=6, threads=6,
        noplan_routes="LIRF", noplan_split_lirf=False, noplan_target="raw",
        noplan_train="group", noplan_weight=20.0, sched_blend=False,
        noplan_l2=None, noplan_min_data=0, noplan_rsm=None, noplan_seeds=1,
        noplan_grow_policy="SymmetricTree",
    )
    base.update(over)
    missing = [k for k in NOPLAN_SETTINGS if k not in base]
    if missing:
        raise SystemExit(f"settings() is missing {missing}; add them here too")
    return SimpleNamespace(**base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="1,3,5")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="results/sweep_noplan.json")
    args = ap.parse_args()

    frame = load_training_features()
    feats = [f for f in FEATURES if f not in DROP and f not in EXTRA_DROP]
    cats = [c for c in CATEGORICAL if c in feats]
    chosen = [FOLDS[int(i) - 1] for i in args.folds.split(",")]

    # Deliberately small: four one-at-a-time variance knobs against the control,
    # plus one combination. Five arms over three folds is well short of the
    # N/8 spurious-winner rate that would make a 3/3 record meaningless.
    GRID = {
        "ctrl":        {},
        "l2_10":       {"noplan_l2": 10.0},
        "l2_30":       {"noplan_l2": 30.0},
        # min_data_in_leaf needs a per-node policy; on symmetric trees CatBoost
        # ignores it silently, which the first smoke test caught.
        "depthwise":   {"noplan_grow_policy": "Depthwise"},
        "dw_leaf20":   {"noplan_grow_policy": "Depthwise", "noplan_min_data": 20},
        "dw_leaf50":   {"noplan_grow_policy": "Depthwise", "noplan_min_data": 50},
        "depth4":      {"noplan_depth": 4},
        "bag3":        {"noplan_seeds": 3},
        "combo":       {"noplan_grow_policy": "Depthwise", "noplan_min_data": 20,
                        "noplan_depth": 4, "noplan_seeds": 3},
    }
    print(f"{len(chosen)} folds x {len(GRID)} configurations "
          f"(~{len(GRID)/8:.1f} spurious 3/3 records expected by chance)")
    print("scoring rmse on no-plan rows only; no global model is fitted\n")

    results: dict[str, list[float]] = {}
    for held in chosen:
        month = pl.col("month")
        train = frame.filter(~month.is_in(held))
        test = frame.filter(month.is_in(held))
        te = test.filter(pl.col("has_flight_plan") == 0)
        y = te[TARGET].to_numpy().astype(float)
        print(f"fold {held}: {te.height:,} no-plan test rows")

        for name, over in GRID.items():
            t0 = time.time()
            model, nfeats = fit_noplan(
                train, feats, cats, settings(threads=args.threads, **over))
            pred = model.predict(te.select(nfeats).to_pandas())
            r = rmse(y, pred)
            results.setdefault(name, []).append(r)
            print(f"    {name:12s} {r:9.1f}   ({time.time()-t0:.0f}s)")
        print()

    base = np.array(results["ctrl"])
    print(f"{'config':<14}{'mean':>10}{'vs ctrl':>10}   per-fold delta{'':>10}{'wins':>6}")
    rows = []
    for name, scores in results.items():
        d = np.array(scores) - base
        rows.append((name, float(np.mean(scores)), d, int((d < 0).sum())))
    for name, mean, d, wins in sorted(rows, key=lambda r: r[1]):
        per = "  ".join(f"{x:+8.1f}" for x in d)
        print(f"{name:<14}{mean:10.1f}{np.mean(d):+10.1f}   {per}   {wins}/{len(d)}")

    print("\nA win here is worth roughly a third as much on the board: the group is")
    print("1.6% of rows, and -100s of no-plan RMSE is about -5s overall.")
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {k: v for k, v in results.items()}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
