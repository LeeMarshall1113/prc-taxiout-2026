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


# Features that actually exist for rows with no flight plan. Everything derived
# from a _flt column is null there, so a global model is asking a specialist
# question of a model whose inputs are 98.5% absent for this group.
NOPLAN_DROP = [
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
    "gap_aobt", "gap_eobt", "gap_lobt", "dep_delay", "sched_vs_eobt",
    "has_flight_plan",  # constant within the group
]


DAY = 86400.0


def drop_day_offsets(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove training rows whose label is a confirmed one-day timestamp error.

    Identified in notes/2026-09-05-day-offset-check.md: about 15 rows a year
    where removing a whole number of days leaves a plausible taxi time (median
    951s against a global median of 912s). Every one is at LIRF or LSZH.

    This is NOT winsorising, which was measured at 78s worse. Winsorising
    falsifies the label on genuinely long taxis, so the model learns to
    underpredict real ones. This deletes rows we have proven are corrupt, and
    only those -- under squared loss each carries on the order of a thousand
    times an ordinary row's gradient weight, so a handful of them steers a
    large share of the fit.

    Applied to TRAINING only. Never to anything we score on: the evaluation
    labels are corrupt in the same way and predicting them is the job.
    """
    y = pl.col(TARGET).cast(pl.Float64)
    residual = y - (y / DAY).round() * DAY
    corrupt = (y >= 20000) & (residual >= 0) & (residual < 7200)
    return frame.filter(~corrupt.fill_null(False))


def _catboost_kwargs(args) -> dict:
    """Loss, and the categorical-encoding knobs that have never been touched.

    ``--loss Huber:delta=2000`` keeps eval_metric on RMSE, which CatBoost allows
    and which costs us nothing since we fix iterations rather than early-stop.
    Huber is the continuous form of dropping outliers: residuals past delta get
    linear rather than quadratic weight, so extreme rows stop dominating split
    selection without their labels being falsified.

    ``one_hot_max_size`` defaults to 2, so today every categorical -- including
    FLIGHT_RULE_mvt and WK_TBL_CAT_flt, which have a handful of levels -- goes
    through target-statistic encoding and its prior shrinkage. For a
    low-cardinality feature that shrinkage is noise, not regularisation.
    """
    kwargs = dict(
        iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
        l2_leaf_reg=args.l2, thread_count=args.threads, random_seed=1113, verbose=False,
    )
    loss = getattr(args, "loss", "RMSE")
    kwargs["loss_function"] = loss
    if loss != "RMSE":
        kwargs["eval_metric"] = "RMSE"
    ohms = getattr(args, "one_hot_max_size", 0)
    if ohms:
        kwargs["one_hot_max_size"] = ohms
    return kwargs


def _fit_predict(train_frame, test_x, feats, cat_idx, y_train, args, seed):
    """Fit on the chosen target scale and return predictions on the SECONDS scale.

    ``--target log`` fits log1p(seconds). A right-skewed duration is the textbook
    case for it: on the log scale the bulk stops being distorted by a handful of
    multi-hour rows, and unlike winsorising the model can still emit large values,
    because the back-transform is exponential rather than clipped.

    The back-transform uses Duan's smearing estimator. exp(E[log y]) is the
    conditional *median*, not the mean, and RMSE wants the mean -- so naively
    exponentiating biases every prediction low. Smearing corrects it
    non-parametrically by the mean of exp(training residuals), which needs no
    assumption that those residuals are normal. Getting this wrong is the usual
    reason a log-target model scores worse than it should.
    """
    from catboost import CatBoostRegressor, Pool

    fit_y = np.log1p(np.maximum(y_train, 0.0)) if args.target == "log" else y_train
    kwargs = _catboost_kwargs(args)
    kwargs["random_seed"] = 1113 + seed * 977
    model = CatBoostRegressor(**kwargs)
    train_x = train_frame.select(feats).to_pandas()
    model.fit(Pool(train_x, fit_y, cat_features=cat_idx))

    if args.target != "log":
        return model.predict(test_x)

    resid = fit_y - model.predict(train_x)
    smearing = float(np.mean(np.exp(resid)))
    return np.expm1(model.predict(test_x)) * smearing


def fit_noplan(train, feats, cat_features, args):
    """A model for rows with no flight plan, on the columns they actually have.

    ``args.noplan_train`` picks what it learns from:

    ``group``   only the ~22k no-flight-plan rows. What v5 shipped.
    ``all``     every departure, with the flight-plan-derived columns removed.
                The feature vector is identical either way -- stand, runway,
                time, congestion, weather all exist for all 2.08M rows -- so
                restricting to 22k throws away two orders of magnitude of data
                about how those features map to taxi time, for no reason beyond
                how the split was first written.
    ``weighted`` same as ``all`` but with no-flight-plan rows upweighted, so the
                extra data informs the shape without drowning the population we
                actually predict.

    This is not per-subgroup model *selection* -- the thing that cost a 2024
    team 2,066 -> 2,276. Nothing is chosen by score; the group is defined by
    which columns exist, fixed before measurement, and both halves scored
    together on folds not used for training.
    """
    from catboost import CatBoostRegressor, Pool

    mode = getattr(args, "noplan_train", "group")
    nfeats = [f for f in feats if f not in NOPLAN_DROP]
    ncats = [nfeats.index(c) for c in cat_features if c in nfeats]

    if mode == "group":
        sub = train.filter(pl.col("has_flight_plan") == 0)
        weights = None
    else:
        sub = train
        if mode == "weighted":
            w = getattr(args, "noplan_weight", 20.0)
            weights = np.where(sub["has_flight_plan"].to_numpy() == 0, w, 1.0)
        else:
            weights = None

    y = sub[TARGET].to_numpy().astype(float)
    # The global --target flag never reached this model, so a log scale was
    # rejected for the bulk while going untested where it is most motivated:
    # this group's wild half has a standard deviation of 11,166s.
    log_scale = getattr(args, "noplan_target", "raw") == "log"
    fit_y = np.log1p(np.maximum(y, 0.0)) if log_scale else y

    x = sub.select(nfeats).to_pandas()
    model = CatBoostRegressor(
        iterations=args.noplan_iterations, depth=getattr(args, "noplan_depth", 6),
        learning_rate=0.05, loss_function="RMSE",
        thread_count=args.threads, random_seed=1113, verbose=False,
    )
    model.fit(Pool(x, fit_y, cat_features=ncats, weight=weights))
    if not log_scale:
        return model, nfeats

    smear = float(np.mean(np.exp(fit_y - model.predict(x))))
    raw_predict = model.predict

    class _Smeared:  # same interface, seconds scale out
        feature_names_ = getattr(model, "feature_names_", None)

        @staticmethod
        def predict(frame):
            return np.expm1(raw_predict(frame)) * smear

    return _Smeared, nfeats


def run_fold(frame, held: tuple[int, int], feats, cat_idx, args) -> dict:
    from catboost import CatBoostRegressor, Pool

    from . import reference

    month = pl.col("month")
    train = frame.filter(~month.is_in(held))
    test = frame.filter(month.is_in(held))
    train, test = reference.attach(train, test)
    if getattr(args, "drop_dayoffset", False):
        before = train.height
        train = drop_day_offsets(train)
        print(f"    dropped {before - train.height} confirmed day-offset training rows")

    y_train = train[TARGET].to_numpy().astype(float)
    if args.winsor:
        y_train = np.minimum(y_train, args.winsor)

    train_pool = Pool(train.select(feats).to_pandas(), y_train, cat_features=cat_idx)
    test_x = test.select(feats).to_pandas()
    y = test[TARGET].to_numpy().astype(float)
    plan = test["has_flight_plan"].to_numpy() == 1

    started = time.time()
    preds = [_fit_predict(train, test_x, feats, cat_idx, y_train, args, s)
             for s in range(args.seeds)]
    stack = np.vstack(preds)
    bagged = stack.mean(axis=0)

    if args.noplan_model:
        cat_names = [feats[i] for i in cat_idx]
        nmodel, nfeats = fit_noplan(train, feats, cat_names, args)
        npred = nmodel.predict(test.select(nfeats).to_pandas())
        split = np.where(plan, bagged, npred)
        print(f"    dedicated no-plan model: global {rmse(y[~plan], bagged[~plan]):8.1f} -> "
              f"{rmse(y[~plan], npred[~plan]):8.1f} on {int((~plan).sum()):,} rows | "
              f"overall {rmse(y, bagged):7.2f} -> {rmse(y, split):7.2f}")
        bagged_global = bagged
        bagged = split

    out = {
        "fold": list(held), "n": int(len(y)), "seeds": args.seeds,
        "rmse": rmse(y, preds[0]),
        "rmse_bagged": rmse(y, bagged),
        "rmse_bulk": rmse(y[plan], bagged[plan]),
        "rmse_noplan": rmse(y[~plan], bagged[~plan]) if (~plan).any() else None,
        "seconds": round(time.time() - started),
    }
    gain = out["rmse"] - out["rmse_bagged"]
    print(
        f"  fold {held}: seed0 {out['rmse']:8.2f}  bagged({args.seeds}) {out['rmse_bagged']:8.2f}  "
        f"gain {gain:+6.2f}  bulk {out['rmse_bulk']:7.2f}  ({out['seconds']}s)"
    )

    if args.save_preds:
        cols = {"y": y, "has_flight_plan": plan.astype(np.int8), "bagged": bagged}
        cols.update({f"seed{i}": p for i, p in enumerate(preds)})
        path = Path(f"results/preds_{args.tag}_{held[0]}_{held[1]}.parquet")
        path.parent.mkdir(exist_ok=True)
        pl.DataFrame(cols).write_parquet(path)
        print(f"    saved {path}")
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
    parser.add_argument("--loss", default="RMSE", help='e.g. "Huber:delta=2000"')
    parser.add_argument("--one-hot-max-size", type=int, default=0)
    parser.add_argument("--drop-dayoffset", action="store_true")
    parser.add_argument("--target", choices=["raw", "log"], default="raw")
    parser.add_argument("--noplan-model", action="store_true")
    parser.add_argument("--noplan-target", choices=["raw", "log"], default="raw")
    parser.add_argument("--noplan-train", choices=["group", "all", "weighted"], default="group")
    parser.add_argument("--noplan-weight", type=float, default=20.0)
    parser.add_argument("--noplan-iterations", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=1, help="models per fold, averaged")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--folds", default="", help="1-based fold indices, e.g. 1,3,5 (default all)")
    args = parser.parse_args()

    dropped = [c.strip() for c in args.drop.split(",") if c.strip()]
    feats = [f for f in FEATURES if f not in dropped]
    cat_idx = [feats.index(c) for c in CATEGORICAL if c not in dropped]

    chosen = FOLDS
    if args.folds:
        chosen = [FOLDS[int(i) - 1] for i in args.folds.split(",")]

    frame = load_training_features()
    print(f"[{args.tag}] {len(feats)} features, {len(FOLDS)} folds, "
          f"{args.iterations} iterations" + (f", winsor {args.winsor:,.0f}s" if args.winsor else ""))

    folds = [run_fold(frame, held, feats, cat_idx, args) for held in chosen]
    scores = np.array([f["rmse_bagged"] for f in folds])
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
