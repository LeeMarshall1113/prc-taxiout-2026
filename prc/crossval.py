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


BASELINE_COLS = ["gap_aobt", "gap_eobt", "gap_lobt", "gap_sched"]


def _baseline_matrix(frame: pl.DataFrame) -> np.ndarray:
    cols = [np.nan_to_num(frame[c].to_numpy().astype(float), nan=0.0,
                          posinf=0.0, neginf=0.0) for c in BASELINE_COLS]
    cols.append(np.ones(frame.height))
    return np.column_stack(cols)


def fit_baseline(train: pl.DataFrame, mode: str):
    """Return a callable giving the baseline to residualise against.

    ``aobt``  the raw Network Manager off-block gap. What v7 ships. Alone it
              scores 384.9s against 546s for a constant, and residualising
              against it moved fold (1,7)'s bulk from 272.4 to 240.2.
    ``blend`` a least-squares combination of all four timestamp gaps, fitted on
              training rows only. gap_eobt and gap_lobt score 676.7s and 740.1s
              alone -- worse than gap_aobt but not redundant with it, since each
              is a different revision of the same estimate. If a better level
              buys what a good level bought, this is where the next gain is.

    Fitted on training rows and applied outward, like prc.reference: the
    coefficients see no row they are later used to predict.
    """
    if mode not in ("blend", "blend_apt"):
        return lambda f: np.nan_to_num(f["gap_aobt"].to_numpy().astype(float),
                                       nan=0.0, posinf=0.0, neginf=0.0)

    x = _baseline_matrix(train)
    y = train[TARGET].to_numpy().astype(float)
    # Fit on flight-plan rows only: elsewhere every gap is zero and the rows are
    # routed to the dedicated model anyway, so they would only drag the fit.
    keep = train["has_flight_plan"].to_numpy() == 1
    apt_train = train["ADEP_mvt"].to_numpy()

    def _solve(mask):
        return np.linalg.lstsq(x[mask], y[mask], rcond=None)[0]

    global_coef = _solve(keep)
    print("    baseline blend: " + "  ".join(
        f"{n}={c:+.3f}" for n, c in zip(BASELINE_COLS + ["const"], global_coef)))

    per_airport = {}
    if mode == "blend_apt":
        # Ten airports with different layouts and procedures have no reason to
        # share one linear relation between a filed estimate and actual taxi.
        for a in np.unique(apt_train[keep]):
            m = keep & (apt_train == a)
            if m.sum() >= 5000:
                per_airport[a] = _solve(m)
        print(f"    per-airport baselines fitted for {len(per_airport)} of "
              f"{len(np.unique(apt_train))} airports")

    def apply(frame: pl.DataFrame) -> np.ndarray:
        mat = _baseline_matrix(frame)
        out = mat @ global_coef
        if per_airport:
            apt = frame["ADEP_mvt"].to_numpy()
            for a, coef in per_airport.items():
                at = apt == a
                if at.any():
                    out[at] = mat[at] @ coef
        # no flight plan -> no baseline; those rows go to the dedicated model
        return np.where(frame["has_flight_plan"].to_numpy() == 1, out, 0.0)

    return apply


def _baseline(frame: pl.DataFrame) -> np.ndarray:
    """Network Manager's own off-block-to-wheels-up gap, as a physical baseline.

    A team currently around 291 on the board models the RESIDUAL from this
    rather than the target directly. The idea is that `MVT_TIME - AOBT_3_flt` is
    already a measurement of roughly the right quantity from an independent
    source -- it scores 384.9s used alone, worse than our model but far better
    than the 546s a constant gives -- so asking the trees to learn only the
    correction should be easier than asking them to reconstruct the level and
    the correction together.

    It is emphatically not a label leak: correlation with the target is 0.536,
    it matches exactly on 0.65% of rows and within a minute on 21%. Verified
    directly rather than taken on trust, because a claim that it reconstructs
    the label was circulating and is wrong.

    Zero where the flight plan is absent, so the residual reduces to the target
    for those rows -- they are routed to the dedicated model regardless.
    """
    gap = frame["gap_aobt"].to_numpy().astype(float)
    return np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_predict(train_frame, test_frame, test_x, feats, cat_idx, y_train, args, seed):
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

    if getattr(args, "residual", False):
        make_base = fit_baseline(train_frame, getattr(args, "baseline", "aobt"))
        base_train, base_test_v = make_base(train_frame), make_base(test_frame)
    else:
        base_train = base_test_v = 0.0
    fit_y = y_train - base_train
    if args.target == "log":
        fit_y = np.log1p(np.maximum(fit_y, 0.0))
    kwargs = _catboost_kwargs(args)
    kwargs["random_seed"] = 1113 + seed * 977
    model = CatBoostRegressor(**kwargs)
    train_x = train_frame.select(feats).to_pandas()
    model.fit(Pool(train_x, fit_y, cat_features=cat_idx))

    base_test = base_test_v
    if args.target != "log":
        return model.predict(test_x) + base_test

    smearing = float(np.mean(np.exp(fit_y - model.predict(train_x))))
    return np.expm1(model.predict(test_x)) * smearing + base_test


class _SchedBlend:
    """No-plan predictions softly blended toward MVT_TIME - SCHED_TIME.

    When an airport's system has no actual off-block reading it writes the
    SCHEDULED time into the block field. Measured on 2025: for departures with a
    taxi-out over two hours, the recorded block time equals the scheduled time to
    within a minute in 74% of cases, and 82.5% of the no-flight-plan rows over
    two hours have |y - gap_sched| under 60s. For those rows the target is
    EXACTLY gap_sched -- an identity, not a correlation.

    A tree cannot represent a coefficient-1 dependence on a continuous feature
    except as a staircase, which is why gap_sched being available as a feature
    has never been enough. The identity has to be supplied, the way residualising
    against the NM gap supplied it for the bulk.

    Blended, never switched. A classifier gives p = P(this row is
    schedule-substituted) and the prediction is (1-p)*model + p*gap_sched. A hard
    threshold cost another team 372 -> 625 live; a soft blend degrades gracefully
    when p is wrong, which on a mixture it often is.
    """

    def __init__(self, model, clf, feats, cfeats):
        self._m, self._c, self._f, self._cf = model, clf, feats, cfeats

    def predict(self, frame):
        base = self._m.predict(frame[self._f])
        p = self._c.predict_proba(frame[self._cf])[:, 1]
        sched = np.nan_to_num(frame["gap_sched"].to_numpy(dtype=float), nan=0.0)
        return (1.0 - p) * base + p * sched


def fit_sched_blend(train, model, nfeats, ncats, args):
    """Train the substitution classifier on the no-plan training rows."""
    from catboost import CatBoostClassifier, Pool

    sub = train.filter(pl.col("has_flight_plan") == 0)
    y = sub[TARGET].to_numpy().astype(float)
    gs = np.nan_to_num(sub["gap_sched"].to_numpy(dtype=float), nan=0.0)
    label = (np.abs(y - gs) <= 60).astype(int)
    if label.sum() < 100 or label.sum() == len(label):
        return model  # nothing to learn from; leave the model alone

    cf = [f for f in nfeats if f != "gap_sched"] + ["gap_sched"]
    ci = [cf.index(c) for c in cf if c in {nfeats[i] for i in ncats}] if ncats else []
    clf = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05, thread_count=args.threads,
        random_seed=1113, verbose=False,
    )
    clf.fit(Pool(sub.select(cf).to_pandas(), label, cat_features=ci))
    print(f"    schedule-substitution classifier: {label.mean()*100:.1f}% positives "
          f"on {len(label):,} no-plan training rows")
    return _SchedBlend(model, clf, nfeats, cf)


def _fit_one_noplan(sub, nfeats, ncats, args, weights=None):
    from catboost import CatBoostRegressor, Pool

    y = sub[TARGET].to_numpy().astype(float)
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
        return model
    smear = float(np.mean(np.exp(fit_y - model.predict(x))))
    raw = model.predict

    class _Smeared:
        @staticmethod
        def predict(frame):
            return np.expm1(raw(frame)) * smear

    return _Smeared


class _RoutedNoPlan:
    """Two no-plan models, routed on whether the airport is LIRF.

    Inside the no-flight-plan group the two halves are barely the same problem:
    at LIRF the mean is 6,531s and essentially every row also lacks an aircraft
    type, while everywhere else the mean is 1,019s against a global 991s -- an
    ordinary departure that happens to be missing its flight plan.

    An earlier test showed a per-half CONSTANT loses badly to the single model
    (3,722 against 2,154), which is why this was left alone. But that measured
    whether a constant beats a model, not whether two models beat one, and those
    are different questions. Routing is on an airport code fixed in advance, not
    chosen by score, so it is not the per-subgroup selection that cost a 2024
    team 2,066 -> 2,276.
    """

    def __init__(self, lirf, other, nfeats):
        self._lirf, self._other, self._nfeats = lirf, other, nfeats

    def predict(self, frame):
        # frame is a pandas DataFrame carrying ADEP_mvt among nfeats
        is_lirf = (frame["ADEP_mvt"] == "LIRF").to_numpy()
        out = self._other.predict(frame)
        if is_lirf.any() and self._lirf is not None:
            out = np.where(is_lirf, self._lirf.predict(frame), out)
        return out


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
    if getattr(args, "sched_blend", False):
        model.fit(Pool(x, fit_y, cat_features=ncats, weight=weights))
        return fit_sched_blend(train, model, nfeats, ncats, args), nfeats

    if getattr(args, "noplan_split_lirf", False):
        lirf_rows = sub.filter(pl.col("ADEP_mvt") == "LIRF")
        other_rows = sub.filter(pl.col("ADEP_mvt") != "LIRF")
        lirf = _fit_one_noplan(lirf_rows, nfeats, ncats, args) if lirf_rows.height > 200 else None
        other = _fit_one_noplan(other_rows, nfeats, ncats, args)
        print(f"    no-plan routed: LIRF {lirf_rows.height:,} rows, other {other_rows.height:,}")
        return _RoutedNoPlan(lirf, other, nfeats), nfeats

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
    preds = [_fit_predict(train, test, test_x, feats, cat_idx, y_train, args, s)
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
    parser.add_argument("--baseline", choices=["aobt", "blend", "blend_apt"], default="aobt")
    parser.add_argument("--residual", action="store_true",
                        help="model the correction to the NM baseline, not the target")
    parser.add_argument("--loss", default="RMSE", help='e.g. "Huber:delta=2000"')
    parser.add_argument("--one-hot-max-size", type=int, default=0)
    parser.add_argument("--drop-dayoffset", action="store_true")
    parser.add_argument("--target", choices=["raw", "log"], default="raw")
    parser.add_argument("--noplan-model", action="store_true")
    parser.add_argument("--noplan-split-lirf", action="store_true")
    parser.add_argument("--sched-blend", action="store_true",
                        help="blend no-plan predictions toward MVT-SCHED where substitution is likely")
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
