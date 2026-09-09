"""Train the submission model on ALL of 2025 and score the ranking set.

Every submission so far (v1, v2, v3) was fitted on months 2-6, 8-10 and 12 --
1,578,296 of 2,085,047 rows. Month 11 went to early stopping and months 1 and 7
were held out. That is the right discipline for *measuring* and the wrong thing
to *ship*: the ranking set is January and July 2026, and we were withholding the
only January and July the model could learn from, 24.3% of the data in total.

So this module deliberately has no holdout. It is for producing a submission
after a decision has already been made elsewhere, on `prc.crossval` folds. It
cannot tell you whether a model is good and must never be used to decide that.

Iterations are fixed at a value near where v1's early stopping landed (1057),
since there is no validation month left to stop against.

    python -m prc.final --seeds 3
    python -m prc.final --seeds 3 --transform "soften@8000x0.5" --upload
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .config import INTERIM_DIR, RAW_DIR, TEAM_NAME, ensure_dirs
from .features import CATEGORICAL, FEATURES, TARGET, build
from .submit import ID_COL, TARGET_COL, build as build_submission
from .train import load_training_features

# Dropped by default: everything rejected on the folds, plus anything not yet
# validated. The default configuration is exactly what is currently shipped, and
# additions are opt-in via --keep. A feature that has not won 3/3 folds must not
# reach a submission because it happened to exist in the feature list.
from .weather import FEATURES as _WEATHER

DROP = [
    "airport_plan",
    "dep_rwy_30min", "dep_rwy_headway", "arr_taxi_60min", "sched_demand_30min",
    "ref_taxi_s", "ref_level",
    # measured but not yet fold-validated; opt in with --keep once they win
    "prev_stand_gap", "prev_stand_headway", "prev_rwy_gap",
    "mvt_sec_00",
    *_WEATHER,
]

# Every feature that has actually won a fold test and is cleared to ship.
#
# Four times now a feature has been added to FEATURES, left out of DROP, and
# silently enrolled in the next submission without ever being validated. DROP
# is opt-out, so forgetting is the default and the failure is silent. This is
# the opt-in half: a new feature is in neither list, the check below fails at
# import, and the build refuses to start until someone decides which it is.
VALIDATED = frozenset({
    "ADEP_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "RUNWAY_mvt",
    "STAND_mvt", "WK_TBL_CAT_flt", "arr_30min", "arr_60min", "dep_30min",
    "dep_60min", "dep_delay", "dow", "doy", "gap_aobt", "gap_eobt", "gap_lobt",
    "gap_sched", "has_flight_plan", "hour", "is_weekend", "minute_of_day",
    "month", "rwy_arr_active", "rwy_dep_active", "rwy_dep_share",
    "rwy_mixed_mode", "sched_vs_eobt", "stand_runway_pair_n",
})


def _check_feature_registry() -> None:
    shipping = {f for f in FEATURES if f not in DROP}
    unclassified = shipping - VALIDATED
    if unclassified:
        raise SystemExit(
            f"these features would ship without ever having won a fold test: "
            f"{sorted(unclassified)}" + chr(10) +
            f"Add each one to prc.final.DROP (untested -- opt in later with "
            f"--keep) or to VALIDATED (it won its folds; say which run)."
        )
    stale = VALIDATED - set(FEATURES)
    if stale:
        raise SystemExit(f"VALIDATED names features that no longer exist: {sorted(stale)}")


_check_feature_registry()


def resolve_transform(spec: str):
    """Turn a name from prc.postprocess's sweep into the function it names."""
    from . import postprocess

    if not spec or spec == "identity":
        return postprocess.identity
    for fn in postprocess.candidates():
        if fn.__name__ == spec:
            return fn
    raise SystemExit(f"unknown transform {spec!r}; run `python -m prc.postprocess` to see names")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1100)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--baseline", choices=["aobt", "blend", "blend_apt"], default="aobt")
    parser.add_argument("--noplan-split-lirf", action="store_true")
    parser.add_argument("--target", choices=["raw", "log"], default="raw")
    parser.add_argument("--noplan-model", action="store_true")
    parser.add_argument("--noplan-iterations", type=int, default=600)
    parser.add_argument("--noplan-depth", type=int, default=6)
    parser.add_argument("--keep", default="",
                        help="comma-separated features to restore to the drop list, "
                             "e.g. the wave-2 columns if they win on folds")
    parser.add_argument("--transform", default="identity")
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--version", type=int)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    from catboost import CatBoostRegressor, Pool

    ensure_dirs()
    keep = {c.strip() for c in args.keep.split(',') if c.strip()}
    unknown = keep - set(DROP)
    if unknown:
        raise SystemExit(f'--keep names features that are not dropped: {sorted(unknown)}')
    feats = [f for f in FEATURES if f not in DROP or f in keep]
    if keep:
        print(f'restored to the model: {sorted(keep)}')
    cat_idx = [feats.index(c) for c in CATEGORICAL if c not in DROP]

    train = load_training_features()
    print(f"training on ALL {train.height:,} rows of 2025 ({len(feats)} features, {args.seeds} seeds)")

    rank = build(pl.read_parquet(RAW_DIR / "ranking.parquet"), with_target=False)
    rank_x = rank.select(feats).to_pandas()
    print(f"scoring {rank.height:,} ranking rows")

    preds = []
    y_train = train[TARGET].to_numpy().astype(float)
    # Model the correction to Network Manager's own off-block gap rather than
    # the target. Won 3/3 folds at -6.44s, and -16.22s on the Jan+Jul fold that
    # matches the ranking months. Zero where no flight plan exists, so those
    # rows reduce to the plain target and are routed to the dedicated model.
    from .crossval import fit_baseline

    if args.residual:
        # Fit the baseline on training rows and apply it outward, exactly as the
        # fold rig does. This previously called the raw-gap helper directly, so
        # a --baseline choice would have been measured on folds and then
        # silently NOT shipped -- the third time a validated setting could not
        # reach a submission because the two paths had drifted apart.
        make_base = fit_baseline(train, args.baseline)
        base_train, base_rank = make_base(train), make_base(rank)
        print(f"  residual mode ({args.baseline}): baseline mean "
              f"{np.mean(base_rank):.1f}s on the ranking set")
    else:
        base_train = base_rank = 0.0
    fit_y = y_train - base_train
    if args.target == "log":
        fit_y = np.log1p(np.maximum(fit_y, 0.0))
    train_x = train.select(feats).to_pandas()
    for seed in range(args.seeds):
        model = CatBoostRegressor(
            iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
            loss_function="RMSE", thread_count=args.threads,
            random_seed=1113 + seed * 977, verbose=250,
        )
        model.fit(Pool(train_x, fit_y, cat_features=cat_idx))
        if args.target == "log":
            # Duan smearing: exp of a log-scale prediction is the conditional
            # median, and RMSE wants the mean. See prc.crossval._fit_predict.
            smear = float(np.mean(np.exp(fit_y - model.predict(train_x))))
            preds.append(np.expm1(model.predict(rank_x)) * smear + base_rank)
            print(f"  seed {seed} done (log target, smearing {smear:.4f})")
        else:
            preds.append(model.predict(rank_x) + base_rank)
            print(f"  seed {seed} done")

    bagged = np.vstack(preds).mean(axis=0)

    if args.noplan_model:
        from .crossval import NOPLAN_DROP, fit_noplan

        class _A:  # fit_noplan reads its settings off an args-like object
            noplan_iterations = args.noplan_iterations
            noplan_depth = args.noplan_depth
            threads = args.threads

        cat_names = [c for c in CATEGORICAL if c not in DROP]
        nmodel, nfeats = fit_noplan(train, feats, cat_names, _A)
        npred = nmodel.predict(rank.select(nfeats).to_pandas())
        route = rank["has_flight_plan"].to_numpy() == 0
        print(f"  dedicated no-plan model: {len(nfeats)} features, routing {int(route.sum()):,} rows")
        print(f"    global mean on those rows {bagged[route].mean():8.1f}s -> "
              f"dedicated {npred[route].mean():8.1f}s")
        bagged = np.where(route, npred, bagged)

    transform = resolve_transform(args.transform)
    shaped = transform(bagged)
    changed = int((np.abs(shaped - bagged) > 1e-6).sum())
    print(f"\ntransform {transform.__name__}: changed {changed:,} of {len(bagged):,} predictions")

    final = np.clip(shaped, 0.0, None)
    print(f"  mean={final.mean():.1f}s median={np.median(final):.1f}s max={final.max():.1f}s")

    out = INTERIM_DIR / "ranking_predictions.parquet"
    pl.DataFrame({ID_COL: rank[ID_COL], TARGET_COL: np.rint(final).astype(np.int32)}).write_parquet(out)
    submission = build_submission(out, RAW_DIR / "submitting.parquet",
                                  team=args.team, version=args.version)
    if args.upload:
        from .data import upload_submission

        upload_submission(submission)


if __name__ == "__main__":
    main()
