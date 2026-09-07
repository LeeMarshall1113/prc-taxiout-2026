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
    *_WEATHER,
]


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
    pool = Pool(train.select(feats).to_pandas(), train[TARGET].to_numpy().astype(float),
                cat_features=cat_idx)

    rank = build(pl.read_parquet(RAW_DIR / "ranking.parquet"), with_target=False)
    rank_x = rank.select(feats).to_pandas()
    print(f"scoring {rank.height:,} ranking rows")

    preds = []
    for seed in range(args.seeds):
        model = CatBoostRegressor(
            iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
            loss_function="RMSE", thread_count=args.threads,
            random_seed=1113 + seed * 977, verbose=250,
        )
        model.fit(pool)
        preds.append(model.predict(rank_x))
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
