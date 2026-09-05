"""Predict on the ranking set and assemble a submission.

    python -m prc.predict                 # writes submissions/<team>_v<N>.parquet
    python -m prc.predict --upload        # ...and puts it in the team bucket

The ranking file is passed to ``features.build`` whole, arrivals included: the
arrival rows are what make the congestion features computable at prediction
time, and build() drops them once they have done their job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import INTERIM_DIR, RAW_DIR, TEAM_NAME, ensure_dirs
from .features import FEATURES, build
from .submit import ID_COL, TARGET_COL, build as build_submission

def variant(tag: str) -> dict:
    """Look a run up in the ablation log.

    The model file alone does not say which features it was fitted on, and
    guessing wrong produces a silently wrong submission rather than an error, so
    the feature list always comes back from the run's own record.
    """
    path = Path("results/ablation.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run prc.train first")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    matching = [r for r in rows if r["tag"] == tag]
    if not matching:
        raise KeyError(f"no run tagged {tag!r} in {path}; have {sorted({r['tag'] for r in rows})}")
    return matching[-1]


def best_variant() -> dict:
    """The run with the lowest trimmed RMSE — the selection metric."""
    path = Path("results/ablation.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return min(rows, key=lambda r: r["trimmed_rmse"])


def _load(run: dict):
    from catboost import CatBoostRegressor

    feats = [f for f in FEATURES if f not in run.get("dropped", [])]
    model = CatBoostRegressor()
    model.load_model(str(Path("models") / f"{run['tag']}.cbm"))
    if model.feature_names_ and list(model.feature_names_) != feats:
        raise RuntimeError(
            f"feature mismatch for {run['tag']}:\n  model: {list(model.feature_names_)}\n"
            f"  rebuilt: {feats}"
        )
    return model, feats


def predict(run: dict, ranking: Path | None = None, tail_run: dict | None = None) -> Path:
    """Score the ranking set. With ``tail_run``, route by ``has_flight_plan``.

    The split exists because the two jobs want different models. Capping
    training labels buys a better bulk (holdout 275.1s against 282.3s) and pays
    for it on the rows with no flight plan (3,171s against 2,846s), since a
    capped model cannot predict a value above its cap. Neither model is best at
    both, so each is used where it wins: paired bootstrap has the split at
    -4.39s [-5.38, -3.59] against the better single model, P(better) = 1.00.
    """
    ensure_dirs()
    ranking = ranking or RAW_DIR / "ranking.parquet"
    frame = build(pl.read_parquet(ranking), with_target=False)
    print(f"{frame.height:,} departure rows to predict")

    bulk_model, bulk_feats = _load(run)
    print(f"  bulk  <- {run['tag']:<12} ({len(bulk_feats)} feats, winsor {run.get('winsor', 0):,.0f}s, "
          f"holdout bulk {run.get('rmse_bulk', float('nan')):.1f}s)")
    raw = bulk_model.predict(frame.select(bulk_feats).to_pandas())

    if tail_run is not None:
        tail_model, tail_feats = _load(tail_run)
        print(f"  tail  <- {tail_run['tag']:<12} ({len(tail_feats)} feats, "
              f"winsor {tail_run.get('winsor', 0):,.0f}s, "
              f"holdout no-plan {tail_run.get('rmse_noplan', float('nan')):.0f}s)")
        tail_raw = tail_model.predict(frame.select(tail_feats).to_pandas())
        route = frame["has_flight_plan"].to_numpy() == 1
        raw = np.where(route, raw, tail_raw)
        print(f"  routed {int((~route).sum()):,} rows to the tail model")

    clipped = np.clip(raw, 0.0, None)
    negatives = int((raw < 0).sum())
    if negatives:
        print(f"  clipped {negatives:,} negative predictions to 0")
    print(
        f"  predictions: mean={clipped.mean():.1f}s median={np.median(clipped):.1f}s "
        f"min={clipped.min():.1f}s max={clipped.max():.1f}s"
    )
    plan = frame["has_flight_plan"].to_numpy()
    for label, mask in (("flight plan present", plan == 1), ("flight plan MISSING", plan == 0)):
        if mask.any():
            print(f"  {label:<21} n={mask.sum():>8,}  mean prediction {clipped[mask].mean():8.1f}s")

    out = INTERIM_DIR / "ranking_predictions.parquet"
    pl.DataFrame(
        {ID_COL: frame[ID_COL], TARGET_COL: np.rint(clipped).astype(np.int32)}
    ).write_parquet(out)
    print(f"  wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="ablation tag for the bulk model")
    parser.add_argument("--tail-tag", help="ablation tag used for rows with no flight plan")
    parser.add_argument("--best", action="store_true", help="lowest trimmed RMSE, single model")
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--version", type=int)
    parser.add_argument("--upload", action="store_true", help="put the file in the team bucket")
    args = parser.parse_args()

    run = variant(args.tag) if args.tag else best_variant()
    tail_run = variant(args.tail_tag) if args.tail_tag else None
    preds = predict(run, tail_run=tail_run)
    submission = build_submission(
        preds, RAW_DIR / "submitting.parquet", team=args.team, version=args.version
    )
    if args.upload:
        from .data import upload_submission

        upload_submission(submission)


if __name__ == "__main__":
    main()
