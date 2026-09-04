"""Predict on the ranking set and assemble a submission.

    python -m prc.predict                 # writes submissions/<team>_v<N>.parquet
    python -m prc.predict --upload        # ...and puts it in the team bucket

The ranking file is passed to ``features.build`` whole, arrivals included: the
arrival rows are what make the congestion features computable at prediction
time, and build() drops them once they have done their job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .config import INTERIM_DIR, RAW_DIR, TEAM_NAME, ensure_dirs
from .features import FEATURES, build
from .submit import ID_COL, TARGET_COL, build as build_submission

DEFAULT_MODEL = Path("models/catboost_baseline.cbm")


def predict(model_path: Path = DEFAULT_MODEL, ranking: Path | None = None) -> Path:
    from catboost import CatBoostRegressor

    ensure_dirs()
    ranking = ranking or RAW_DIR / "ranking.parquet"
    frame = build(pl.read_parquet(ranking), with_target=False)
    print(f"{frame.height:,} departure rows to predict")

    model = CatBoostRegressor()
    model.load_model(str(model_path))

    raw = model.predict(frame.select(FEATURES).to_pandas())
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
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--version", type=int)
    parser.add_argument("--upload", action="store_true", help="put the file in the team bucket")
    args = parser.parse_args()

    preds = predict(args.model)
    submission = build_submission(
        preds, RAW_DIR / "submitting.parquet", team=args.team, version=args.version
    )
    if args.upload:
        from .data import upload_submission

        upload_submission(submission)


if __name__ == "__main__":
    main()
