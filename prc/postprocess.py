"""Prediction-level post-processing, chosen on folds rather than guessed.

Our error decomposition says the damage is large wrong predictions, not a
poorly-fitted bulk: 250 holdout rows where the model predicted above 5,000s
carried 51.3% of squared error, and 74 of those were ordinary departures under
2,000s. RMSE is symmetric, so predicting 40,000s on a normal row costs exactly
what missing a real 40,000s row costs. That makes the aggressive end of the
prediction range worth calibrating.

Every transform here is a pure function of the predictions, so the sweep runs
offline against saved fold predictions with no retraining. A transform is only
accepted if it beats the identity on a MAJORITY OF FOLDS, not on the mean --
one fold carried every wrong call this project has made so far.

    python -m prc.postprocess --tag bag3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def identity(p: np.ndarray) -> np.ndarray:
    return p


def clip_at(threshold: float):
    """Hard ceiling. The bluntest way to stop a model betting big."""
    def f(p: np.ndarray) -> np.ndarray:
        return np.minimum(p, threshold)
    f.__name__ = f"clip@{threshold:,.0f}"
    return f


def soften_above(threshold: float, alpha: float):
    """Keep the ordering above ``threshold`` but compress the excess.

    alpha=0 is a hard clip, alpha=1 is the identity. Anything between says "the
    model is right that this row is unusual, but it is overstating by how much".
    """
    def f(p: np.ndarray) -> np.ndarray:
        excess = np.maximum(p - threshold, 0.0)
        return np.minimum(p, threshold) + alpha * excess
    f.__name__ = f"soften@{threshold:,.0f}x{alpha:g}"
    return f


def shrink_to_mean(alpha: float):
    """Pull every prediction toward the training mean. Pure variance reduction."""
    def f(p: np.ndarray) -> np.ndarray:
        return p.mean() + alpha * (p - p.mean())
    f.__name__ = f"shrink{alpha:g}"
    return f


def candidates() -> list:
    out = [identity]
    for t in (3000, 5000, 8000, 12000, 20000, 30000):
        out.append(clip_at(t))
        for a in (0.25, 0.5, 0.75):
            out.append(soften_above(t, a))
    out += [shrink_to_mean(a) for a in (0.9, 0.95, 0.98)]
    return out


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(((y - p) ** 2).mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="bag3")
    parser.add_argument("--column", default="bagged", help="prediction column to transform")
    args = parser.parse_args()

    files = sorted(Path("results").glob(f"preds_{args.tag}_*.parquet"))
    if not files:
        raise SystemExit(f"no saved predictions for tag {args.tag!r}")
    folds = [pl.read_parquet(f) for f in files]
    print(f"{len(folds)} folds: {', '.join(f.stem.replace('preds_' + args.tag + '_', '') for f in files)}")

    base = [rmse(f["y"].to_numpy(), f[args.column].to_numpy()) for f in folds]
    print(f"\nidentity per fold: {'  '.join(f'{b:.2f}' for b in base)}   mean {np.mean(base):.2f}\n")

    rows = []
    for fn in candidates():
        scores = [rmse(f["y"].to_numpy(), fn(f[args.column].to_numpy())) for f in folds]
        deltas = np.array(scores) - np.array(base)
        rows.append((fn.__name__, np.mean(scores), deltas, int((deltas < 0).sum())))

    rows.sort(key=lambda r: r[1])
    print(f"{'transform':<22} {'mean':>9} {'vs identity':>12} {'per-fold delta':>28} {'wins':>6}")
    for name, mean, deltas, wins in rows[:16]:
        per = " ".join(f"{d:+7.2f}" for d in deltas)
        print(f"{name:<22} {mean:>9.2f} {mean - np.mean(base):>+12.2f}   {per:>28} {wins:>4}/{len(folds)}")

    best = min(rows, key=lambda r: r[1])
    print()
    if best[0] == "identity":
        print("VERDICT: identity wins. No post-processing is justified.")
    elif best[3] == len(folds):
        print(f"VERDICT: {best[0]} wins every fold ({best[1] - np.mean(base):+.2f}s). Safe to apply.")
    elif best[3] > len(folds) / 2:
        print(f"VERDICT: {best[0]} wins {best[3]}/{len(folds)} folds ({best[1] - np.mean(base):+.2f}s). Marginal.")
    else:
        print(f"VERDICT: best mean is {best[0]} but it wins only {best[3]}/{len(folds)} folds. Reject.")


if __name__ == "__main__":
    main()
