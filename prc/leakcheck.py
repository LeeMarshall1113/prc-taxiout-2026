"""Refuse to use a feature the ranking file cannot produce.

The target is MVT_TIME - BLOCK_TIME, so the competition withholds BLOCK_TIME:
it is 100% null across all 344,841 scored departures. Any feature derived from
it is computable in training, invisible at serve time, and will win a fold test
spectacularly before shipping a model that reads nulls.

That is not hypothetical -- block_minus_sched, block_is_sched, block_sec_00 and
stand_sub_rate were all built on 2026-09-09 and all had this defect. A fold test
would have rewarded every one of them.

So: build the matrix both ways and compare null rates per column. Anything much
emptier at serve time than in training is derived from a withheld column.
"""

from __future__ import annotations

import polars as pl

from .features import FEATURES, REFERENCE, build

# A feature legitimately null at serve time only as often as in training.
# gap_aobt and friends ride on AOBT_3_flt, which is genuinely absent for ~1.5%
# of scored rows -- that is the signal the no-plan model exists for, not a leak.
TOLERANCE = 0.20


def null_rates(frame: pl.DataFrame) -> dict[str, float]:
    built = [f for f in FEATURES if f not in REFERENCE]
    out = build(frame, with_target=False)
    n = max(out.height, 1)
    return {c: out[c].null_count() / n for c in built if c in out.columns}


def compare(train_path: str, rank_path: str) -> list[tuple[str, float, float]]:
    train = null_rates(pl.read_parquet(train_path))
    rank = null_rates(pl.read_parquet(rank_path))
    bad = []
    for c, r in sorted(rank.items()):
        t = train.get(c, 0.0)
        if r - t > TOLERANCE:
            bad.append((c, t, r))
    return bad


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/raw/training_2025-03-01_2025-04-01.parquet")
    p.add_argument("--ranking", default="data/raw/ranking.parquet")
    args = p.parse_args()

    bad = compare(args.train, args.ranking)
    if not bad:
        print("no leaks: every feature is as available at serve time as in training")
        return
    print(f"{len(bad)} feature(s) cannot be computed at serve time:\n")
    print(f"  {'feature':24s} {'train null':>11s} {'serve null':>11s}")
    for c, t, r in bad:
        print(f"  {c:24s} {t*100:10.2f}% {r*100:10.2f}%")
    print("\nEach is derived from a column the ranking file withholds. A fold test")
    print("will reward it and the submission will read nulls. Remove it.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
