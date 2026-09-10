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


def parser_parity() -> list[str]:
    """Both entry points must be able to set every setting fit_noplan reads.

    crossval measures, final ships, and they have drifted apart six times. On
    2026-09-10 the drift was the other way round for once: final gained every
    flag and crossval was the one missing --noplan-depth, which cost a fold run.
    """
    import inspect
    import re

    from . import crossval, final

    out = []
    for mod, name in ((crossval, "crossval"), (final, "final")):
        src = inspect.getsource(mod)
        opts = {m.group(1).replace("-", "_")
                for m in re.finditer(r'add_argument\("--([a-z0-9-]+)"', src)}
        for k in crossval.NOPLAN_SETTINGS:
            if k not in opts:
                out.append(f"{name} cannot set --{k.replace('_','-')}")
    return out


def value_ranges(frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-numeric-feature (min, max), ignoring nulls and NaNs."""
    import numpy as np

    from .features import CATEGORICAL

    built = [f for f in FEATURES if f not in REFERENCE and f not in CATEGORICAL]
    out = build(frame, with_target=False)
    ranges = {}
    for c in built:
        if c not in out.columns:
            continue
        v = out[c].to_numpy().astype(float)
        v = v[~np.isnan(v)]
        if len(v):
            ranges[c] = (float(v.min()), float(v.max()))
    return ranges


def compare_framing(rank_path: str) -> list[tuple[str, float, float, float]]:
    """Does build() give the same answer per row when the frame is split?

    Training calls build() once per monthly file. The real serve-time call
    passes the whole of ranking.parquet, which holds January AND July 2026 in
    one frame. Any feature computed over "whatever rows are in this frame" --
    a per-key rate, a search for a neighbouring row -- therefore runs a
    different computation at serve time than the one the folds validated.

    So build the SAME rows both ways and diff per row. Same data, same
    calendar, same year: every difference is frame composition alone, which
    makes this free of the drift false-positives a train-vs-serve range
    comparison drowns in.

    Caught in one pass on 2026-09-09: stand_runway_pair_n (13,446 rows served
    exactly half their trained value) and prev_stand_headway (203.7 days on
    2,154 rows, against 31 the most a training month can hold).
    """
    import numpy as np

    from .features import CATEGORICAL

    raw = pl.read_parquet(rank_path)
    numeric = [f for f in FEATURES if f not in REFERENCE and f not in CATEGORICAL]

    served = build(raw, with_target=False)
    months = raw.select(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1mo")).to_series().unique()
    split = pl.concat([
        build(raw.filter(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1mo") == m),
              with_target=False)
        for m in sorted(months)
    ])

    key = "MVT_ID_mvt"
    j = served.join(split.rename({c: f"_s_{c}" for c in numeric if c in split.columns}),
                    on=key, how="inner")
    bad = []
    for c in numeric:
        if c not in j.columns or f"_s_{c}" not in j.columns:
            continue
        a = j[c].to_numpy().astype(float)
        b = j[f"_s_{c}"].to_numpy().astype(float)
        ok = ~(np.isnan(a) & np.isnan(b))
        differs = ok & ~np.isclose(np.nan_to_num(a, nan=-1e18),
                                   np.nan_to_num(b, nan=-1e18), rtol=1e-6, atol=1e-9)
        if differs.any():
            scale = max(np.nanmax(np.abs(b[ok])) if ok.any() else 1.0, 1e-9)
            worst = np.nanmax(np.abs(np.nan_to_num(a[differs], nan=0.0)
                                     - np.nan_to_num(b[differs], nan=0.0)))
            bad.append((c, differs.mean(), worst, scale))
    return bad


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

    parity = parser_parity()
    if parity:
        print("parser parity:")
        for line in parity:
            print("  " + line)
        print("")

    rng = compare_framing(args.ranking)
    if rng:
        print(f"{len(rng)} feature(s) change value when the frame is split by month:")
        print("")
        print(f"  {'feature':24s} {'rows differ':>12s} {'worst delta':>14s} {'scale':>12s}")
        for c, frac, worst, scale in rng:
            print(f"  {c:24s} {frac*100:11.2f}% {worst:14,.3f} {scale:12,.1f}")
        print("")
        print("Training builds one month at a time; ranking.parquet is served as")
        print("January AND July in one frame. These features compute something")
        print("different in the two cases, so the fold result does not apply.")
        print("")

    bad = compare(args.train, args.ranking)
    if not bad:
        if rng or parity:
            raise SystemExit(1)
        print("clean: every feature is as available, and as bounded, at serve "
              "time as in training")
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
