# v8 worked. v9 did nothing, and the reason matters more than the result.

2026-09-10

    v7  330.23
    v8  300.35   -29.88   rank 48 -> 31 of 95
    v9  300.42    +0.07   no change

## v8

Airport routing, the per-airport baseline, and two correctness fixes. It landed
at **300.35**, better than the fold-(1,7) reading (304.3) and well past the
six-fold mean reading (318.7).

Worth recording why this one was uncertain: `--noplan-split-lirf` had never
executed in `final.py` until the fix that morning, so no previous submission
carried the routing at all, and there was no precedent for how it would
transfer. The build log finally shows it firing:

    no-plan routed: LIRF 1,488, other 20,982

## v9, and a prediction that was wrong by a factor of 335

The band-mixture patch on the 181 LIRF no-plan rows with `gap_sched >= 7,200`.

| | |
|---|---|
| predicted change in board SSE | **−4.953e9** |
| actual | **+1.477e7** |
| ratio | **335x** |

The estimate rested on one assumption, which I flagged in the note, in the
commit message and in the tool's own output every time it ran: *2025's per-bin
conditional distribution transfers to 2026*. It does not.

The arithmetic is worth keeping because it says something precise. For fixed
truth `t`, moving a prediction from `p8` to `p9` changes squared error by
`(p9-p8)(p9+p8-2t)`. Summed over the 181 rows that came to +1.477e7 on a mean
move of 2,787s, which gives

    p8 + p9 - 2t  ~  62 s

**The truth sits almost exactly halfway between what v8 predicted and what the
patch predicted.** v8 was already close on these rows. They are not carrying the
error I attributed to them.

## What this overturns

`notes/2026-09-10-the-lirf-pocket.md` claims 348 rows hold ~57% of our board
squared error, and that a perfect pocket would put us at 225. That number came
from 2025's conditional variance applied to 2026 row counts. The scored set has
now answered directly: **it does not.** The 2026 pocket is far tamer than
2025's — the rollovers and multi-hour substitutions that dominate 2025 are not
there in the same proportion.

Everything built on it inherits the correction:

- the band patch: measured, worth ~0
- the substitution classifier: aimed at the same rows, so its 25% capture on
  2025 folds is worth ~0 on the board too
- "the road to 245 runs through the pocket": wrong

## Where the remaining 55 seconds actually is

By elimination, in the bulk — the 98.9% of rows that three separate analyses
examined and found no lever in. That is an uncomfortable place to be pointed,
but it is where the measurement points.

The two experiments already running are, by luck rather than judgement, the
right ones:

- `--bulk-plan-only`: the global model trains on 22,470 rows whose predictions
  it discards, and on the residual scale they carry sd ~3,400s against a few
  hundred for the rest.
- `--noplan-routes LIRF,LFPG,LSZH`: dispersion at two more airports.

## The lesson, stated so it survives

A conditional distribution measured on one period is not evidence about
another, however carefully the counts are matched. I did match the counts — the
pocket looked *heavier* in the scored months — and it made no difference,
because what transferred was the row count and not the distribution of the
target within it.

Where a claim rests on that kind of transfer, the honest forecast is a range
that includes zero, not a point estimate with a caveat attached.
