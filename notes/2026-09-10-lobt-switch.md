# The airport sometimes stamps off-block from LOBT, not AOBT

2026-09-10

Came out of the organisers' Discord export. `roma_no` posted a decomposition on
2026-09-08 that we did not have; every figure in it reproduces here exactly.

## The population

Of the 981 rows on fold (1,7) with taxi over an hour:

| class | n |
|---|---|
| `BLOCK == SCHED` within 30s | 345 |
| `+1 day` rollover | 6 |
| **neither** | **630** |

558 of the 630 carry an `AOBT_3_flt`. They hold **16.6% of the fold's squared
error** and, unlike everything we have chased for three days, they are not a Rome
story: **EGLL 234**, LFPG 133, LIRF 122, EHAM 54.

## What they are

The target follows the **last calculated** off-block, not the actual one. On
those rows:

| gap | median | median \|y − gap\| | corr with y |
|---|---|---|---|
| `gap_aobt` | 1,384 | 2,844 | **−0.061** |
| `gap_sched` | 5,338 | 894 | 0.406 |
| `gap_eobt` | 4,325 | 776 | 0.520 |
| **`gap_lobt`** | **4,381** | **780** | **0.618** |
| target | 4,192 | | |

The Network Manager sees an ordinary taxi of 1,384s; the airport reports 4,192s;
and `LOBT` — NM's last calculated off-block — sits at 4,381s, almost exactly on
the target. So where these disagree, the airport's `BLOCK_TIME` is tracking the
planned off-block rather than the sensed one.

## The rule, validated out of fold

Where `gap_lobt − gap_aobt > T`, predict `gap_lobt` instead of `gap_aobt`.
**T is chosen on the other ten months for each fold** — and converged on 4,200s
independently in all six:

| fold | rows | recorded | after | delta |
|---|---|---|---|---|
| (1,7) | 398 | 339.27 | 304.15 | **−35.12** |
| (2,8) | 129 | 287.95 | 278.14 | −9.81 |
| (3,9) | 145 | 237.74 | 214.70 | −23.04 |
| (4,10) | 125 | 222.73 | 207.03 | −15.70 |
| (5,11) | 119 | 269.69 | 260.24 | −9.45 |
| (6,12) | 238 | 271.32 | 251.69 | −19.63 |

**Wins 6/6, mean −18.79s.** Fold (1,7) maps to a board of **267** through the
measured −36.93 offset.

## Why this is not the LIRF pocket again

The pocket died because 2025's conditional distribution did not transfer to 2026.
The obvious check here, done first this time:

    rows with gap_lobt - gap_aobt > 4,200s
      2025 training      1,154   5.5 per 10k
      2026 scored set      256   7.4 per 10k

The population is present in the scored months at a slightly **higher** rate.
`LOBT_flt` is 1.53% null in `ranking.parquet` — the same rows as `AOBT_3_flt` —
so the difference computes on 98.47% of scored rows. `leakcheck` passes.

That said, the -18.79s is an arithmetic substitution on raw gaps, not a refit.
The model predicts a residual on top of a blend in which `gap_aobt` carries 0.436
and `gap_lobt` only 0.061, so the real gain will differ. **This needs a fold run
before it goes anywhere near a submission** -- which is the lesson v10 taught
four hours ago.

## Two implementations to test

- `lobt_minus_aobt`, a feature. A tree cannot form the difference itself, only
  threshold each column separately, so handing it over is not redundant with
  having both gaps present. Keeps `blend_apt`.
- `--baseline lobt_switch`, which changes the anchor above 4,200s. Closer to what
  was measured, but discards the per-airport blend.

Both are gated: the feature sits in `final.py`'s DROP, the baseline is opt-in.
