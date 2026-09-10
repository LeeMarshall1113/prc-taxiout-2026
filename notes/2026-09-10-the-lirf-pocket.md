# 348 rows hold most of our board error

2026-09-10

Two agents working from different angles arrived independently at the same
structure, which is why I spent the time verifying it rather than filing it.
Every number below I re-derived myself.

## The population

LIRF departures with no flight plan. 1,488 rows in 2025, **383 in the scored
set** — 0.11% of 344,841. The routing key is exact and serve-available:
`ADEP_mvt == "LIRF"` and `AOBT_3_flt` null.

48.5% of them are schedule substitutions (`|y − gap_sched| <= 60`), and among
those over two hours, **90.2%** are. And the substitution probability rises
monotonically with `gap_sched`:

| `gap_sched` bin | n (2025) | P(sub) | E[y] | E[y \| not sub] |
|---|---|---|---|---|
| 3,600–7,200 | 748 | 0.37 | 2,925 | 1,141 |
| 7,200–14,400 | 500 | 0.59 | 6,380 | 1,819 |
| 14,400–28,800 | 77 | 0.81 | 15,284 | 1,127 |
| 28,800–60,000 | 39 | 0.92 | 46,137 | **87,403** |
| >= 60,000 | 11 | 0.27 | 87,376 | **87,585** |

The last column is the part worth staring at. In the low bins the
non-substituted rows are ordinary taxis of 1,100–1,800s. In the high bins they
sit at ~87,400s — a day rollover, `BLOCK_TIME` dated 24h early. So the target is
a mixture of three processes with sharply different means, and which one you are
looking at is predictable from a column we already have.

## What it is costing

Joining our current live predictions to those 383 scored rows, and scoring each
bin against 2025's own conditional distribution (`E[(y−p)²] = Var(y) +
(E[y]−p)²`):

| bin | n (2026) | we predict | 2025 E[y] | our E[SSE] | bin-optimal |
|---|---|---|---|---|---|
| 3,600–7,200 | 167 | 2,692 | 2,925 | 1.25e9 | 1.00e9 |
| 7,200–14,400 | 138 | 5,549 | 6,380 | 5.00e9 | 4.36e9 |
| 14,400–28,800 | 21 | 16,624 | 15,284 | 2.30e9 | 1.23e9 |
| 28,800–60,000 | 18 | 50,292 | 46,137 | **9.19e9** | 3.43e9 |
| >= 60,000 | 4 | **62,557** | **87,376** | **3.78e9** | 1.07e9 |

Pocket expected SSE **2.15e10** against a whole-board SSE of 3.76e10 at our
330.23. **348 rows, 0.1% of the scored set, carry an estimated 57% of our board
error.** Predicting each bin's 2025 conditional mean instead would give

    RMSE 330.23 -> 280.74

Top 10 is 277.90.

## Why I am not acting on it yet

Three reasons, in order of how much they could bite.

1. **81% of the gain sits in 22 rows** — the two highest bins. That is thin
   enough that a distribution shift between 2025 and 2026 could reverse it. What
   argues the other way is that the mechanism is an identity, not a correlation:
   where `BLOCK == SCHED`, `y` *is* `gap_sched`, exactly.
2. **It is measured against v7.** v8 is fitting now and is the first build in
   which `--noplan-split-lirf` actually routes anything — the flag had never
   worked in `final.py`. A dedicated LIRF model may already close part of this.
   The honest next step is to re-measure the same table against v8's predictions
   the moment it lands.
3. **The estimate assumes 2025's per-bin distribution transfers.** The bin
   counts do look stable: 2026 has 22 rows above 28,800s in two months where a
   two-month slice of 2025 holds about 8, so if anything the pocket is heavier
   in the scored months, not lighter.

## The test, when v8 has scored

Zero fits. Rebuild the table on all of 2025, patch the scored rows with
`gap_sched >= 7,200` to `p·gap_sched + (1−p)·m` per bin, submit. Falsified if it
scores worse than v8. That is a clean, cheap, one-submission experiment, and it
is the only lever I have found that is worth more than about ten seconds.

## What this says about the gap to the leaders

It is not a secret dataset. It is that a model trained on all 2M rows with all
four timestamp gaps learns the `MVT − SCHED` identity from the ~180k substituted
with-plan rows, and carries it into this pocket. We route the pocket to a
22k-row specialist that has to relearn the identity from 1,300 rows. Seven teams
had a first submission under 300 where ours was 423, which fits "the default
pipeline already has this" better than "they found something clever".
