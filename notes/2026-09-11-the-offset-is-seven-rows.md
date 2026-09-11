# The fold-to-board offset is seven rows, not a season

2026-09-11

I have been quoting a -35s offset between fold (1,7) and the live score for two
days, and explaining it as season exposure: the fold holds out January and July
while the shipped model trains on them. `final.py`'s docstring says exactly that,
which made it easy to believe.

It is wrong, and the disproof is one line of arithmetic.

## The offset is constant in MSE, not in RMSE

    v7    fold 133,356 MSE   board 109,052   diff 24,305
    v8    fold 115,104 MSE   board  90,210   diff 24,894
    v11   fold 111,950 MSE   board  85,357   diff 26,593

A **constant difference in MSE** is the signature of a fixed set of rows with no
counterpart in the scored set. A population-wide effect -- never having seen a
January -- would scale with the rest of the error, not sit at a fixed 24-26k
while the total falls by a fifth.

## The rows

Fold (1,7) contains **seven departures with y >= 80,000s**:

| airport | y | |
|---|---|---|
| LFPG | 84,240 | no plan |
| LIRF | 87,177 | no plan |
| LIRF | 87,361 | no plan |
| LIRF | 87,170 | no plan |
| LIRF | 87,002 | with plan |
| LIRF | 88,132 | no plan |
| LIRF | 87,186 | no plan |

At 55,000s -- roughly what the routed specialist predicts on rows with
`gap_sched` in the 50-68k band -- they cost **20,702 MSE**, which is essentially
the whole offset.

All of 2025 holds 16 such rows, so **2.7 are expected in any two-month fold.
This one drew 7.** Our most-trusted validation fold is an unlucky Poisson draw.

And the calendar features cannot carry a 35s effect anyway: `doy` is 2.0-2.6%
of importance in the saved models, `month` 0.1-0.4%.

## What this changes

- **`--holdout-days` was built to test the season hypothesis and is no longer
  worth a fit.** The flag stays -- it is cheap and correct -- but the question is
  answered.
- **The offset is not a stable calibration.** I have been mapping fold (1,7) to
  the board by subtracting ~35, including in the v11 forecast. That worked by
  coincidence: it happens to equal the cost of rows the board does not have. It
  will not hold for a change that alters how those seven rows are predicted.
- **Fold (1,7) systematically overstates any change that helps the extreme
  tail**, which is exactly the profile of the LIRF pocket work that failed twice.
- **v12 needs re-checking before it ships.** The depth-4 result was judged on all
  no-plan rows, and folds (1,7)/(3,9)/(5,11) hold 5/2/2 of these rows. A
  shallower tree averages harder at the top of the `gap_sched` staircase, so part
  of the -59.8s may be rollover-driven and worth nothing in 2026.

## The rule that follows

Report every fold twice: with and without `y >= 80,000`. A change whose gain
disappears under the second number is a change that fits seven rows.

Related: measure the concentration of any gain. If most of the summed |delta SSE|
between control and arm sits in a handful of rows, the fold number will not
transfer. The transfer ratios already on record say this plainly -- routing,
broad, transferred at 0.88; bulk-plan-only, broad, at ~1.7; the v9 patch, 22
rows, at 0.
