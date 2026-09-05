# Three submissions, three regressions — 2026-09-05

| | holdout (Jan+Jul 2025) | month 11 | **leaderboard (Jan+Jul 2026)** |
|---|---:|---:|---:|
| v1 | 459.23 | 316.84 | **423.12** |
| v2 | 458.89 | — | 430.87 |
| v3 (split) | 448.20 | 308.17 | 430.94 |

Every model that beat v1 on 2025 data lost to it on 2026 data. Team best is
still v1; rank has slipped from 10 of 19 to **14 of 24** as the field grew.

## It is not the choice of holdout months

That was the obvious suspect — Jan and Jul are the two highest-variance months
of 2025, and their score is set by ~16 rows. But month 11, which is an
independent set, ranks the models the same way 2025's Jan+Jul does: split
308.17, wave2 314.15, base 316.84. Two 2025 validation sets agree with each
other and disagree with 2026.

So this is a **2025 → 2026 transfer failure**, not a bad fold.

## The regressions are suspiciously alike

v2 was +7.75s against v1 and v3 was +7.82s, despite being very different models
— v2 is v1 plus one categorical, v3 is four new features, a capped bulk model
and a routed split. Two unrelated changes landing within 0.07s of each other
does not look like model quality. It looks like something shared.

Candidates, none confirmed:

- Both were validated on 2025 and selected for it. If the 2026 extremes sit in
  different places, anything tuned to 2025's tail transfers badly and the size
  of the penalty is set by the 2026 rows, not by our model.
- Our metric is decided by a handful of rows at each end. Improving the bulk by
  9s (284.2 -> 275.1) is invisible next to one row landing differently.

## One real bug found, fixed, and too small to be the cause

`dep_rwy_headway` measured back across the January/July boundary in
`ranking.parquet`, because both months sit in one frame while training builds
per month. 85 ranking rows had a "headway" over seven days, max 199 days,
against 10 in a training month. Now bounded at six hours and NaN beyond, which
means the same thing in both frames — 0.47% NaN in training against 0.53% in
ranking, medians 118s and 120s.

Worth perhaps 2s of the 7.8s at the most generous estimate. Fixed because it is
wrong, not because it explains the result.

## Where this leaves things

Shipping more variants selected on 2025 data is not justified: it has now failed
twice, in the same direction, by the same amount. The leaderboard is the only
measurement of 2026 that exists, and deliberately mining it is off the table.

What would actually help, in order:

1. **Multi-fold validation.** Retrain holding out each month-pair in turn and
   require a change to win consistently, not once. Six folds at ~10 minutes is
   an hour of compute and would have caught both regressions.
2. **Check the bulk transfers at all.** Our bulk RMSE improved 284.2 -> 275.1
   across the ablation and the board moved the wrong way. If bulk gains do not
   transfer either, the fault is in the features, not the tail.
3. Leave the model alone until one of those says something.
