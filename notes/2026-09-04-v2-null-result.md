# v2: the Rome interaction is a null result — 2026-09-04

Added `airport_plan`, a per-airport crossing of the flight-plan flag, as the
only change from v1. Same split, same hyperparameters.

| | v1 | v2 |
|---|---:|---:|
| holdout RMSE (Jan + Jul 2025) | 459.2255s | **458.8918s** |
| flight plan present | 278.4s | 282.6s |
| flight plan missing | 2,937.3s | 2,908.7s |
| worst 100 removed | 260.4s | 265.7s |

**0.33s.** `airport_plan` lands 4th in importance (9.82) so the model does use
it, but it buys nothing.

## Why: the premise was wrong

I claimed v1 averaged Rome in with 21,000 harmless no-flight-plan rows and so
predicted ~1,452s where ~6,500s was wanted. On the actual holdout rows:

| | LIRF/no-plan, 397 holdout rows |
|---|---|
| truth mean | 6,272s |
| **v1 predicted mean** | **6,879s** |
| v2 predicted mean | 7,378s |
| v1 RMSE on the group | 9,206s |
| v2 RMSE on the group | 9,074s |

v1 was already predicting the group *above* its true mean. CatBoost had found
Rome on its own through `ADEP_mvt` and `has_flight_plan`, without needing the
crossing handed to it. The 1,452s figure I reasoned from was the mean across the
whole no-flight-plan population, not the Rome part of it — I read a group
average as if it were the model's treatment of a subgroup.

## The real barrier is within-group variance, not the mean

The group's sd is **12,441s**. Predicting its mean perfectly still leaves
12,441s of RMSE on those rows, and that is 45% of all holdout squared error.
No amount of getting the mean right touches it.

The group is **bimodal**: of 397 rows, 244 sit below 3,000s (median 1,090s — an
ordinary taxi) and 153 sit above. A single conditional mean is wrong for both
modes at once, which is exactly why moving it around changes nothing.

Thirteen rows above 40,000s hold **31% of all v2 squared error**.

## What v3 has to do

Stop predicting a group mean and start predicting a per-row mixture. The
generative story is that a Rome no-flight-plan row is an ordinary departure with
probability 1-p and an ordinary departure plus a ~86,400s date rollover with
probability p. RMSE wants `E[y] = normal + p * 86,400`, so the whole problem is
estimating p per row.

That is a classifier on the 1,488 training rows in the group, using whatever
separates the two modes — hour of day, runway, stand, month, aircraft operator.
If p is flat at the base rate the mixture mean collapses back to the group mean
and there is nothing to win; if p varies, this is the entire remaining gap.
Check that first, cheaply, before fitting anything.

This also explains the leaderboard. Our whole-model bulk is 278-283s and the
leader's total is 277.29s, which is impossible unless they are predicting inside
this group rather than around it.
