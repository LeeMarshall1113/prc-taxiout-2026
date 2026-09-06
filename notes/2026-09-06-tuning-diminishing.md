# Tuning the split, and where the returns went — 2026-09-06

## The no-plan model's hyperparameters do not matter

Twelve configurations across three folds, scored against cached global
predictions. **Not one won 3/3**, against roughly 1.5 expected to do so by
chance:

| config | mean | vs shipped | per-fold | wins |
|---|---:|---:|---|---:|
| base/d6/i2000 | 303.18 | −0.67 | +1.73, −2.15, −1.59 | 2/3 |
| base/d4/i2000 | 303.57 | −0.27 | +5.82, −5.62, −1.02 | 2/3 |
| **base/d6/i600** | **303.85** | — | *shipped in v5* | — |
| wave2/d8/i2000 | 306.96 | +3.11 | +1.47, +5.38, +2.50 | 0/3 |

Per-fold swings of +5.82 to −5.62 on the best config: noise. The
majority-of-folds rule earned its keep here — on the mean alone I would have
shipped `base/d6/i2000` for a −0.67s "gain".

It also **falsifies the feature-scarcity hypothesis** the sweep was built on.
Every `wave2/*` row is worse than the shipped configuration, so the no-plan
model's 18 features were not its binding constraint. Adding movement-side
congestion features to a 22k-row model makes it worse, not better.

## The wave-2 features do help — in the global model

| config | per-fold vs v5 | mean | wins |
|---|---|---:|---:|
| split_wave2 | −1.07, −1.15, −1.08 | **−1.10** | **3/3** |
| split_ref | −0.88, +1.85, +0.68 | +0.55 | 1/3 |

A 1.10s effect with a 0.08s spread — thirteen times the fold noise, and the
cleanest small result measured here. It confirms these features were real when
first tested and were buried under a metric that was then 61% tail. The
reference feature is rejected a second time, now on a de-noised metric, and
should be considered closed.

## Where the returns went

| change | gain |
|---|---:|
| dedicated no-plan model | **−51.6s** (live) |
| training on all 12 months | −12s (live, est.) |
| seed-bagging | −3.4s |
| wave-2 features | −1.1s |
| no-plan hyperparameters | nothing |
| reference feature | nothing, twice |
| post-processing | nothing, 0/84 |

One structural fix was worth more than everything else combined by a factor of
forty. Ordinary tuning is now returning single decimal places, and the gap to
the median is 19s with 61s to tenth place. Those are not going to arrive one
second at a time; the next move has to be another structural finding, or none.

## The remaining structural asymmetry

The no-flight-plan group is itself two populations. At LIRF a no-plan departure
averages **6,531s**; everywhere else it averages **1,019s**, against a global
mean of 991s. One model currently serves both. It has `ADEP_mvt` and can split
on Rome internally, so a further split may be redundant — but it is the last
place in this data where two groups this different share a model.
