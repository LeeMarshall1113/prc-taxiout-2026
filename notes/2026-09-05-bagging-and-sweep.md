# Seed-bagging accepted, post-processing rejected — 2026-09-05

## Seed-bagging: wins 3 of 3 folds

| fold | single seed | bagged (3) | gain |
|---|---:|---:|---:|
| (1, 7) | 467.85 | 464.56 | **+3.30** |
| (3, 9) | 308.53 | 301.94 | **+6.59** |
| (5, 11) | 318.27 | 317.88 | **+0.38** |
| mean | 364.88 | 361.46 | **+3.42** |

Accepted. Consistent with the 2024 challenge winner, who got about 3% from
averaging 20 seeds of one LightGBM and beat a five-model ensemble doing it.

## Post-processing: identity wins outright

28 transforms — hard clips at 3k to 30k seconds, softening that keeps the
ordering above a threshold but compresses the excess, and global shrinkage
toward the mean. Every one of them lost, on every fold:

| transform | mean | vs identity | wins |
|---|---:|---:|---:|
| **identity** | **361.46** | — | — |
| soften@30,000x0.75 | 361.70 | +0.24 | 0/3 |
| shrink0.98 | 361.77 | +0.31 | 0/3 |
| soften@30,000x0.5 | 362.07 | +0.61 | 0/3 |
| clip@30,000 | 363.21 | +1.75 | 0/3 |
| soften@12,000x0.5 | 370.53 | +9.07 | 0/3 |

Not one win in 84 transform-fold comparisons, and the loss grows monotonically
as the transform bites harder.

**This kills the hypothesis it was built to test.** The reasoning was that 250
holdout rows predicted above 5,000s carried 51.3% of squared error while 74 of
them were ordinary departures, so the model was betting too big. The sweep says
the opposite: those bets pay. Where the model predicts an extreme value it is
right often enough to cover the times it is wrong, and every attempt to make it
more cautious costs more than it saves.

Worth stating because it closes a line of attack: the tail is not
over-prediction, and it is not fixable at the prediction layer. Nothing that
only reshapes the output is going to help.

## What ships in v4

- 3 seeds, averaged. Measured, 3/3 folds.
- Trained on **all 2,085,047 rows** of 2025. v1, v2 and v3 were each fitted on
  1,578,296 rows — months 2-6, 8-10, 12 — with month 11 spent on early stopping
  and months 1 and 7 held out. The ranking set is January and July 2026, so we
  had been withholding the only matching season in the data, 24.3% of it.
- v1's feature set. Everything added since has been rejected on folds:
  `airport_plan`, the four wave-2 congestion features, and the EUROCONTROL
  reference feature.
- No post-processing.

Expected gain is modest: ~3.4s from bagging plus an unmeasured amount from the
extra data. That is worth a few places on a board where 17th to 19th spans 16
seconds. It is not a route back into contention, and nothing found today
suggests one exists at the prediction layer.
