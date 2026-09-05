# The reference feature is rejected, and the folds explain everything else

Paired test: three month-pair folds, 800 iterations, same data, the only
difference being `ref_taxi_s` + `ref_level`.

| fold | no ref | with ref | delta | bulk no ref | bulk ref | delta |
|---|---:|---:|---:|---:|---:|---:|
| (1, 7) | 460.71 | 457.34 | **−3.38** | 279.95 | 278.88 | −1.07 |
| (3, 9) | 306.91 | 311.29 | +4.38 | 218.28 | 217.81 | −0.47 |
| (5, 11) | 316.44 | 325.10 | +8.66 | 213.58 | 222.72 | +9.14 |
| **mean** | **361.36** | **364.58** | **+3.22** | 237.27 | 239.80 | +2.53 |

Wins 1 of 3 folds overall, 2 of 3 on bulk. **Rejected** — no evidence of
improvement, and the point estimate is worse.

## Why a +0.378 correlation bought nothing

`ref_taxi_s` correlates +0.378 with the target, easily the strongest single
association we measured, and it is still worthless. CatBoost's handling of
categorical features *is* target encoding — it computes ordered target
statistics for `STAND_mvt`, `RUNWAY_mvt` and `ADEP_mvt` internally. We handed it
a hand-rolled target encoding of exactly those keys.

The lesson generalises past this feature: **correlation with the target is not
incremental value given the other features.** Everything we have added since v1
has been measured the wrong way round — by association with the target rather
than by what it adds to a model that already has the rest. That is the same
error, three times: `airport_plan` (+7s on the board), the wave-2 features
(within noise), and now this.

## The finding that matters more

Fold RMSE ranges from **306.91 to 460.71** — a 154s spread on identical models
and identical settings, purely from which two months are held out.

`(1, 7)` is by far the hardest fold, and **the ranking set is January and July**.
That single fact retroactively explains most of this project's confusion:

- Why our single holdout was so noisy: we picked the year's worst fold as the
  only thing we measured against.
- Why the 95% interval on that holdout was 159s wide.
- Why v2 and v3 improved on it and regressed on the board.
- Why the board reads 423.12 while our other folds read ~310. Those numbers were
  never comparable and we kept comparing them.

Sanity check in the right direction: fold (1,7) on 2025 gives 460.71, the board
gives 423.12 on 2026. 2026's January and July are somewhat easier than 2025's,
which agrees with the arrival-tail comparison done on day one.

## What this means for how we measure from here

Selection must be on the **fold mean**, and a change must win a majority of
folds. Neither the single Jan+Jul holdout nor any single-fold number should be
quoted again as evidence for a change. On this evidence, no feature added since
v1 has earned its place — which is consistent with v1 still being our best
submission at 423.12.
