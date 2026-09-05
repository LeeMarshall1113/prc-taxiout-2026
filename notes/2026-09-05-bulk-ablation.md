# Bulk ablation, and a correction to the selection rule — 2026-09-04

Six fits, crossing two dimensions: the wave-2 congestion features, and capping
training labels. `airport_plan` dropped throughout (it cost ~7s). Identical
config, 1500 iterations, od_wait 100.

| tag | feats | winsor | trimmed | **plain RMSE** | bulk | no-plan |
|---|---:|---:|---:|---:|---:|---:|
| wave2 | 32 | 0 | 265.90 | **452.58** | 282.3 | 2,846 |
| base | 28 | 0 | 266.58 | 463.96 | 284.2 | 2,950 |
| wave2_w20k | 32 | 20,000 | **253.26** | 481.03 | **275.1** | 3,171 |
| w20k | 28 | 20,000 | 255.34 | 484.57 | 276.8 | 3,196 |
| wave2_w7200 | 32 | 7,200 | 253.50 | 529.27 | 279.1 | 3,611 |
| w7200 | 28 | 7,200 | 255.02 | 530.56 | 280.8 | 3,615 |

## Correction: the selection rule was too broad

I set "select on trimmed RMSE" after it predicted the leaderboard on the
v1-vs-v2 comparison. That rule is wrong as stated. Winsorising is the best thing
in the table on trimmed RMSE (253.26 vs 266.58) and among the worst on plain
RMSE (481.03 vs 463.96). Both readings are correct: capping training labels buys
bulk accuracy by giving up the extremes, and trimming is precisely the operation
that hides that trade.

The rule needs its scope: **trimmed RMSE is valid for changes that do not target
the tail.** v1-vs-v2 qualified — two models with near-identical tail behaviour,
where trimming removed variance without removing the effect. Winsorising does
not qualify. Anything that changes what the model does with extremes has to be
judged on plain RMSE, noise and all, with a paired bootstrap to get resolution.

Selecting on trimmed here would have shipped a model about 28s worse.

## Paired bootstrap, plain RMSE, 400 resamples

Against `wave2`:

| variant | difference | 95% CI | verdict |
|---|---:|---|---|
| base | +11.60s | [−0.99, +25.36] | indistinguishable |
| wave2_w20k | +26.42s | [−2.39, +54.73] | indistinguishable |
| w20k | +30.52s | [+3.75, +59.14] | **worse** |
| wave2_w7200 | +75.06s | [+33.16, +119.89] | **worse** |
| w7200 | +78.40s | [+30.92, +124.41] | **worse** |

Winsorising at 7,200s is decisively bad. The wave-2 features are ahead of base
on the point estimate but the interval still touches zero, so on this evidence
they are promising rather than proven.

## The split, which is what got submitted

No single model is best at both jobs. The capped model has the better bulk
(275.1s vs 282.3s) and the worse tail (3,171s vs 2,846s), for the obvious
reason that a model capped at 20,000s cannot predict above 20,000s. So route on
`has_flight_plan` and use each where it wins:

| | plain RMSE | bulk | no-plan |
|---|---:|---:|---:|
| wave2 | 452.58 | 282.3 | 2,846 |
| wave2_w20k | 481.03 | 275.1 | 3,171 |
| **split** | **448.20** | **275.1** | **2,846** |

Paired against `wave2`: **−4.39s, CI [−5.38, −3.59], P(better) = 1.00.** Small
but unambiguous, and the only result in this round whose interval clears zero.

Submitted as `jolly-lobster_v3.parquet` (bulk `wave2_w20k`, tail `wave2`).
