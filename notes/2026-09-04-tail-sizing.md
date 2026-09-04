# Tail sizing — 2026-09-04

Question this was meant to answer: how much of the gap between the field
(~320s) and the leaders (~277s) is tail behaviour rather than bulk modelling?

Answer: **the tail is not a nuisance, it is the problem.** 1.08% of departures
hold 41.8% of the target's variance, and that 1.08% is identifiable in advance.

## The tail has a name: rows with no flight plan

Departures split cleanly on whether a flight-plan row joined (`AOBT_3_flt`, and
every other `_flt` column, non-null):

| group | rows | share | mean | sd | p99.9 | P(y > 2h) |
|---|---:|---:|---:|---:|---:|---:|
| flight plan present | 2,062,577 | 98.92% | 986.9s | **417.5s** | 3,959s | 0.005% |
| flight plan MISSING | 22,470 | **1.08%** | 1,384.4s | **3,397.5s** | 52,337s | **2.14%** |

A missing flight plan makes a >2h taxi-out **428.7x** more likely. Within-group
variance puts `0.0108 x 3397.5² = 41.8%` of all variance inside that 1.08%.

All ten of the worst residuals under a median predictor are flight-plan-less
rows. Ten of ten.

## How concentrated

Against the global mean (RMSE 546.4s):

| rows | share of data | share of variance |
|---|---:|---:|
| y >= 7,200s | 584 | 0.0280% | 41.7% |
| y >= 40,000s | 41 | 0.0020% | 30.1% |
| y >= 80,000s | 16 | 0.0008% | **20.7%** |

Sixteen rows out of 2.08 million carry a fifth of the variance.

## The 24h rollover is real but mostly invisible

> **CORRECTED 2026-09-04, later the same day.** This section is right about the
> ~87,000s rows but I over-generalised it into a story about the whole tail.
> Measured on the LIRF/no-plan group: only **1.2%** of its long rows are within
> an hour of 86,400s, and 92% are below 20,000s with a median of 7,738s. The
> rollover explains a dozen-odd extreme rows, not the group. Those long taxi
> times are, as far as the data shows, real. See `2026-09-04-v3-dead-ends.md`.

Monthly maxima cluster at 87,177 / 87,247 / 87,341 / 87,543 / 87,598 / 87,605 /
88,132 / 88,392s — all ~86,400 plus a normal taxi. These are day-rollover
errors in the off-block timestamp, not 24-hour taxis.

Of the 15 rows in the 80k–95k band, **only 1 has a flight-plan row**, and that
one shows the same 87,181s gap in `MVT_TIME - AOBT_3_flt` — so where the flight
plan exists, the corruption is visible. The other 14 have nothing to see it
with. The corruption is therefore mostly undetectable per-row, and its rows are
exactly the ones with no flight plan.

## What does not work

Using the `AOBT_3` gap as a tail detector. At `gap >= 3600s` it flags 2,064
rows at precision 0.116 and recall 0.079, and substituting the gap for those
rows makes a median-predictor baseline **worse** (552.1s -> 555.5s). Correlation
between gap and y is 0.948 among rows that are genuinely long, but the rule
cannot find them. Do not spend time here.

## The 2026 ranking set has the same tail

We cannot see the departure targets, but the ranking file keeps every arrival
taxi-in time, so the two years can be compared on arrivals:

| | n | mean | p99.9 | p99.99 | max |
|---|---:|---:|---:|---:|---:|
| 2025 Jan+Jul (train) | 343,999 | 548.2s | 2,584s | 4,405s | 33,601s |
| 2026 Jan+Jul (ranking) | 344,693 | 537.0s | 2,701s | 6,229s | 50,517s |

Same distribution, same kind of tail, if anything heavier in 2026. Nothing here
suggests the ranking set was cleaned.

Also worth knowing: **July is the hardest month in 2025** (sd 745.0s, p99.9
5,895s) and January second (sd 604.8s). The two ranking months are the two worst
months of the year, which is presumably deliberate.

## Consequences for how we model

1. `has_flight_plan` is the single most important feature, and every `_flt`
   column's missingness must be preserved rather than imputed.
2. RMSE's optimum is the conditional mean, so the flight-plan-less group should
   be predicted **high** — near its own mean of 1,384s, not the global median of
   912s. Any model that shrinks those rows toward the bulk is leaving points on
   the table, and that is the most likely thing separating 320 from 277.
3. Bulk feature work operates on the 98.92% that holds 58% of the variance. It
   is worth doing, but it cannot on its own reach the leaders.
4. Do not chase the individual extreme rows. Sixteen rows carry a fifth of the
   variance and are corrupted timestamps with no signal attached; every team
   eats them. That is a floor, not an opportunity.
