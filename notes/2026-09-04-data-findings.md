# First contact with the data — 2026-09-04

Ran after `python -m prc.data pull`. Everything here is measured, not read off
the website; where the two disagree the website is wrong.

## The evaluation set changed today, mid-competition

`ranking.parquet` and `submitting.parquet` carry a LastModified of 2026-09-04
while the training files are dated 2026-08-13. The leaderboard confirms what
that means:

| scored rows | submissions | teams | window |
|---|---|---|---|
| 215,876 | 707 | 32 | 2026-09-01 09:45Z .. 2026-09-04 08:26Z |
| 344,841 | 27 | 17 | 2026-09-04 09:58Z .. ongoing |

**Every score posted before 2026-09-04 09:58Z is on a smaller, different
evaluation set and is not comparable to a current one.** The headline 249.46 is
a superseded-set number. On the set now in force the field is:

```
1. 277.2904  enthusiastic-daisy      6. 318.0584  resilient-kiwi
2. 279.3700  upbeat-goblin           7. 326.5971  vibrant-jewel
3. 279.9832  quick-boat              8. 343.3060  reliable-hamburger
4. 309.8474  intelligent-ladder      9. 345.5756  optimistic-panda
5. 312.7621  generous-jungle        10. 523.7311  jolly-snowflake
```

Only 17 of 37 scoring teams have resubmitted. Two consequences: the standings
are unusually soft right now, and anything that compares our score to a number
written down before today is meaningless. `prc.leaderboard` now refuses to mix
the two and says which set is in force.

## Structure

- 30 columns, ~307k rows per training month, **2,085,047 DEP rows across 2025**.
- `PHASE_mvt` is `DEP`/`ARR` — the files carry both. Taxi-**out** is the DEP
  rows; ARR rows are taxi-in and are context, not targets.
- **10 departure airports, not 11**: LTFM, LFPG, EGLL, EHAM, LEMD, EDDF, LEBL,
  LIRF, EDDM, LSZH. The site's data page says 11; its overview page says 10.
- Ranking file: 689,534 rows, schema identical to training. 344,841 DEP rows
  with the target blanked; the 344,693 ARR rows **keep their taxi times**.
- `submitting.parquet` ids are exactly the ranking DEP ids. Confirmed equal.

## The target is exactly reconstructable in training

`TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` for **100.00%** of
DEP rows, zero residual. (For ARR the sign flips — it is taxi-in.)

In `ranking.parquet` both `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are 100%
null for DEP, so the target is not directly recoverable. Everything else is
present, including takeoff time `MVT_TIME_UTC_mvt`.

**`AOBT_3_flt` is not a back door.** It is an actual-off-block time present on
98.9% of DEP rows, so `MVT_TIME - AOBT_3` looked like it might reconstruct the
target. It does not: corr 0.536, exact match 0.65%, only 21% within a minute,
RMSE 384.9s on its own. A strong feature, not a leak. The `EOBT_1` variant is
worse (RMSE 676.7s). The competition is sound.

## Baselines to beat (2025 DEP, in-sample)

| predictor | RMSE |
|---|---|
| global mean (991.2s) | 546.4s |
| airport x hour-of-day mean | 507.7s |
| `MVT_TIME - AOBT_3_flt` | 384.9s |
| current leaderboard best | 277.3s |

The bottom three teams on the live board (610, 662, 726) are **worse than
predicting a single constant**. Something is broken in their pipelines, which is
worth remembering when reading the percentiles as if they were a difficulty
curve.

## RMSE here is a tail problem, not a bulk problem

Target: mean 991s, median 912s, sd 546s. p99 = 2,339s, p99.9 = 4,501s,
**max = 131,167s** — 36 hours. 388 rows are <= 0.

Under the airport x hour baseline, squared error concentrates brutally:

| rows | share of data | share of total squared error |
|---|---|---|
| worst 10 | 0.0005% | **15.7%** |
| worst 100 | 0.005% | **40.6%** |
| worst 1,000 | 0.048% | 49.1% |
| worst 20,850 | 1.0% | 64.1% |

Capping the target at p99.9 would drop that baseline from 507.7s to 366.4s.

**This is the strategic fact of the competition.** Ten rows out of two million
carry a sixth of the error. Ordinary feature work moves the bulk, which is most
of the data and a minority of the loss. Before investing in features, work out
how much of the 277 -> 250 range is tail behaviour, because if it dominates then
the leaders are winning on tail calibration and matching them means doing that
deliberately rather than hoping a better model fixes it.

## Next

1. Quantify the irreducible tail: fit the bulk, then measure what an oracle on
   the top 0.1% would be worth. That sizes the real prize.
2. Congestion features. The ARR rows in the ranking file keep their times, so
   arrival pressure is knowable at prediction time; departure queue length has
   to come from `MVT_TIME` and scheduled times, since off-block is blanked.
3. Only then a CatBoost baseline, validated on Jan + Jul 2025.
