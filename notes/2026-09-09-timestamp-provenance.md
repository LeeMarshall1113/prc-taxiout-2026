# Serve-time availability, and two corrections in one day

2026-09-09

Two things happened here and the second reverses the first. Both are recorded
because the sequence is the lesson.

## Correction 1, which was itself wrong

Reading an agent's report on stand-level off-block capture, I re-measured the
schedule-substitution rate and got 0.39% of departures against the 74% I had
been quoting. I told Lee the original number was wrong, retracted the ~313
ceiling that depended on it, and demoted `--sched-blend`.

That was a mistake. I measured **exact** equality; the original figure was
measured **within 60 seconds**. Different tolerances on the same quantity:

| population | n | ==0s | <=5s | <=60s |
|---|---|---|---|---|
| all departures | 2,085,047 | 0.39% | 4.26% | 9.57% |
| taxi > 3600s | 4,126 | 3.51% | 37.28% | 42.44% |
| taxi > 7200s | 584 | 7.88% | 67.64% | **73.97%** |
| no-plan, > 7200s | 481 | 9.36% | 75.88% | **82.54%** |

The original 74% and 82.5% were correct. **The retraction was the error, not the
claim.** The ~313 ceiling stands and `--sched-blend` goes back to the top of the
queue. Within 60s the median offset is 0s and the IQR is [-3, +3] — this is a
copied timestamp, not a coincidence, at 7.7x the 9.57% base rate.

## Correction 2: four features that cannot exist at serve time

Acting on the exact-equality reading I added `block_minus_sched`,
`block_is_sched`, `block_sec_00` and `stand_sub_rate`, on the reasoning that
where BLOCK == SCHED the target equals `gap_sched` exactly (verified, max error
0 across all 8,123 rows).

The arithmetic is right and the features are worthless. The target is
`MVT_TIME - BLOCK_TIME`, so the competition withholds BLOCK_TIME:

    ranking.parquet, 344,841 scored departures
      MVT_TIME_UTC_mvt      0 null
      SCHED_TIME_UTC_mvt    0 null
      BLOCK_TIME_UTC_mvt    344,841 null   (100.0%)

Every one of the four reads BLOCK. In training they compute; at serve time they
are null. A fold test would have rewarded all four enormously and the submission
would have read nulls. The identity `target = gap_sched - block_minus_sched`
holds for every row in the dataset and is useless, because the term that makes
it hold is exactly the term that is withheld.

This is why `fit_sched_blend` is built the way it is, which I had not
appreciated: it cannot *detect* substitution, so it **predicts** it — a
classifier over serve-available features, trained on `|y - gap_sched| <= 60`,
blending toward `gap_sched` by predicted probability. Checked: no leak.

Only `mvt_sec_00` survives. MVT_TIME is present at serve time and lands on :00
for 5.13% of rows against the 1.67% a sensor would give.

## The guard that should have existed

`prc/leakcheck.py`. Build the matrix on a training month and on `ranking.parquet`
and compare per-column null rates; anything more than 20 points emptier at serve
time is derived from a withheld column. Run before any fold test.

Confirmed it fires on the real bug before removal:

    4 feature(s) cannot be computed at serve time:
      block_is_sched         0.00%    100.00%
      block_minus_sched      0.00%    100.00%
      block_sec_00           0.00%    100.00%
      stand_sub_rate         0.00%    100.00%

and passes on the cleaned tree. It also clears every pre-existing feature, so
this defect was new today and not latent elsewhere. The 20-point tolerance
exists because `gap_aobt` and friends ride on `AOBT_3_flt`, genuinely absent for
1.5% of scored rows — that absence is the signal the no-plan model exists for.

`final.py` also gained `VALIDATED` and `_check_feature_registry()`, an opt-in
list that aborts the build on any feature that is in neither it nor `DROP`.
That addresses a different failure (four features previously shipped without a
fold test) and would **not** have caught this one. Leakage needs the null-rate
check; forgetting needs the registry.

## Findings that survive both corrections

- **Substitution clusters by stand.** LIRF stands with n >= 200 range 0.57% to
  9.1% around a 1.5% airport mean, low exactly on the high-throughput 400-series
  — the documented remote-vs-contact split, remote positions having no sensor.
  Not directly usable (the rate is computed from a withheld column) but it says
  `STAND_mvt`, which the model already has, carries real substitution signal.
- **It does not trend across 2025** (LIRF monthly .0157 .0155 .0171 .0165 .0160
  .0148 .0125 .0144 .0133 .0144 .0157 .0165), so the Assaia camera rollout at
  Fiumicino is not visible in this field. Killed.
- **`AOBT_3_flt` is minute-resolution** — 97.7% of its values land on :00 —
  which independently explains its 384s reconstruction error measured on
  2026-09-06.
- **Three airports report a purely sensed off-block time.** BLOCK lands on :00
  at ~8.3% for EGLL, LEMD, LIRF, EDDM, LTFM, LEBL and LFPG, but at the 1.67%
  uniform rate for LSZH, EDDF and EHAM.
