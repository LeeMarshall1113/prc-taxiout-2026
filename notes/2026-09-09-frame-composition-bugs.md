# Two live train/serve skews, and the guard that finds them mechanically

2026-09-09

An adversarial audit of the pipeline came back with eight findings. Four were
real, and two of those were live in the shipping model. All numbers below were
re-derived here rather than taken from the report.

## The root cause both share

`build()` is called **once per monthly file** in training, and **once over the
whole of `ranking.parquet`** at serve time. The ranking file is January *and*
July 2026 in one frame -- 62 days, five months apart:

    ranking.parquet   2026-01-01 .. 2026-07-31   62 distinct dates

So any feature computed over "whatever rows this frame happens to hold" runs a
different computation in the two cases. The fold rig cannot see it: crossval
builds from monthly files too, so it reproduces the *training* behaviour on both
sides of every split.

That blind spot now accounts for three separate bugs: `minute_of_day`
(2026-09-08) and the two below.

## Live bug 1: stand_runway_pair_n, 97.96% of scored rows

Already "fixed" once -- a raw count became a per-day rate, precisely to survive
the one-month-vs-two-month difference. The rate fixed the scale and left the
grouping alone, so `n_days` was still 31 in training and 62 at serve.

Built `ranking.parquet` whole, then built its two months separately and diffed
per row:

| | before | after |
|---|---|---|
| rows matching a single-month build | 3.26% | **100.00%** |
| rows served exactly half | 13,446 | 0 |
| mean absolute error | 0.45 on a base of 2.0 | 0.000000 |

Every key present in only one of the two months was served exactly half its
trained value, and the rest were pulled toward the other month's rate. This was
**shipping in every submission to date**.

Fixed by grouping on calendar month as well as key, so a training month and a
serve month produce the same number.

## Live bug 2: _sequence_proxy had no lookback cap

`_wave2` caps its previous-movement search at `_MAX_HEADWAY_S = 6h`, with a
comment naming this exact hazard. `_sequence_proxy` reimplements the same
pattern inline and did not.

    prev_stand_headway on ranking.parquet, before:  max 17,600,670s = 203.7 days
                                            after:  max     21,600s =   6.00 h
    rows > 31 days (impossible in any training month), before: 2,154   after: 0

July's first departure from a stand was inheriting January's gap, from a flight
five months earlier, along with a headway no training row can approach. Presence
rates now agree: 63.6% in a training month, 64.4% at serve.

These three features sat in `DROP`, so nothing shipped -- but they were queued
for a fold test that would have passed and promoted them.

## Fixed: final.py --keep could not restore a categorical

`feats` filtered on `DROP` minus `keep`; `cat_idx` filtered on `DROP` alone. A
kept categorical entered the matrix without being declared categorical, and
CatBoost died trying to read `LSZH|1` as a float -- i.e. `--keep airport_plan`,
the feature this module's docstring is written around, could never ship.

Both now derive from one `dropped` set, as `crossval.py` has always done. This
is the **fourth** time the two paths have disagreed about how to ship a
validated result. Also added `--loss`, `--l2`, `--one-hot-max-size` and
`--drop-dayoffset`, all of which crossval could measure and final could not ship.

## Rejected: _runway_config self-inclusion

The audit flagged `rwy_dep_active`/`rwy_dep_share` counting the query row itself,
against sibling features that explicitly exclude self. True, but it is not a
bug: the same function runs on the training and the ranking frame, so it is a
consistent definition, not a skew. The model learns the feature as defined. Only
the ~395 rows reporting "1 active runway, 100% own share" are questionable, and
that is a rounding curiosity, not a defect worth a fold test.

## Rejected: the crossval.py docstring is stale

The audit read commit `9723a5e` -- my own erroneous retraction of the 74% figure
-- and concluded that the premise of `--sched-blend` was disproven. It was not;
see `2026-09-09-timestamp-provenance.md`. The docstring is correct. Worth
recording that a wrong commit message propagated into an independent audit
within hours.

## The guard

`prc/leakcheck.py` now runs two checks:

1. **Availability.** Null rates per feature, training build vs ranking build.
   Catches anything derived from a withheld column.
2. **Framing.** Build `ranking.parquet` whole, build it split by month, diff per
   row. Same rows, same calendar, same year, so every difference is frame
   composition alone -- no drift false positives.

Verified by reintroducing both bugs and re-running:

    4 feature(s) change value when the frame is split by month:
      feature                   rows differ    worst delta        scale
      prev_stand_gap                  0.62%      4,751.000     86,340.0
      prev_stand_headway              0.62% 17,600,670.000  2,422,069.0
      prev_rwy_gap                    0.02%      1,856.000     86,340.0
      stand_runway_pair_n            97.96%          2.435         10.7

then restored (md5 identical) and confirmed clean. An earlier version compared
training-month ranges against serve ranges and flagged `doy`, `month`,
`gap_aobt`, `gap_eobt`, `gap_lobt` -- all false positives from comparing one
month against two. Diffing the ranking file against itself removes that
entirely.

Run before any fold test.
