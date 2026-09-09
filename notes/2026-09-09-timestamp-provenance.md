# Where the off-block timestamp comes from, and a number I had wrong

2026-09-09

Two agents came back with external material. One of them — a search for whether
anyone has documented the block-time artefact — turned up a mechanism worth
testing and, in testing it, caught a claim I had been repeating that is wrong.

## The correction first

I have been saying that **74% of multi-hour rows have BLOCK == SCHED to the
second**, and that **82.5% of no-plan rows over two hours** are that artefact.
Both numbers are wrong. They were never written into a note, only asserted in
conversation, which is how they survived. Measured properly over all 2,085,047
departure rows:

| population | n | BLOCK == SCHED |
|---|---|---|
| all departures | 2,085,047 | 0.39% |
| taxi > 3600s | 4,126 | 3.5% |
| taxi > 7200s | 584 | 7.9% |
| no-plan, taxi > 7200s | 481 | **9.4%** |

Not 74%. Not 82.5%. **9.4%.**

This matters beyond the embarrassment. The `--sched-blend` experiment was built
on the 74% figure, and the "computed ceiling ~313 on the board from a perfect
schedule-substitution fix" I quoted came from it. That ceiling is not real.
The substitution group is 8,123 rows carrying about 2.8% of squared error, not
the majority of the tail. `--sched-blend` drops down the queue accordingly.

The habit that failed here: a number that decides what to build next has to go
into a note where it can be re-derived. One that lives only in conversation gets
repeated until someone measures it again.

## The rollout hypothesis: dead

Assaia's camera-based off-block detection went live at Fiumicino in October 2024
across an initial 57 gates, "with the remaining gates being added as they become
available." If stands were converting from hand-reported to sensed off-block
times through 2025, the substitution rate at LIRF should fall across the year.

It does not. LIRF by month, 2025:

    1 .0157   2 .0155   3 .0171   4 .0165   5 .0160   6 .0148
    7 .0125   8 .0144   9 .0133  10 .0144  11 .0157  12 .0165

Flat. No trend. Whatever the rollout did, it is not visible in this field.
Killed.

## The stand hypothesis: alive

Remote stands are not sensor-equipped; off-block is relayed by a handler or a
pilot. Contact stands with a jet bridge can be captured directly. That predicts
the artefact clusters by stand, and it does — LIRF stands with n >= 200:

| stand group | rate |
|---|---|
| 101 | 9.1% |
| 234-238 | 3.5-4.5% |
| 820, 827, 828 | 3.3-4.2% |
| **402-412** | **0.57-0.72%** |

A 16x spread around a 1.5% airport mean, and the low group is exactly the
high-throughput 400-series (2,200-2,800 movements each) against low-throughput
200s and 800s. That is the remote/contact split showing up in the data.

## The seconds field says where a timestamp came from

A sensor reading has uniformly distributed seconds — 1/60 = 1.67% land on :00.
A time copied from a schedule or typed by a person piles up there. So the
seconds field is a provenance channel that costs nothing to read:

| field | P(sec == :00) | lift |
|---|---|---|
| BLOCK_TIME | 6.40% | 3.8x |
| MVT_TIME | 5.13% | 3.1x |
| AOBT_3 (Network Manager) | **97.70%** | 58.6x |

AOBT_3 is stored at minute resolution, which independently explains why it
could never reconstruct the label (offset sd 384s, measured 2026-09-06).

The airports split into two clean populations:

| P(BLOCK sec == :00) | airports |
|---|---|
| ~8.3% | EGLL, LEMD, LIRF, EDDM, LTFM, LEBL, LFPG |
| ~1.7% (= uniform) | LSZH, EDDF, EHAM |

Three airports report a purely sensed off-block time. Seven mix in a
minute-quantised one for roughly 6.7% of rows. And at LEMD and LEBL the two ends
of the target come from different systems: BLOCK rounds at 8.3% while MVT rounds
at 1.65%.

Rows with a :00 block time hold 8.61% of squared error while being 6.40% of rows
(1.35x over-represented), sd 621 against 541.

## What was built

Five features, none of which the model could previously derive:

- `block_minus_sched`, `block_is_sched` — BLOCK - SCHED was **not** a feature.
  When it is zero the target equals `gap_sched` exactly, verified on all 8,123
  rows at max error 0. The model already holds the answer in a feature it cannot
  locate.
- `block_sec_00`, `mvt_sec_00` — provenance and label quantisation.
- `stand_sub_rate` — per-stand substitution rate, shrunk toward the airport rate
  with a 200-row prior so a thin stand does not arrive as 0 or 1. Built from
  input columns only, so it carries no target information.

Honest sizing: against a per-airport-mean baseline, predicting every substituted
row perfectly moves RMSE 514.84 -> 507.65. Scaled to a model at ~280 the tail
rows are worth more in relative terms, plausibly -5 to -12s. This is not a
hundred-second finding and should not be described as one.

All five are in `final.py`'s DROP list and ship only via `--keep`.

## The guard

Four times a feature has gone into FEATURES, been left out of DROP, and
enrolled itself in the next submission unvalidated. DROP is opt-out, so
forgetting is the default and the failure is silent.

`prc/final.py` now carries `VALIDATED`, an explicit list of features that have
won a fold test, and `_check_feature_registry()` runs at import. A feature in
neither list aborts the build with a message naming it. Verified to fire.
