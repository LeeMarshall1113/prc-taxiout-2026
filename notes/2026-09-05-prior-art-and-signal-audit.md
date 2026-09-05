# Prior art, literature, and a signal audit — 2026-09-05

Three parallel investigations. They converge from three directions, and most of
what they found is negative, which is the useful part.

## 1. Our bulk model is at the published ceiling

The closest published analogue — NASA Ames at Charlotte, schedule and movement
metadata only, no surface surveillance — reports **288-330s RMSE**. We are at
**278s** on rows with a flight plan.

Every study reaching 76-125s uses real-time surface-trajectory data (ASDE-X /
A-SMGCS): per-taxiway aircraft counts, occupancy probabilities, holding
decomposed by location. That data is not in our columns and cannot be
reconstructed from off-block and wheels-up timestamps.

Benchmark ladder, Lim et al. (SESAR 2021), Atlanta, outliers pre-filtered:

| approach | RMSE |
|---|---:|
| naive mean | 216.4s |
| EUROCONTROL PRU 10th-percentile reference | 199.2s |
| FAA APO methodology | 123.1s |
| gradient boosting (with surface flow) | 78.4s |
| graph neural network (with surface flow) | 76.0s |

Generic bulk feature engineering has little left to give. Independent
confirmation of what our own decomposition said.

## 2. There is no unused signal in the columns

Every column we do not use, measured against our model's holdout residuals:

| candidate | verdict |
|---|---|
| FLIGHT_ID_mvt null | **dead - it is has_flight_plan twice.** Differs by exactly 7 rows in training and exactly 7 in ranking.parquet |
| ADES_FILED_flt vs ADES_flt | **dead.** 0.11% coverage, corr with residual +0.0002, ~0.00% of squared error. Causally inert: a diversion is decided in flight, after this leg's taxi-out |
| CALLSIGN_flt structure | **dead.** The prefix maps almost bijectively onto AIRCRAFT_OPERATOR_flt - 494 of 497 operator hashes map to exactly one prefix |
| IOBT_flt | **dead.** Equals LOBT_flt in 99.81% of rows |
| planning-revision spread | marginal. Trimmed-SE corr +0.076; strongest cut correlates 0.85 with existing dep_delay, dropping to +0.06 incremental. Ceiling ~0.5-1% of squared error |
| ARVT_1 vs ARVT_3 spread | highest measured (trimmed-SE corr +0.14 to +0.16), but both sit ~3h **after** wheels-up - this flight's own arrival record, not a pre-departure estimate. Present in ranking.parquet so it is permitted; check redundancy first |
| turnaround from previous movement | 71% coverage via stand-matching, correlation +0.030. Not worth the join |

**No aircraft registration exists anywhere in the schema**, and callsigns cannot
substitute: among stand-matched arrival/departure pairs agreeing on aircraft
type and operator - strong evidence of a genuine turnaround - the callsigns
agree only **1.7%** of the time. Callsigns are per-leg designators, not
per-airframe. Airframe-level history is ruled out.

## 3. The 2024 challenge does not transfer, and says so clearly

Prize eligibility in 2024 required publishing code, so ~29 team repos are public.

- **The 2024 split was random, not temporal.** Confirmed in the organisers'
  paper and the winner's code. Their validation agreed with their leaderboard
  because train and test were drawn from the same 2022 flights. Copying it would
  be actively wrong for us.
- **A top-3 team documented our exact failure mode.** team_brave_pillow trained
  per-aircraft-type models and selected each by validation RMSE: 2,066 in
  validation, 2,276 on the final set. Independent corroboration of our own
  per-hour dead end.
- Gradient boosting dominated; neural approaches ran ~75% worse. **Seed-bagging
  a single model beat a five-model ensemble** - the winner went 1,612 to 1,562
  kg over 20 seeds, about 3%.
- **No 2024 team faced anything shaped like our problem.** The nearest analogue,
  rare aircraft types, went unsolved even by the winners.

## 4. Weather: the two investigations disagree, and the negative one is on point

The 2024 winner used METAR (Iowa Mesonet ASOS, free, covers all ten airports)
and it helped - but for **takeoff weight**, where it ranked *last* in their
ablation at +1.8%.

The SESAR graph-neural-network study built ATMAP weather scores from TAF/METAR
**specifically for taxi-out** and found weather non-predictive, dropping it.
That is the study actually about our task.

METAR moves from "top lead" to "test, do not assume". January and July being the
de-icing and convective months keeps it worth one experiment.

## 5. What is left

1. **EUROCONTROL's own reference taxi time.** Their published recipe: the 10th
   percentile of taxi-out per (stand, runway) over a rolling 12 months, used
   only where at least 10 flights fall at or below it, backing off to runway
   then airport level when sparse. A real unimpeded-taxi-distance proxy - the
   most-cited dominant feature in the literature - where we currently use a
   crude movement count. Cheap, principled, built from columns we have.
2. **OSM taxiway shortest-path distance** per stand to runway. The genuine
   version of the same idea. ODbL, ten airports, real one-time effort.
3. **Seed-bagging.** Mechanical ~3%, no new data.
4. **ADS-B forensics on the extreme rows.** The 2024 winner found timestamps in
   this exact data lineage unreliable enough to rebuild from ADS-B 10 NM
   crossings. We have working OpenSky credentials and only a few hundred rows to
   check. The only route to a question the literature admits it cannot answer:
   whether a given multi-hour taxi is genuine or an artefact.

Calibration on that last point: EUROCONTROL's own indicator **discards any
taxi-out over 120 minutes (7,200s)**, and our p99.9 is 4,501s. Most of our tail
is what the organisers themselves count as genuine severe queuing; only the 584
rows above 7,200s fall outside their convention.
