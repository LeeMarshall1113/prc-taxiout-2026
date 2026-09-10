# PRC Data Challenge 2026 — verified facts

Two tiers below. **VERIFIED** means pulled directly from an API or tested against
the live service on the date given. **REPORTED** means read off the challenge
website; re-confirm against the Terms & Conditions before anything depends on it.

Site: <https://prc-data-challenge-2026.netlify.app/>
(`https://ansperformance.eu/study/data-challenge/dc2026/` 301s here.)

## Task

Predict **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for departures at
**10** airports — LTFM Istanbul, LFPG Paris CDG, EGLL Heathrow, EHAM Amsterdam,
LEMD Madrid, EDDF Frankfurt, LEBL Barcelona, LIRF Rome, EDDM Munich, LSZH
Zürich. VERIFIED 2026-09-04 from `ranking.parquet` — exactly 10 distinct
`ADEP_mvt` values among departures. The site's data page says 11 and is wrong;
its overview page says 10 and is right.

Train on full-year 2025 movements. Predict Jan + Jul 2026. REPORTED.

## Metric and ranking

- **RMSE in seconds**, lower is better. VERIFIED — scores come back on that scale.
- Teams are ranked on their **best-ever submission**, not their last. REPORTED.
- **The evaluation set was reissued mid-competition on 2026-09-04**, between
  08:26Z and 09:58Z: `usedPairs` went 215,876 -> **344,841**, matching the
  reissued `ranking.parquet`. Scores across that boundary are **not comparable**;
  707 of 734 submissions on the board are on the superseded set. VERIFIED.
  `prc.leaderboard` filters to one set and names which is in force.
- **There is no private split.** The ranking set *is* the public leaderboard set.
- **There IS a daily submission cap.** 3/day from 2026-09-03, raised to **5/day**
  on 2026-09-07. Over the limit the result file returns
  `DAILY_LIMIT_REACHED: n of n used today, resets at 00:00 UTC`. VERIFIED from
  the organisers' Discord; supersedes the earlier "submissions appear unlimited"
  reading, which was drawn from one team's 542-submission burst before the cap
  existed. That burst was a script bug, acknowledged by its author.
  The cap was introduced because RMSE feedback is **invertible**: for a single
  changed row, `N(MSE' - MSE) = d^2 + 2d(p_k - y_k)` solves for the true label,
  so one extra submission recovers one ground-truth value and a binary search
  finds the highest-leverage rows in a few dozen queries. We do not do this and
  should not; it is named here so nobody reinvents it thinking it is clever.

## Timeline

- Opens **2026-09-01**; closes **2026-10-11, 23:59:59 CET**. REPORTED.
- No registration cutoff stated anywhere on the site. **UNVERIFIED — check the
  T&Cs.**

## Prize and obligations

- **5000 EUR combined across the top 3 teams.** Split not published. REPORTED.
- Prize eligibility requires: source code public on GitHub under **GNU GPLv3**;
  documentation sufficient to reproduce; any external datasets openly
  accessible and documented; solution original (reusing an existing
  implementation needs rights + significant modification). REPORTED.
- **External data is explicitly permitted** if openly licensed. Weather/METAR and
  ATFM regulation feeds are therefore in play. REPORTED.
- Open-access paper via the *Journal of Open Aviation Science* is encouraged, not
  required. REPORTED.

## Data

VERIFIED 2026-09-04 by listing the bucket. **The website's figures are wrong** —
it says ~283 MB of training, a 27 MB ranking file and a 1.1 MB template. Real
sizes below; the ranking file is 61% bigger than advertised.

Endpoint `https://s3.opensky-network.org`, bucket **`prc-2026-datasets`**,
14 objects, **330.2 MB**:

| object | size |
|---|---|
| `training_2025-<mm>-01_2025-<mm+1>-01.parquet` × 12 | 19.8 – 26.0 MB each, 285.0 MB total |
| `ranking.parquet` | 43.6 MB |
| `submitting.parquet` | 1.7 MB |

Training files are dated 2026-08-13; `ranking.parquet` and `submitting.parquet`
were rewritten 2026-09-04 — worth re-checking their checksums before the close in
case the organisers reissue them.

The team also gets a private bucket, **`prc-2026-jolly-lobster`**, empty on
issue. That is where submissions are uploaded.

Military, Head of State, and Sensitive movements are excluded. REPORTED.

## Submission format

Filename **`<team-name>_v<incremental integer>.parquet`**. Every `MVT_ID_mvt` in
the template must be present exactly once, no additions, no omissions —
mismatches are rejected. REPORTED; `prc.submit` enforces it locally.

## API endpoints

- Competition id: `bb3693e1-26bc-4a9e-8619-4fe78b4eab0c` VERIFIED
- Leaderboard: `https://datacomp.opensky-network.org/api/competitions/<id>/leaderboard`
  — no auth, cursor-paginated 50 at a time. VERIFIED 2026-09-04.
- Swagger: `https://datacomp.opensky-network.org/api/swagger-ui/`
- OSN OAuth2 token endpoint (Keycloak, `client_credentials` grant, 30-min
  bearer): `https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token`
  VERIFIED 2026-09-04.
- Announcements go to the OSN Discord `#prc-data-competition`; the site's own
  leaderboard widget has been **broken since 2026-09-01** (Observable framework
  migration), which is why we read the API directly.

## Our team

**`jolly-lobster`** — approved 2026-09-04 (team bucket created 18:30 UTC).
Submissions must therefore be named `jolly-lobster_v<N>.parquet`.

## Field size

- **2024 edition** (takeoff weight): 369,013 flights, **132 teams / 261
  individuals**. REPORTED from the challenge overview paper.
- **2026 edition, as of 2026-09-04**: 256 teams registered, but only **37 with a
  scored submission**. VERIFIED. See `notes/2026-09-04-field-snapshot.md`.

## From the organisers' Discord (`#prc-data-competition`, OpenSky Community)

Exported 2026-09-10, 522 messages back to the channel's creation on 2024-11-08.
`espinielli` is Enrico Spinielli (EUROCONTROL PRC, the organiser); `john_fitz_nz`
runs the submission infrastructure. Everything below is quoted or paraphrased
from them unless a participant is named.

### Rulings that decide what is allowed

- **Feature engineering from other movement rows is explicitly permitted.**
  Asked directly whether a departure at time T may use trailing counts of
  earlier movements: *"feature engineering is of course permitted, do whatever
  makes sense to you with the provided data. Your model could take into account
  what is occupied, what is coming..."* (2026-09-09). This clears our congestion
  and sequence features.
- **METAR is fine**; open-source data are allowed "if declared in the
  documentation for the solution/repo" (2026-09-08).
- **OSN state vectors are discouraged**, because "our ground truth is from
  airport reported timestamps and not derived from on-board means, so
  discrepancies with what flown would be there" (2026-09-03). **OPDI** is
  permitted but "milestones for now do not cover off-block" (2026-09-06).
- **No separate final phase planned**, "but we reserve the right to go for it."

### The block-time artefact is public, and the organisers will not fix it

Raised by `henri_59479` on **2026-09-03**, seven days before we thought we had
found it independently. At least five teams have since posted measurements.
The organisers' position:

- *"Block time is what the airport provided: it should be filled with actual
  because for SCHEDULE there is SCHED_TIME_UTC_mvt"* (2026-09-08).
- *"We exported what we got without substitutions"* (2026-09-08) — so the
  substitution happens upstream, in the airport's own reporting, not in the
  export.
- *"I do not know of recurring operational reasons for those outliers, but they
  are there. Some could be linked to some specific events, others could just be
  messy data from airport or NM"* (2026-09-08).
- *"The decision about how to deal with strange/noisy/messy data is full part of
  the challenge"* (2026-09-09). The rows stay, and they are scored.

**Consequence for us:** this was never an edge. It explains how teams reached
271-279 in three or four submissions.

### `IOBT` / `EOBT` / `LOBT`, from the organiser

*"IOBT is calculated from the flight plan ('I' = initial), EOBT is calculated
from any messages/updates that have come after ('E' = estimated), and LOBT is
the latest calculated value for Off-block ('L' = last). These come from
operations: no post-ops adjustments."* And `AOBT_3_flt` is *"what NM knows from
the flight once flown"*. `_mvt` columns are airport-reported and
"~validated by EUROCONTROL"; `_flt` columns are Network Manager.

### Why the evaluation set changed on 2026-09-04

*"The submit dataset is incorrect for July: my fault... airports didn't report
yet when I initially extracted the data"* (2026-09-03). July initially contained
only EDDF, EGLL and EHAM. Reissued 2026-09-04 ~08:30Z with all ten, and **old
submissions were removed from the ranking**. That is the 215,876 -> 344,841 jump.

`javieriom` posted the calibration we never had: the same approach scored
**278.38 on the old set and 326.60 on the new**, model unchanged.

### The finding we did not have: `roma_no`'s 630 "neither" rows

Posted 2026-09-08 and **reproduced here exactly**, every figure:

On the Jan+Jul 2025 fold (344,419 departures) there are 981 rows with taxi over
an hour. They split three ways:

| class | n | |
|---|---|---|
| `BLOCK == SCHED` within 30s | 345 | the known artefact |
| `+1 day` rollover | 6 | the known artefact |
| **neither** | **630** | **not previously examined by us** |

558 of the 630 carry an `AOBT_3_flt`. Their median target is **4,202s** while
the median `MVT_TIME - AOBT_3_flt` is **1,264s** — the Network Manager sees an
ordinary taxi where the airport reports over an hour. Median gap 3,055s, and
only **8.6%** agree within 300s.

**They hold 16.6% of the fold's squared error**, and unlike everything else we
have chased they are *not* a Rome story: EGLL 234, LFPG 133, LIRF 122, EHAM 54.

Measured here across all of 2025 (2,046 such rows), two structures stand out:

- **Hour of day.** Rate per 10k departures is **107.4 at 00h** against ~10 for
  the rest of the day, and 2.0 at 04h. A tenfold midnight spike.
- **Airport.** EGLL 27.5, LTFM 20.6, LIRF 17.4, LFPG 14.2 per 10k, against
  LEMD 0.0, LEBL 0.1, LSZH 1.5. A local reporting convention, not a fleet or
  weather effect.
- Month is mixed: February 33.0 and July 17.0 both elevated, March-May at
  3.2-5.5. Not de-icing, or not only de-icing.

`roma_no`'s hypotheses, unanswered by the organisers: return-to-stand, remote
de-icing stamped as off-block, or a local convention on when the block event is
recorded.

**Why this matters for the model:** at serve time we see `gap_aobt` of ~1,264s
and predict accordingly, while the truth is ~4,202s. The correction available is
a posterior shift of `p * 2,832s` where `p` is the airport-by-hour rate — which
is exactly the kind of interaction an oblivious tree must spend a whole level on.
It is an independent argument for the `grow_policy` test already running.
