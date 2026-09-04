# PRC Data Challenge 2026 — verified facts

Two tiers below. **VERIFIED** means pulled directly from an API or tested against
the live service on the date given. **REPORTED** means read off the challenge
website; re-confirm against the Terms & Conditions before anything depends on it.

Site: <https://prc-data-challenge-2026.netlify.app/>
(`https://ansperformance.eu/study/data-challenge/dc2026/` 301s here.)

## Task

Predict **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for departures at 11
major European airports — Frankfurt, Munich, London Heathrow, Amsterdam,
Barcelona, Madrid, Paris CDG, Rome, Istanbul, Zürich (+1; the overview page says
"10", the data page lists 11 — resolve against the actual parquet). REPORTED.

Train on full-year 2025 movements. Predict Jan + Jul 2026. REPORTED.

## Metric and ranking

- **RMSE in seconds**, lower is better. VERIFIED — scores come back on that scale.
- Teams are ranked on their **best-ever submission**, not their last. REPORTED.
- **`usedPairs` = 215,876** on every scored submission so far — the evaluation set
  size. VERIFIED 2026-09-04 via the leaderboard API.
- **There is no private split.** The ranking set *is* the public leaderboard set.
  Submissions appear unlimited: one team logged 542 in the first four days.
  VERIFIED 2026-09-04. Consequence: the metric rewards leaderboard grinding, and
  nothing external protects against fitting the ranking set.

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
