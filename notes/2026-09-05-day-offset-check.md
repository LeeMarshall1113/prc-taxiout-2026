# The day-offset hypothesis, tested and mostly rejected — 2026-09-05

Item 2 on the list was ADS-B forensics on the extreme rows, motivated by the
2024 winner having found timestamps in this data lineage unreliable enough to
rebuild from ADS-B. Before spending API calls, arithmetic settled most of it.

## What prompted it

Listing the 20 largest taxi-out values, off-block and wheels-up sit on
consecutive days at almost the same clock time:

```
2025-06-24 15:29:54  ->  2025-06-25 16:03:06   = 88,392s
2025-07-20 08:00:04  ->  2025-07-21 08:28:56   = 88,132s
2025-11-11 11:55:01  ->  2025-11-12 12:14:59   = 87,598s
```

That looks like an unambiguous one-day date error, and I said so.

## Tested properly, it does not generalise

Across all 76 departures at or above 20,000s in 2025:

| after removing the nearest whole number of days | rows |
|---|---:|
| residual 0–1h (a plausible taxi) | **14** |
| residual 1–4h | 0 |
| residual 4–24h | 42 |

Only **18.4%** survive. And the distance from a whole-day multiple,
`|y/86400 − round(y/86400)|`, has a median of **0.328** — *further* from an
integer than the 0.25 a uniform null would give. The pattern I saw was a
selection effect: the largest values are near 86,400s by construction, so
eyeballing the top of the list guarantees the appearance of a day offset.

## What is true

About 15 rows in all of 2025 are genuine one-day errors. Their residuals have a
median of 951s against a global median taxi of 912s, which is convincing for
those rows specifically. Every one is at LIRF or LSZH and 13 of 14 have no
flight plan, consistent with everything else we know.

The remaining ~60 extreme rows sit between 20,000s and 64,000s and are not
day-offset artefacts. What they are remains unknown, and the literature says
there is no way to tell from schedule data alone.

## Why item 2 is dropped

Correcting every identifiable day error moves the LIRF/no-flight-plan group mean
from 6,531s to 5,834s — 11%. The model already predicts that group at 6,879s
against a true 6,272s, so it is not mispredicting the group in the first place.
Chasing the remaining rows through the OpenSky API, at roughly 60 flights with
no registration in the schema to match on, cannot pay for itself.

## The one cheap thing worth trying instead

We tested *capping* training labels (winsorising) and it improved the bulk while
wrecking plain RMSE, because a capped model cannot predict above its cap.
**Dropping** the 76 rows from training is a different operation: it removes the
gradient distortion — each carries on the order of a thousand times an ordinary
row's weight under squared loss — without limiting what the model can output.
Worth one cross-validated run.
