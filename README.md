# prc-taxiout-2026

Entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict taxi-out time, in seconds, for departures at 11 major European airports.
Train on 2025 movements, predict January and July 2026, ranked on RMSE.

**Closes 2026-10-11, 23:59:59 CET.**

Verified competition facts, with sources and a verified/reported split, live in
[docs/COMPETITION-FACTS.md](docs/COMPETITION-FACTS.md). Credential handling is in
[docs/DATA-ACCESS.md](docs/DATA-ACCESS.md).

## Status

Data is in hand (`data/raw`, 330 MB, not tracked). Team is **`jolly-lobster`**.
First-contact findings are in
[notes/2026-09-04-data-findings.md](notes/2026-09-04-data-findings.md) — read
that before modelling; it covers the mid-competition evaluation-set change, the
fact that the target is exactly `MVT_TIME - BLOCK_TIME` in training, and that
squared error is dominated by a few dozen tail rows.

No model yet.

Working and tested:

- `prc.leaderboard` — reads the field straight off the ranking REST API. The
  site's own leaderboard widget has been broken since 2026-09-01, so this is the
  only way to see where the field is.
- `prc.osn` — OAuth2 bearer tokens for the OpenSky Network API.
- `prc.submit` — enforces every submission rule locally before an upload.
- `prc.validate` — the metric and the holdout scheme.

Written but untested, because they need credentials or data:

- `prc.data` — bucket listing and download.
- `prc.schema` — parquet profiling, to run on first contact with the files.

Deliberately **not** written yet: features and training. The column names in
`COMPETITION-FACTS.md` came off a web page, not off the parquet. Building a
feature pipeline against guessed column names is how you lose two days to a
rename. `prc.schema` runs first; feature code follows the real schema.

## Quick start

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
python -m prc.leaderboard              # where the field is right now
python -m prc.osn                      # check OSN credentials still work
```

Once the approval email lands:

```bash
export PRC_TEAM_NAME=<your-team-name>
export PRC_BUCKET_CREDENTIALS=~/.config/prc/bucket.json
python -m prc.data list
python -m prc.data pull
python -m prc.schema data/raw
```

## Validation

The ranking set is January + July 2026 — one deep-winter month and one
peak-summer month. We hold out **January and July 2025** and train on the other
ten, so the local split has the same seasonal composition as the target. It
cannot reproduce the 2025 → 2026 year gap; expect leaderboard scores worse than
local, and treat the gap as an offset rather than trying to correct it. See
`prc/validate.py`.

## House rules

- **Secrets never enter this tree.** The repo has to go public under GPLv3 to be
  prize-eligible. Credentials load from paths outside the working tree and the
  credential dataclasses redact their secret halves in `__repr__`.
- **The compute broker owns the machine.** Anything holding >4 GB RAM or running
  >15 min takes a lease first — training runs qualify, data pulls do not:

  ```bash
  /mnt/c/Users/hackathon/.compute-broker/lease-run.sh \
      --project prc-taxiout --min-gb 6 --minutes 120 \
      --why "catboost taxi-out training" -- python -m prc.train
  ```

  This is CPU-only tabular work, so it does not contend for the RX 9070 XT that
  arc-agi-2 needs. Cap thread counts anyway; the box is 8C/16T and other agents
  are on it.

## Licence and visibility

GPLv3, as required for prize eligibility. See [LICENSE](LICENSE).

**This repo is private and must be public before 2026-10-11 to be
prize-eligible.** It starts private so the approach is not visible to the rest
of the field during the 37 days the competition runs; the rules only require the
code to be open at the end. Flipping it is one command:

```bash
gh repo edit LeeMarshall1113/prc-taxiout-2026 --visibility public --accept-visibility-change-consequences
```

Before flipping, re-run the secret scan — the tree must stay clean, since a
public push cannot be taken back:

```bash
git grep -nIE '(clientSecret|secretKey|aws_secret|BEGIN [A-Z ]*PRIVATE KEY)' -- ':!docs' ':!README.md'
```
