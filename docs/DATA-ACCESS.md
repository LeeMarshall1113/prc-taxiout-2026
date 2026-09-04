# Data access

## Two credentials, easy to confuse

**1. OSN OAuth2 API client** — `clientId` + `clientSecret` in a `credentials.json`
downloaded from your OpenSky Network account page.

- Status: **in hand and verified working** (2026-09-04). The token endpoint
  returns a 30-minute `Bearer` token, scope `profile email`.
- Location: `~/Downloads/credentials.json`. Override with `$OSN_CREDENTIALS`.
- What it is for: the OSN REST API — live and historical ADS-B state vectors.
  That is *supplementary* data for this challenge, not the competition dataset.
- Downloads gets cleaned out. Worth moving to somewhere stable, e.g.
  `~/.config/opensky/credentials.json`, and setting `$OSN_CREDENTIALS` to match.
  **Do not move it into this repo** — `.gitignore` blocks the filename, but the
  repo has to go public under GPLv3 at the end, so keep secrets out of the tree
  entirely.

**2. Competition bucket keys** — MinIO/S3 access key + secret for the challenge's
own parquet bucket.

- Status: **not yet issued.** They arrive by email once the team-creation request
  is approved.
- On arrival: save the JSON outside the repo and export
  `$PRC_BUCKET_CREDENTIALS=/path/to/it`, or export `PRC_S3_ACCESS_KEY`,
  `PRC_S3_SECRET_KEY`, `PRC_S3_ENDPOINT`, `PRC_S3_BUCKET` directly. The endpoint
  host and bucket name come with the email — `prc/config.py` guesses
  `https://s3.opensky-network.org` and an empty bucket name, so both need
  confirming.

Neither credential is ever printed in full by this codebase; the dataclasses in
`prc/config.py` have `__repr__`s that redact the secret halves so a stray
traceback cannot leak them.

## One caveat about OSN tiers

OSN gates its *general* historical archive (Trino/S3) to university-affiliated
researchers, government organisations, and aviation authorities; private and
commercial users need a licence. That gate should not apply here — the
competition distributes its own bucket via the approval email rather than the
research archive — but it is the thing that could bite an independent entrant.
If the approval stalls, that is the likely reason, and it is a support-email
problem rather than a hard block.

## First run once the keys land

```bash
export PRC_BUCKET_CREDENTIALS=~/.config/prc/bucket.json
python -m prc.data list                 # what is actually in the bucket
python -m prc.data pull                 # download to data/raw/
python -m prc.schema data/raw           # columns, dtypes, nulls, ranges
```

`prc.schema` is the gate on everything downstream. No feature code has been
written yet on purpose — the column list in `docs/COMPETITION-FACTS.md` came off
a web page, not off the files, and building a feature pipeline against guessed
column names is how you spend two days debugging a rename.

## Python environment

`.venv/` here, Python 3.14.0. Every wheel we need built without trouble
(2026-09-04): catboost 1.2.10, lightgbm 4.7.0, pandas 3.0.5, polars 1.44.1,
pyarrow 25.0.1, duckdb 1.5.5, scikit-learn 1.9.0, boto3 1.43.88. No 3.14 wheel
gaps, so there is no reason to drop to an older interpreter.

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
