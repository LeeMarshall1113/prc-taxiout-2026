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

**2. Competition bucket keys** — MinIO service-account JSON from
`s3-console.opensky-network.org`: `accessKey` (20 chars), `secretKey` (40),
`api: s3v4`, `path: auto`, and a `url` field.

- Status: **issued 2026-09-04 and verified working.**
- Location: `~/Downloads/credentials(1).json`. Override with
  `$PRC_BUCKET_CREDENTIALS`. That filename is fragile — a second download becomes
  `credentials(2).json`, and Downloads gets cleaned — so move it somewhere stable
  and set the env var.
- **The `url` field is a decoy.** It points at the console API
  (`https://s3-console.opensky-network.org/api/v1/service-account-credentials`),
  not the S3 endpoint. S3 is on `https://s3.opensky-network.org`; `prc/config.py`
  ignores `url` deliberately.
- Grants access to two buckets: `prc-2026-datasets` (read, the shared data) and
  `prc-2026-jolly-lobster` (our team bucket, for submissions).

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
