"""Paths and credential loading.

Nothing secret is stored in this repo. Credentials are read from files that
live outside the working tree; ``.gitignore`` blocks the filenames anyway as a
second line of defence.

Two *different* credentials are involved in this challenge, and they are easy
to confuse:

``OSN_CREDENTIALS``
    OpenSky Network OAuth2 API client credentials — a JSON file with
    ``clientId`` / ``clientSecret``, downloaded from your OSN account page.
    These authenticate against the OSN REST API (ADS-B state vectors, etc.),
    which is *supplementary* data for this challenge.

``PRC_BUCKET_CREDENTIALS``
    The MinIO/S3 access key + secret for the competition's own parquet bucket.
    These arrive by email once your team-creation request is approved. They are
    what you actually need to download the training data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("PRC_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
SUBMISSIONS_DIR = Path(os.environ.get("PRC_SUBMISSIONS_DIR", REPO_ROOT / "submissions"))

# Competition identifiers (see docs/COMPETITION-FACTS.md).
COMPETITION_ID = "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
DATACOMP_API = "https://datacomp.opensky-network.org/api"
OSN_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)

# Verified 2026-09-04 by listing the buckets the service account can see.
TEAM_NAME = os.environ.get("PRC_TEAM_NAME", "jolly-lobster")
S3_ENDPOINT = os.environ.get("PRC_S3_ENDPOINT", "https://s3.opensky-network.org")
DATASETS_BUCKET = os.environ.get("PRC_S3_BUCKET", "prc-2026-datasets")
TEAM_BUCKET = os.environ.get("PRC_S3_TEAM_BUCKET", f"prc-2026-{TEAM_NAME}")

_DEFAULT_OSN_CREDS = Path.home() / "Downloads" / "credentials.json"
# MinIO service-account JSON as downloaded from the S3 console.
_DEFAULT_BUCKET_CREDS = Path.home() / "Downloads" / "credentials(1).json"


@dataclass(frozen=True)
class OSNClientCredentials:
    """OAuth2 client-credentials pair for the OpenSky Network API."""

    client_id: str
    client_secret: str

    def __repr__(self) -> str:  # keep secrets out of tracebacks and logs
        return f"OSNClientCredentials(client_id={self.client_id[:4]}…, client_secret=<redacted>)"


@dataclass(frozen=True)
class BucketCredentials:
    """Access key pair for the competition MinIO/S3 bucket."""

    access_key: str
    secret_key: str
    endpoint: str
    bucket: str

    def __repr__(self) -> str:
        return (
            f"BucketCredentials(endpoint={self.endpoint!r}, bucket={self.bucket!r}, "
            f"access_key={self.access_key[:4]}…, secret_key=<redacted>)"
        )


def osn_credentials(path: str | Path | None = None) -> OSNClientCredentials:
    """Load OSN OAuth2 client credentials.

    Resolution order: explicit ``path`` → ``$OSN_CREDENTIALS`` → ``~/Downloads/credentials.json``.
    """
    candidate = Path(path or os.environ.get("OSN_CREDENTIALS") or _DEFAULT_OSN_CREDS)
    if not candidate.exists():
        raise FileNotFoundError(
            f"OSN credentials not found at {candidate}. Download credentials.json from "
            "your OpenSky Network account page, or set $OSN_CREDENTIALS to its location."
        )
    blob = json.loads(candidate.read_text(encoding="utf-8"))
    try:
        return OSNClientCredentials(blob["clientId"], blob["clientSecret"])
    except KeyError as exc:  # pragma: no cover - depends on what OSN hands out
        raise KeyError(
            f"{candidate} has keys {sorted(blob)}; expected 'clientId' and 'clientSecret'. "
            "If this file holds an access key / secret key pair instead, it is probably the "
            "competition bucket credential — point $PRC_BUCKET_CREDENTIALS at it."
        ) from exc


def bucket_credentials(path: str | Path | None = None) -> BucketCredentials:
    """Load competition bucket credentials.

    The file is the MinIO service-account JSON downloaded from
    ``s3-console.opensky-network.org``: ``accessKey`` (20 chars), ``secretKey``
    (40 chars), ``api``, ``path``, and a ``url`` field.

    Note that ``url`` is the *console* API endpoint, not the S3 endpoint, so it
    is deliberately ignored — S3 lives on ``s3.opensky-network.org`` (verified
    2026-09-04 by listing buckets against it).

    Resolution order: explicit ``path`` → ``$PRC_BUCKET_CREDENTIALS`` →
    ``~/Downloads/credentials(1).json``. Endpoint and bucket come from the
    module constants unless the JSON overrides them.
    """
    candidate = Path(path or os.environ.get("PRC_BUCKET_CREDENTIALS") or _DEFAULT_BUCKET_CREDS)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Bucket credentials not found at {candidate}. Download the service-account JSON "
            "from the S3 console, or set $PRC_BUCKET_CREDENTIALS to its location."
        )
    blob = json.loads(candidate.read_text(encoding="utf-8"))

    access = blob.get("accessKey") or blob.get("access_key") or os.environ.get("PRC_S3_ACCESS_KEY", "")
    secret = blob.get("secretKey") or blob.get("secret_key") or os.environ.get("PRC_S3_SECRET_KEY", "")
    if not (access and secret):
        raise RuntimeError(
            f"{candidate} has keys {sorted(blob)}; expected 'accessKey' and 'secretKey'. "
            "If this file holds clientId/clientSecret instead, it is the OSN OAuth "
            "credential — point $OSN_CREDENTIALS at it."
        )
    return BucketCredentials(
        access, secret, blob.get("endpoint") or S3_ENDPOINT, blob.get("bucket") or DATASETS_BUCKET
    )


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RAW_DIR, INTERIM_DIR, SUBMISSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
