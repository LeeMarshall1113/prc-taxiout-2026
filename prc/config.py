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

# Set this once your team is approved; it drives submission filenames.
TEAM_NAME = os.environ.get("PRC_TEAM_NAME", "")

_DEFAULT_OSN_CREDS = Path.home() / "Downloads" / "credentials.json"


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

    Reads ``$PRC_BUCKET_CREDENTIALS`` (a JSON file) or the individual environment
    variables ``PRC_S3_ACCESS_KEY`` / ``PRC_S3_SECRET_KEY`` / ``PRC_S3_ENDPOINT`` /
    ``PRC_S3_BUCKET``.

    NOT YET WIRED UP: the exact endpoint host and bucket name come with the
    approval email. Fill them in there, or via the env vars, on arrival.
    """
    candidate = path or os.environ.get("PRC_BUCKET_CREDENTIALS")
    if candidate:
        blob = json.loads(Path(candidate).read_text(encoding="utf-8"))
    else:
        blob = {}

    def pick(*names: str, default: str = "") -> str:
        for name in names:
            if blob.get(name):
                return str(blob[name])
            if os.environ.get(name.upper()):
                return os.environ[name.upper()]
        return default

    access = pick("accessKey", "access_key", "PRC_S3_ACCESS_KEY")
    secret = pick("secretKey", "secret_key", "PRC_S3_SECRET_KEY")
    endpoint = pick("endpoint", "PRC_S3_ENDPOINT", default="https://s3.opensky-network.org")
    bucket = pick("bucket", "PRC_S3_BUCKET", default="")

    if not (access and secret):
        raise RuntimeError(
            "No competition bucket credentials found. They arrive by email after the "
            "team-creation request is approved. Set $PRC_BUCKET_CREDENTIALS to the JSON "
            "file, or export PRC_S3_ACCESS_KEY / PRC_S3_SECRET_KEY / PRC_S3_ENDPOINT / "
            "PRC_S3_BUCKET."
        )
    return BucketCredentials(access, secret, endpoint, bucket)


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RAW_DIR, INTERIM_DIR, SUBMISSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
