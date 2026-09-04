"""OpenSky Network OAuth2 client.

OSN issues short-lived bearer tokens (30 min) from a Keycloak realm via the
``client_credentials`` grant. Verified working 2026-09-04.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import json

from .config import OSN_TOKEN_URL, OSNClientCredentials, osn_credentials

_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
# Refresh a little before expiry so a long-running job never uses a dead token.
_REFRESH_MARGIN_S = 60.0


def access_token(creds: OSNClientCredentials | None = None) -> str:
    """Return a cached bearer token, fetching a new one when it is close to expiry."""
    creds = creds or osn_credentials()
    cached = _TOKEN_CACHE.get(creds.client_id)
    if cached and cached[1] - _REFRESH_MARGIN_S > time.time():
        return cached[0]

    body = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
        }
    ).encode()
    request = urllib.request.Request(OSN_TOKEN_URL, data=body)
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.load(response)

    token = payload["access_token"]
    _TOKEN_CACHE[creds.client_id] = (token, time.time() + float(payload.get("expires_in", 1800)))
    return token


def auth_header(creds: OSNClientCredentials | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token(creds)}"}


if __name__ == "__main__":
    # Smoke test that prints nothing secret.
    token = access_token()
    print(f"OSN auth OK — bearer token acquired (len={len(token)}, redacted)")
