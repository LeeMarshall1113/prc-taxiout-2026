"""Download the competition parquet from the MinIO/S3 bucket.

Untested end to end — the bucket credentials are not issued until the team is
approved, so ``list`` is the first thing to run when they land. It will tell us
the real object names, which the challenge page only describes in prose.

    python -m prc.data list
    python -m prc.data pull
    python -m prc.data pull --prefix 2025-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import RAW_DIR, bucket_credentials, ensure_dirs


def _client():
    import boto3
    from botocore.config import Config

    creds = bucket_credentials()
    if not creds.bucket:
        raise RuntimeError(
            "Bucket name unknown. It comes with the approval email — set $PRC_S3_BUCKET "
            "or put 'bucket' in the credentials JSON."
        )
    session = boto3.session.Session(
        aws_access_key_id=creds.access_key, aws_secret_access_key=creds.secret_key
    )
    client = session.client(
        "s3",
        endpoint_url=creds.endpoint,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )
    return client, creds.bucket


def list_objects(prefix: str = "") -> list[tuple[str, int]]:
    client, bucket = _client()
    paginator = client.get_paginator("list_objects_v2")
    out: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend((obj["Key"], obj["Size"]) for obj in page.get("Contents", []))
    return sorted(out)


def pull(prefix: str = "", dest: Path = RAW_DIR, overwrite: bool = False) -> list[Path]:
    """Download every object under ``prefix``, skipping files already the right size."""
    ensure_dirs()
    client, bucket = _client()
    written: list[Path] = []
    for key, size in list_objects(prefix):
        target = dest / Path(key).name
        if target.exists() and target.stat().st_size == size and not overwrite:
            print(f"  skip  {key} ({size / 1e6:.1f} MB, already local)")
            written.append(target)
            continue
        print(f"  get   {key} ({size / 1e6:.1f} MB)")
        client.download_file(bucket, key, str(target))
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "pull"])
    parser.add_argument("--prefix", default="")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command == "list":
        objects = list_objects(args.prefix)
        total = sum(size for _, size in objects)
        for key, size in objects:
            print(f"{size / 1e6:9.1f} MB  {key}")
        print(f"\n{len(objects)} objects, {total / 1e6:.1f} MB total")
    else:
        files = pull(args.prefix, args.dest, args.overwrite)
        print(f"\n{len(files)} files in {args.dest}")


if __name__ == "__main__":
    main()
