"""Build and check a submission file.

The ranking service rejects a submission outright if the ID set does not match
the template exactly, so every check it makes is worth making locally first —
a rejected upload costs a round trip and tells you very little.

    python -m prc.submit build preds.parquet --template data/raw/submission_template.parquet
    python -m prc.submit check submissions/<team>_v3.parquet --template ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import SUBMISSIONS_DIR, TEAM_NAME, ensure_dirs

ID_COL = "MVT_ID_mvt"
TARGET_COL = "TAXITIME_SEC_mvt"
FILENAME_RE = re.compile(r"^(?P<team>[a-z0-9-]+)_v(?P<version>\d+)\.parquet$")


class SubmissionError(ValueError):
    """Raised for anything the ranking service would reject."""


def next_version(team: str = "", directory: Path = SUBMISSIONS_DIR) -> int:
    """One past the highest version already written locally."""
    team = team or TEAM_NAME
    highest = 0
    for path in directory.glob(f"{team}_v*.parquet"):
        match = FILENAME_RE.match(path.name)
        if match and match["team"] == team:
            highest = max(highest, int(match["version"]))
    return highest + 1


def check(submission: Path, template: Path) -> None:
    """Raise SubmissionError unless the file would be accepted."""
    import polars as pl

    match = FILENAME_RE.match(submission.name)
    if not match:
        raise SubmissionError(
            f"filename {submission.name!r} does not match '<team-name>_v<integer>.parquet'"
        )
    if TEAM_NAME and match["team"] != TEAM_NAME:
        raise SubmissionError(f"filename team {match['team']!r} != PRC_TEAM_NAME {TEAM_NAME!r}")

    sub = pl.read_parquet(submission)
    tpl = pl.read_parquet(template, columns=[ID_COL])

    missing = {ID_COL, TARGET_COL} - set(sub.columns)
    if missing:
        raise SubmissionError(f"missing column(s): {sorted(missing)}")
    extra = set(sub.columns) - {ID_COL, TARGET_COL}
    if extra:
        raise SubmissionError(f"unexpected column(s): {sorted(extra)}")

    if sub.height != tpl.height:
        raise SubmissionError(f"row count {sub.height:,} != template {tpl.height:,}")
    if sub[ID_COL].n_unique() != sub.height:
        raise SubmissionError("duplicate MVT_ID_mvt values")

    sub_ids, tpl_ids = set(sub[ID_COL].to_list()), set(tpl[ID_COL].to_list())
    if sub_ids != tpl_ids:
        raise SubmissionError(
            f"id set mismatch: {len(tpl_ids - sub_ids):,} missing, {len(sub_ids - tpl_ids):,} extra"
        )

    nulls = sub[TARGET_COL].null_count()
    if nulls:
        raise SubmissionError(f"{nulls:,} null predictions")
    if not sub[TARGET_COL].dtype.is_numeric():
        raise SubmissionError(f"{TARGET_COL} is {sub[TARGET_COL].dtype}, expected numeric")
    negative = int((sub[TARGET_COL] < 0).sum())
    if negative:
        raise SubmissionError(f"{negative:,} negative taxi times")


def build(
    predictions: Path,
    template: Path,
    team: str = "",
    version: int | None = None,
    directory: Path = SUBMISSIONS_DIR,
) -> Path:
    """Join predictions onto the template, write the file, then check it."""
    import polars as pl

    ensure_dirs()
    team = team or TEAM_NAME
    if not team:
        raise SubmissionError("team name unknown — set $PRC_TEAM_NAME or pass --team")
    version = next_version(team, directory) if version is None else version

    tpl = pl.read_parquet(template, columns=[ID_COL])
    preds = pl.read_parquet(predictions).select([ID_COL, TARGET_COL])
    out = tpl.join(preds, on=ID_COL, how="left")

    unmatched = out[TARGET_COL].null_count()
    if unmatched:
        raise SubmissionError(
            f"{unmatched:,} template rows have no prediction — the model did not cover "
            "the whole ranking set"
        )

    target = directory / f"{team}_v{version}.parquet"
    out.write_parquet(target)
    check(target, template)
    print(f"wrote {target}  ({out.height:,} rows)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("predictions", type=Path)
    b.add_argument("--template", type=Path, required=True)
    b.add_argument("--team", default="")
    b.add_argument("--version", type=int)

    c = sub.add_parser("check")
    c.add_argument("submission", type=Path)
    c.add_argument("--template", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build":
        build(args.predictions, args.template, args.team, args.version)
    else:
        check(args.submission, args.template)
        print(f"{args.submission.name} looks acceptable")


if __name__ == "__main__":
    main()
