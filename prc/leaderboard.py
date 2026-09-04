"""Read the public leaderboard.

The challenge site's own leaderboard widget broke when Observable migrated its
notebook framework on 2026-09-01, but the REST API behind it still works and
needs no authentication. This module is how we watch the field.

Usage::

    python -m prc.leaderboard              # summary of the field
    python -m prc.leaderboard --json out.json   # dump every scored submission
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import COMPETITION_ID, DATACOMP_API

LEADERBOARD_URL = f"{DATACOMP_API}/competitions/{COMPETITION_ID}/leaderboard"
_MAX_PAGES = 1000


@dataclass(frozen=True)
class Submission:
    submission_id: str
    team_id: str
    team_name: str
    filename: str
    score: float
    used_pairs: int
    processed_at: str


def fetch_all(url: str = LEADERBOARD_URL) -> list[Submission]:
    """Page through every scored submission. Sorted ascending by score."""
    out: list[Submission] = []
    cursor: str | None = None
    for _ in range(_MAX_PAGES):
        page_url = f"{url}?cursor={cursor}" if cursor else url
        with urllib.request.urlopen(page_url, timeout=60) as response:
            page = json.load(response)
        out.extend(
            Submission(
                item["submissionId"],
                item["teamId"],
                item["teamName"],
                item["filename"],
                float(item["score"]),
                int(item["usedPairs"]),
                item["processedAt"],
            )
            for item in page["items"]
        )
        cursor = page.get("nextCursor")
        if not cursor:
            break
    else:  # pragma: no cover - only if the leaderboard grows past _MAX_PAGES
        raise RuntimeError(f"stopped after {_MAX_PAGES} pages; cursor still {cursor!r}")
    return out


def team_bests(subs: list[Submission]) -> list[tuple[str, float, int]]:
    """(team, best RMSE, submission count), best first."""
    best: dict[str, float] = {}
    counts: Counter[str] = Counter()
    for sub in subs:
        counts[sub.team_name] += 1
        if sub.team_name not in best or sub.score < best[sub.team_name]:
            best[sub.team_name] = sub.score
    return sorted(((t, s, counts[t]) for t, s in best.items()), key=lambda row: row[1])


def summarise(subs: list[Submission], top: int = 20) -> str:
    ranked = team_bests(subs)
    scores = [row[1] for row in ranked]
    lines = [
        f"{len(subs)} scored submissions from {len(ranked)} teams",
        f"latest: {max(s.processed_at for s in subs)}",
        "",
        f"-- top {top} --",
    ]
    lines += [
        f"{i:3d}. {score:9.4f}  {team}  ({n} subs)"
        for i, (team, score, n) in enumerate(ranked[:top], 1)
    ]
    lines += ["", "-- percentiles of team-best RMSE --"]
    for q in (0, 10, 25, 50, 75, 90, 100):
        idx = min(len(scores) - 1, round(q / 100 * (len(scores) - 1)))
        lines.append(f"p{q:<3d}: {scores[idx]:.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write every submission to this path")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    subs = fetch_all()
    print(summarise(subs, top=args.top))
    if args.json:
        args.json.write_text(
            json.dumps([sub.__dict__ for sub in subs], indent=1), encoding="utf-8"
        )
        print(f"\nwrote {len(subs)} submissions to {args.json}")


if __name__ == "__main__":
    main()
