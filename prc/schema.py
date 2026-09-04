"""Profile the parquet files on first contact.

Everything downstream depends on what the columns are actually called and how
dirty they are. Run this before writing a single feature.

    python -m prc.schema data/raw
    python -m prc.schema data/raw/2025-01.parquet --rows 5
"""

from __future__ import annotations

import argparse
from pathlib import Path


def profile(path: Path, sample_rows: int = 0) -> str:
    import polars as pl

    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    height = lazy.select(pl.len()).collect().item()

    lines = [f"== {path.name}  —  {height:,} rows x {len(schema)} cols", ""]
    stats = lazy.select(
        [pl.col(name).null_count().alias(f"__null__{name}") for name in schema.names()]
        + [pl.col(name).n_unique().alias(f"__uniq__{name}") for name in schema.names()]
    ).collect()

    lines.append(f"{'column':<28} {'dtype':<18} {'nulls':>10} {'null%':>7} {'unique':>10}")
    lines.append("-" * 78)
    for name, dtype in schema.items():
        nulls = stats[f"__null__{name}"][0]
        uniq = stats[f"__uniq__{name}"][0]
        pct = 100.0 * nulls / height if height else 0.0
        lines.append(f"{name:<28} {str(dtype):<18} {nulls:>10,} {pct:>6.2f}% {uniq:>10,}")

    # Time span, for whichever timestamp columns exist.
    time_cols = [n for n, d in schema.items() if d.base_type() in (pl.Datetime, pl.Date)]
    if time_cols:
        spans = lazy.select(
            [pl.col(c).min().alias(f"min_{c}") for c in time_cols]
            + [pl.col(c).max().alias(f"max_{c}") for c in time_cols]
        ).collect()
        lines += ["", "-- time spans --"]
        lines += [f"{c:<28} {spans[f'min_{c}'][0]}  ..  {spans[f'max_{c}'][0]}" for c in time_cols]

    if sample_rows:
        lines += ["", f"-- first {sample_rows} rows --", str(lazy.head(sample_rows).collect())]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--rows", type=int, default=0, help="also print this many sample rows")
    args = parser.parse_args()

    targets = sorted(args.path.glob("*.parquet")) if args.path.is_dir() else [args.path]
    if not targets:
        raise SystemExit(f"no parquet files under {args.path}")
    for i, target in enumerate(targets):
        if i:
            print("\n")
        print(profile(target, args.rows))


if __name__ == "__main__":
    main()
