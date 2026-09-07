"""METAR weather features from the Iowa Environmental Mesonet ASOS archive.

Source: https://mesonet.agron.iastate.edu/ASOS/ -- a free, openly accessible
archive of global METAR observations, which satisfies the challenge rule that
external datasets be openly accessible and documented. One request per airport;
the data is small (hourly observations, ~14k rows per station over the period).

The evidence for this helping is genuinely mixed and worth stating. The 2024
challenge winner used METAR, but for takeoff weight, where it ranked *last* in
their feature-group ablation. The one published study that tested weather
specifically for taxi-out (Lim et al., SESAR 2021, ATMAP scores from TAF/METAR)
found it non-predictive and dropped it -- though that was one airport over two
winter months.

What makes it worth one experiment here: our worst rows are multi-hour taxi
times at Rome in January and July, the de-icing and convective months, and
nothing in the current feature set can see weather at all. This is the last
source of genuinely new information rather than a rearrangement of what we have.

    python -m prc.weather fetch      # download all ten airports
    python -m prc.weather features   # build and inspect the derived columns
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

from .config import RAW_DIR

ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AIRPORTS = ["LTFM", "LFPG", "EGLL", "EHAM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]
FIELDS = "tmpf,dwpf,sknt,gust,p01i,vsby,skyc1,skyl1,wxcodes"
WEATHER_DIR = RAW_DIR / "weather"

# Derived columns, in the order the model sees them.
FEATURES = [
    "wx_temp_c", "wx_dewpoint_c", "wx_wind_kt", "wx_gust_kt",
    "wx_visibility_km", "wx_ceiling_ft", "wx_precip", "wx_heavy_precip",
    "wx_deice_risk", "wx_thunder", "wx_snow", "wx_freezing", "wx_lowvis",
    "wx_age_min",
]

# European METARs do not report p01i -- that is a US ASOS field, and it came
# back identically zero for all ten stations, which would have made the
# de-icing flag silently dead. Precipitation is read from the present-weather
# codes instead: RA/DZ/SN/GR/GS/PL/SG/IC/UP, optionally prefixed -/+ for light
# and heavy and SH for showers.
PRECIP_CODES = r"(RA|DZ|SN|GR|GS|PL|SG|IC|UP)"


def fetch(station: str, start: str = "2025-01-01", end: str = "2026-08-01") -> Path:
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out = WEATHER_DIR / f"{station}.csv"
    if out.exists() and out.stat().st_size > 10_000:
        print(f"  {station}: cached ({out.stat().st_size / 1e6:.1f} MB)")
        return out
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    query = {
        "station": station, "data": FIELDS,
        "year1": y1, "month1": m1, "day1": d1,
        "year2": y2, "month2": m2, "day2": d2,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no",
        "report_type": "3", "report_type": "4",
    }
    url = ASOS + "?" + urllib.parse.urlencode(query, doseq=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        text = response.read().decode("utf-8", "replace")
    out.write_text(text, encoding="utf-8")
    print(f"  {station}: {len(text.splitlines()) - 1:,} observations")
    return out


def load() -> pl.DataFrame:
    """Parse every station into one frame of derived weather features."""
    frames = []
    for station in AIRPORTS:
        path = WEATHER_DIR / f"{station}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python -m prc.weather fetch`")
        raw = pl.read_csv(path, null_values=["M", "", "None"], infer_schema_length=0)
        frames.append(raw)
    df = pl.concat(frames, how="vertical_relaxed")

    num = lambda c: pl.col(c).str.replace("T", "0.0001").cast(pl.Float64, strict=False)  # noqa: E731
    codes = pl.col("wxcodes").fill_null("")
    return (
        df.select(
            pl.col("station").alias("wx_station"),
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False)
              .dt.replace_time_zone("UTC").alias("wx_time"),
            ((num("tmpf") - 32) * 5 / 9).alias("wx_temp_c"),
            ((num("dwpf") - 32) * 5 / 9).alias("wx_dewpoint_c"),
            num("sknt").alias("wx_wind_kt"),
            num("gust").fill_null(0.0).alias("wx_gust_kt"),
            codes.str.contains(PRECIP_CODES).cast(pl.Int8).alias("wx_precip"),
            codes.str.contains(r"\+").cast(pl.Int8).alias("wx_heavy_precip"),
            (num("vsby") * 1.609).alias("wx_visibility_km"),
            pl.when(pl.col("skyc1").is_in(["BKN", "OVC", "VV "]))
              .then(num("skyl1")).otherwise(pl.lit(30000.0)).alias("wx_ceiling_ft"),
            codes.str.contains("TS").cast(pl.Int8).alias("wx_thunder"),
            codes.str.contains("SN").cast(pl.Int8).alias("wx_snow"),
            codes.str.contains("FZ").cast(pl.Int8).alias("wx_freezing"),
        )
        .drop_nulls("wx_time")
        .with_columns(
            # De-icing is ordered when it is cold AND something is falling; either
            # alone is routine. This is the mechanism that plausibly explains a
            # multi-hour January taxi-out.
            ((pl.col("wx_temp_c") <= 3.0) & (pl.col("wx_precip") == 1))
            .cast(pl.Int8).alias("wx_deice_risk"),
            (pl.col("wx_visibility_km") < 1.5).cast(pl.Int8).alias("wx_lowvis"),
        )
        .sort("wx_station", "wx_time")
        .unique(subset=["wx_station", "wx_time"], keep="last")
        .sort("wx_station", "wx_time")
    )


def attach(movements: pl.DataFrame, weather: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join the most recent observation at or before each movement's time.

    Backward-only: a departure cannot be informed by a report filed after it
    left. ``wx_age_min`` carries how stale the matched observation is, so the
    model can discount a reading from three hours ago.
    """
    weather = load() if weather is None else weather
    out = (
        movements.sort("MVT_TIME_UTC_mvt")
        .join_asof(
            weather.sort("wx_time"),
            left_on="MVT_TIME_UTC_mvt", right_on="wx_time",
            by_left="ADEP_mvt", by_right="wx_station",
            strategy="backward",
        )
        .with_columns(
            ((pl.col("MVT_TIME_UTC_mvt") - pl.col("wx_time")).dt.total_seconds() / 60)
            .alias("wx_age_min")
        )
    )
    # join_asof consumes its right-hand key columns, so drop only what survived.
    return out.drop([c for c in ("wx_time", "wx_station") if c in out.columns])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "features"])
    args = parser.parse_args()

    if args.command == "fetch":
        for i, station in enumerate(AIRPORTS):
            fetch(station)
            if i < len(AIRPORTS) - 1:
                time.sleep(2)  # be a polite client of a free archive
        print(f"\nall stations in {WEATHER_DIR}")
    else:
        wx = load()
        print(f"{wx.height:,} observations across {wx['wx_station'].n_unique()} stations")
        print(wx.select(FEATURES[:-1]).describe().to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
