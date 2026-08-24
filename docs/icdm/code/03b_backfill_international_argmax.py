"""Backfill international stations only (LONDON/TOKYO/SHANGHAI/SEOUL/TORONTO).

Loads existing daily_high_argmax.parquet (US stations already present),
downloads the 5 international stations with the fixed monthly logic, and
merges the results back into the same file.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

# import functions from 03_download_daily_high_argmax.py via importlib
_spec = importlib.util.spec_from_file_location(
    "argmax_mod",
    r"d:\NN\PolyWeatherData\docs\icdm\code\03_download_daily_high_argmax.py",
)
_argmax_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_argmax_mod)
END_DATE = _argmax_mod.END_DATE
daily_argmax = _argmax_mod.daily_argmax
fetch_openmeteo = _argmax_mod.fetch_openmeteo
months_between = _argmax_mod.months_between

OUT_FILE = Path(r"d:\NN\PolyWeatherData\docs\icdm\output\daily_high_argmax.parquet")

# international stations: (lat, lon, tz, start)
INTL = {
    "LONDON": (51.5074, -0.1278, "Europe/London", "2025-01-22"),
    "TOKYO": (35.6762, 139.6503, "Asia/Tokyo", "2026-03-11"),
    "SHANGHAI": (31.2304, 121.4737, "Asia/Shanghai", "2026-03-13"),
    "SEOUL": (37.5665, 126.9780, "Asia/Seoul", "2025-12-06"),
    "TORONTO": (43.6532, -79.3832, "America/Toronto", "2025-12-06"),
}


def main() -> int:
    if OUT_FILE.exists():
        existing = pd.read_parquet(OUT_FILE)
    else:
        existing = pd.DataFrame()
    print(f"[start] existing rows: {len(existing)}", flush=True)

    new_rows: list[pd.DataFrame] = []
    for station, (lat, lon, tz, start) in INTL.items():
        frames = []
        try:
            months = months_between(start, END_DATE)
            for d1, d2 in months:
                df = fetch_openmeteo(lat, lon, d1, d2)
                frames.append(df)
                time.sleep(0.8)
            raw = pd.concat(frames, ignore_index=True)
            arg = daily_argmax(raw, "tmpc", tz)
            arg["station"] = station
            arg["unit"] = "C"
            new_rows.append(arg)
            print(
                f"[complete] {station}: {len(arg)} days ({len(months)} months)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[error] {station}: {type(exc).__name__}: {str(exc)[:120]}",
                flush=True,
            )
        time.sleep(1.0)

    if new_rows:
        merged = pd.concat([existing] + new_rows, ignore_index=True)
        merged.to_parquet(OUT_FILE, index=False)
        print(
            f"[done] merged {len(merged)} rows "
            f"stations={sorted(merged['station'].unique())}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
