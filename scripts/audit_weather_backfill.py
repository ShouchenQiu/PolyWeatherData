"""Audit local weather backfill completeness without downloading data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.stations import STATIONS
from data.storage.parquet_store import ParquetStore


def audit_weather_backfill(
    data_store: Path | str,
    station_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    root = Path(data_store)
    expected_hours = _expected_hours(start_date, end_date)
    forecast = _audit_hourly_store(
        ParquetStore(root / "hrrr"),
        station_id,
        expected_hours,
        timestamp_col="valid_time",
        read_offset=pd.Timedelta(hours=-1),
    )
    obs = _audit_hourly_store(
        ParquetStore(root / "obs"),
        station_id,
        expected_hours,
        timestamp_col="valid_time",
    )
    decoder = _audit_decoder_cache(root / "_decoder_cache", station_id, expected_hours)
    return {
        "data_store": str(root),
        "station_id": station_id,
        "start_date": start_date,
        "end_date": end_date,
        "expected_hours": len(expected_hours),
        "hrrr_f01": forecast,
        "obs": obs,
        "decoder": decoder,
    }


def _expected_hours(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date + " 00:00:00", tz="UTC")
    end = pd.Timestamp(end_date + " 23:00:00", tz="UTC")
    return list(pd.date_range(start, end, freq="h"))


def _audit_hourly_store(
    store: ParquetStore,
    station_id: str,
    expected_hours: list[pd.Timestamp],
    timestamp_col: str,
    read_offset: pd.Timedelta | None = None,
) -> dict[str, Any]:
    if not expected_hours:
        return {"rows": 0, "complete_hours": 0, "missing_hours": []}
    offset = read_offset or pd.Timedelta(0)
    df = store.read(station_id, expected_hours[0] + offset, expected_hours[-1] + offset)
    if df.empty or timestamp_col not in df.columns:
        present: set[pd.Timestamp] = set()
        rows = 0
    else:
        times = pd.to_datetime(df[timestamp_col], utc=True).dt.floor("h")
        present = set(pd.Timestamp(ts) for ts in times)
        rows = len(df)
    expected = set(expected_hours)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    return {
        "rows": rows,
        "complete_hours": len(expected) - len(missing),
        "missing_count": len(missing),
        "missing_hours": [str(ts) for ts in missing],
        "extra_count": len(extra),
        "extra_hours": [str(ts) for ts in extra],
    }


def _audit_decoder_cache(
    cache_dir: Path,
    station_id: str,
    expected_inits: list[pd.Timestamp],
) -> dict[str, Any]:
    expected = set(expected_inits)
    complete: set[pd.Timestamp] = set()
    partial: dict[pd.Timestamp, int] = {}
    station_dir = cache_dir / station_id
    if station_dir.exists():
        for path in sorted(station_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception:  # noqa: BLE001
                continue
            if df.empty or "init_time" not in df.columns or "fxx" not in df.columns:
                continue
            work = df.copy()
            work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
            work["fxx"] = work["fxx"].astype(int)
            for init_utc, group in work.groupby("init_time", sort=False):
                init = pd.Timestamp(init_utc)
                if init not in expected:
                    continue
                fxx = set(group["fxx"].astype(int))
                if fxx >= set(range(1, 49)):
                    complete.add(init)
                else:
                    partial[init] = len(fxx & set(range(1, 49)))
    missing = sorted(expected - complete - set(partial))
    return {
        "requested_inits": len(expected),
        "complete_inits": len(complete),
        "partial_inits": len(partial),
        "missing_inits": len(missing),
        "missing_init_times": [str(ts) for ts in missing],
        "partial_init_times": {
            str(ts): rows for ts, rows in sorted(partial.items())
        },
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local HRRR f01, IEM obs, and decoder cache completeness"
    )
    parser.add_argument("--data-store", default=".demo/data_store")
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS.keys()))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    summary = audit_weather_backfill(
        Path(args.data_store),
        args.station,
        args.start,
        args.end,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
