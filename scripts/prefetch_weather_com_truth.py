"""Prefetch Weather.com truth into the daily parquet cache used by Data Health."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from data.obs.weather_com import WeatherComTruthClient, local_day_hour_coverage
from data.storage.parquet_store import ParquetStore


def prefetch_weather_com_truth(
    data_store_dir: Path | str,
    truth_store_dir: Path | str,
    station_id: str = "KLGA",
    start: str | None = None,
    end: str | None = None,
    workers: int = 4,
    client_cls: Any = WeatherComTruthClient,
) -> dict[str, Any]:
    """Fetch uncached Weather.com local dates represented by IEM observations."""
    data_root = Path(data_store_dir)
    truth_root = Path(truth_store_dir)
    obs = ParquetStore(data_root / "obs").read(
        station_id,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        end=pd.Timestamp("2100-01-01", tz="UTC"),
    )
    if obs.empty or "valid_time" not in obs.columns:
        return _prefetch_summary(station_id, [], 0, 0, [])

    valid_times = pd.to_datetime(obs["valid_time"], utc=True)
    if start is not None:
        valid_times = valid_times[valid_times >= _utc_day(start)]
    if end is not None:
        valid_times = valid_times[
            valid_times < _utc_day(end) + pd.Timedelta(days=1)
        ]
    local_dates = sorted(
        valid_times.dt.tz_convert(STATIONS[station_id].timezone)
        .dt.date.astype(str)
        .drop_duplicates()
        .tolist()
    )

    cache_reader = client_cls(truth_root)
    timezone = STATIONS[station_id].timezone

    def complete_local_day(frame: pd.DataFrame | None, day: str) -> bool:
        if frame is None:
            return False
        observed, expected = local_day_hour_coverage(
            frame,
            local_date=day,
            timezone=timezone,
        )
        return observed == expected

    pending = []
    for day in local_dates:
        cached = cache_reader.read_cached_date(station_id, day)
        if not complete_local_day(cached, day):
            pending.append(day)
    cached_count = len(local_dates) - len(pending)
    fetched_count = 0
    errors: list[dict[str, str]] = []

    def fetch(day: str) -> str:
        frame = client_cls(truth_root).fetch_date(station_id, day)
        observed, expected = local_day_hour_coverage(
            frame,
            local_date=day,
            timezone=timezone,
        )
        if observed != expected:
            raise ValueError(
                "incomplete Weather.com local day after fetch "
                f"station={station_id} date={day} "
                f"observed_hours={observed} expected_hours={expected}"
            )
        return day

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(fetch, day): day for day in pending}
        for future in as_completed(futures):
            day = futures[future]
            try:
                future.result()
                fetched_count += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"date": day, "error": str(exc)})

    return _prefetch_summary(
        station_id,
        local_dates,
        cached_count,
        fetched_count,
        errors,
    )


def _utc_day(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _prefetch_summary(
    station_id: str,
    local_dates: list[str],
    cached_count: int,
    fetched_count: int,
    error_details: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "station_id": station_id,
        "requested_dates": len(local_dates),
        "first_local_date": local_dates[0] if local_dates else None,
        "last_local_date": local_dates[-1] if local_dates else None,
        "cached_dates": int(cached_count),
        "fetched_dates": int(fetched_count),
        "errors": len(error_details),
        "error_details": error_details,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch Weather.com truth for dates represented in the IEM store"
    )
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-store")
    parser.add_argument("--truth-store")
    args = parser.parse_args()

    settings = Settings()
    data_store = Path(
        args.data_store
        or os.environ.get("POLYWEATHER_DATA_STORE")
        or settings.data_store_dir
    )
    truth_store = Path(
        args.truth_store
        or settings.truth_store_dir
    )
    summary = prefetch_weather_com_truth(
        data_store_dir=data_store,
        truth_store_dir=truth_store,
        station_id=args.station,
        start=args.start,
        end=args.end,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
