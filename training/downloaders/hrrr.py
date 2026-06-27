"""Batch scheduling for canonical HRRR f01 training data."""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config.stations import STATIONS
from data.forecast.hrrr_s3 import HrrrS3Source
from data.storage.cleaner import Cleaner
from data.storage.parquet_store import ParquetStore


class BatchHrrrF01Downloader:
    """Fetch hourly f01 rows through a forecast source with skip/resume."""

    def __init__(
        self,
        store: ParquetStore,
        station_id: str = "KLGA",
        max_workers: int = 4,
        station_config: Any | None = None,
        forecast_source: Any | None = None,
    ) -> None:
        self._store = store
        self._station = station_config or STATIONS[station_id]
        self._station_id = self._station.station_id
        self._max_workers = max(1, min(int(max_workers), 4))
        self._source = forecast_source

    def run(
        self,
        start_date: str,
        end_date: str,
        init_hours: list[int] | None = None,
    ) -> int:
        all_inits = self._build_inits(start_date, end_date, init_hours)
        done_keys = self._already_done_keys()
        pending = [
            (day, hour)
            for day, hour in all_inits
            if f"{day.strftime('%Y%m%d')}_{hour:02d}z" not in done_keys
        ]
        print(
            f"BatchHrrrF01Downloader: {len(all_inits)} inits total, "
            f"{len(done_keys)} already done, {len(pending)} pending.",
            flush=True,
        )
        if not pending:
            return 0
        return self._execute(pending)

    @staticmethod
    def _build_inits(
        start_date: str,
        end_date: str,
        init_hours: list[int] | None,
    ) -> list[tuple[Any, int]]:
        dates = pd.date_range(start_date, end_date, freq="D")
        hours = init_hours if init_hours is not None else list(range(24))
        return [(day.date(), hour) for day in dates for hour in hours]

    def _already_done_keys(self) -> set[str]:
        try:
            existing = self._store.read(
                self._station_id,
                start=pd.Timestamp("2000-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            if existing.empty or "init_time" not in existing.columns:
                return set()
            init_times = pd.to_datetime(existing["init_time"], utc=True)
            return set(init_times.dt.strftime("%Y%m%d_%Hz"))
        except Exception:  # noqa: BLE001
            return set()

    def _execute(self, pending: list[tuple[Any, int]]) -> int:
        total_rows = 0
        errors = 0
        completed = 0
        started = time.perf_counter()

        def fetch_one(day: Any, hour: int) -> pd.DataFrame:
            init_utc = pd.Timestamp(
                datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
            )
            source = self._source or HrrrS3Source()
            result = source.fetch(init_utc, range(1, 2), self._station)
            return result.get(self._station_id, pd.DataFrame())

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {
                pool.submit(fetch_one, day, hour): (day, hour)
                for day, hour in pending
            }
            for future in as_completed(future_map):
                day, hour = future_map[future]
                completed += 1
                try:
                    frame = future.result()
                    if frame is None or frame.empty:
                        print(f"  EMPTY {day} {hour:02d}z", flush=True)
                    else:
                        frame = Cleaner.process_hrrr(frame)
                        total_rows += self._store.append(self._station_id, frame)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"  ERROR {day} {hour:02d}z: {exc}", flush=True)
                    if errors <= 3:
                        traceback.print_exc()

                if completed <= 5 or completed % 10 == 0 or completed == len(pending):
                    elapsed = time.perf_counter() - started
                    rate = completed / max(elapsed, 0.001)
                    remaining = len(pending) - completed
                    eta_min = remaining / rate / 60.0 if rate > 0 else float("inf")
                    print(
                        f"[{completed}/{len(pending)}] "
                        f"rate={rate:.2f} init/s ETA={eta_min:.1f}m "
                        f"errors={errors} written={total_rows}",
                        flush=True,
                    )

        print(
            f"\nBatchHrrrF01Downloader done: {total_rows} rows, errors={errors}",
            flush=True,
        )
        return total_rows
