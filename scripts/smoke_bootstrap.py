"""Bootstrap the fixed 16-day smoke dataset without touching production stores."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from data.forecast.hrrr_s3 import HrrrS3Source
from data.storage.cleaner import Cleaner
from data.storage.parquet_store import ParquetStore
from model.training_data import TrainingDecoderCache
from scripts.batch_download_iem import BatchIemDownloader
from scripts.batch_download_zarr import BatchZarrDownloader
from scripts.seed_demo_data import seed_demo

SMOKE_START = "2026-05-20"
SMOKE_END = "2026-06-04"
SMOKE_SPLIT = {
    "train": ["2026-05-26", "2026-05-30"],
    "val": ["2026-05-31", "2026-06-01"],
    "test": ["2026-06-02", "2026-06-04"],
}
DEFAULT_SLUG = "highest-temperature-in-nyc-on-june-5-2026"


class BatchHrrrS3F01Downloader(BatchZarrDownloader):
    """Thin downloader adapter that writes canonical f01 via production S3 source."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("forecast_source", HrrrS3Source())
        super().__init__(*args, **kwargs)

    def _execute_via_source(self, pending: list[tuple[Any, int]]) -> int:
        total_rows = 0
        errors = 0
        completed = 0
        started = time.perf_counter()
        init_workers = max(1, min(int(self._max_workers), 4))

        def fetch_one(d: Any, h: int) -> pd.DataFrame:
            init_utc = pd.Timestamp(
                datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
            )
            source = HrrrS3Source()
            result = source.fetch(init_utc, range(1, 2), self._station)
            return result.get(self._station_id, pd.DataFrame())

        with ThreadPoolExecutor(max_workers=init_workers) as pool:
            future_map = {pool.submit(fetch_one, d, h): (d, h) for d, h in pending}
            for fut in as_completed(future_map):
                d, h = future_map[fut]
                completed += 1
                try:
                    df = fut.result()
                    if df is None or df.empty:
                        print(f"  EMPTY {d} {h:02d}z", flush=True)
                        continue
                    df = Cleaner.process_hrrr(df)
                    total_rows += self._store.append(self._station_id, df)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"  ERROR {d} {h:02d}z: {exc}", flush=True)
                    if errors <= 3:
                        traceback.print_exc()

                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 0.001)
                remaining = len(pending) - completed
                eta_min = remaining / rate / 60.0 if rate > 0 else float("inf")
                if completed % 10 == 0 or completed <= 5:
                    print(
                        f"[{completed}/{len(pending)}] "
                        f"rate={rate:.2f} init/s ETA={eta_min:.1f}m "
                        f"errors={errors} written={total_rows}",
                        flush=True,
                    )

        print(
            f"\nBatchHrrrS3F01Downloader done: {total_rows} rows, errors={errors}",
            flush=True,
        )
        return total_rows


def run_smoke_bootstrap(
    smoke_dir: Path | str = Path(".demo"),
    station_id: str = "KLGA",
    start_date: str = SMOKE_START,
    end_date: str = SMOKE_END,
    workers: int = 12,
    dry_run: bool = False,
    skip_polymarket: bool = False,
    skip_hrrr: bool = False,
    skip_iem: bool = False,
    skip_decoder: bool = False,
    f01_downloader_cls: Any = BatchHrrrS3F01Downloader,
    iem_downloader_cls: Any = BatchIemDownloader,
    decoder_cache_cls: Any = TrainingDecoderCache,
) -> dict[str, Any]:
    """Create a small isolated weather/model smoke dataset."""
    root = Path(smoke_dir)
    data_store_dir = root / "data_store"
    settings = Settings(data_store_dir=data_store_dir)
    station = STATIONS[station_id]
    init_times = _init_times(start_date, end_date)

    plan: dict[str, Any] = {
        "dry_run": dry_run,
        "smoke_dir": str(root),
        "data_store_dir": str(data_store_dir),
        "station_id": station_id,
        "start_date": start_date,
        "end_date": end_date,
        "init_count": len(init_times),
        "decoder_init_count": len(init_times),
        "decoder_skipped_tail_cap": 0,
        "workers": workers,
        "smoke_split": SMOKE_SPLIT,
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan

    root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, dict[str, Any]] = {}

    if skip_polymarket:
        polymarket_summary = {"skipped": True}
    else:
        polymarket_summary = seed_demo(
            demo_dir=root,
            station_id=station_id,
            seed_date=end_date,
            polymarket_slug=DEFAULT_SLUG,
            skip_hrrr=True,
            skip_iem=True,
        )

    hrrr_f01_written = 0
    if not skip_hrrr:
        t0 = time.perf_counter()
        hrrr_store = ParquetStore(settings.forecast_store_dir)
        downloader = f01_downloader_cls(
            hrrr_store,
            station_config=station,
            max_workers=workers,
        )
        hrrr_f01_written = downloader.run(start_date, end_date, init_hours=None)
        timings["hrrr_f01"] = _timing(time.perf_counter() - t0, hrrr_f01_written, 0)
    else:
        timings["hrrr_f01"] = _timing(0.0, 0, 0)

    obs_written = 0
    if not skip_iem:
        t0 = time.perf_counter()
        obs_store = ParquetStore(settings.obs_store_dir)
        downloader = iem_downloader_cls(
            obs_store,
            station_config=station,
            delay=0.0,
        )
        iem_station = (
            station.station_id[1:]
            if station.station_id.startswith("K")
            else station.station_id
        )
        obs_written = downloader.run(
            start_date,
            end_date,
            station=iem_station,
            network=station.iem_network,
        )
        timings["iem"] = _timing(time.perf_counter() - t0, obs_written, 0)
    else:
        timings["iem"] = _timing(0.0, 0, 0)

    if skip_decoder:
        decoder_summary = {
            "requested_inits": len(init_times),
            "skipped_cached": 0,
            "fetched_inits": 0,
            "rows_written": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
            "init_per_second": 0.0,
            "major_inits": 0,
            "offhour_inits": 0,
            "skipped_tail_cap": 0,
        }
    else:
        cache = decoder_cache_cls(settings.training_decoder_cache_dir)
        decoder_workers = max(1, min(int(workers), 4))
        decoder_summary = cache.prewarm(
            station, init_times, workers=decoder_workers, progress=True
        )
        decoder_summary["requested_workers"] = workers
        decoder_summary["effective_workers"] = decoder_workers
        decoder_summary["skipped_tail_cap"] = 0

    summary = {
        **plan,
        "polymarket": polymarket_summary,
        "hrrr_f01_written": hrrr_f01_written,
        "obs_written": obs_written,
        "decoder": decoder_summary,
        "timings": timings,
    }
    (root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def _init_times(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date + " 00:00:00", tz="UTC")
    end = pd.Timestamp(end_date + " 23:00:00", tz="UTC")
    return list(pd.date_range(start, end, freq="h"))


def _timing(elapsed_seconds: float, rows_written: int, errors: int) -> dict[str, Any]:
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "rows_written": int(rows_written),
        "errors": int(errors),
        "rows_per_second": (
            float(rows_written) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap fixed smoke data")
    parser.add_argument("--smoke-dir", default=".demo")
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS.keys()))
    parser.add_argument("--start", default=SMOKE_START)
    parser.add_argument("--end", default=SMOKE_END)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-polymarket", action="store_true")
    parser.add_argument("--skip-hrrr", action="store_true")
    parser.add_argument("--skip-iem", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    args = parser.parse_args()

    run_smoke_bootstrap(
        smoke_dir=Path(args.smoke_dir),
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        workers=args.workers,
        dry_run=args.dry_run,
        skip_polymarket=args.skip_polymarket,
        skip_hrrr=args.skip_hrrr,
        skip_iem=args.skip_iem,
        skip_decoder=args.skip_decoder,
    )


if __name__ == "__main__":
    _cli()
