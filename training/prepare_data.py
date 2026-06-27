"""Prepare HRRR f01, IEM obs, and decoder-cache training data."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from data.storage.parquet_store import ParquetStore
from training.dataset import TrainingDecoderCache
from training.downloaders.hrrr import BatchHrrrF01Downloader
from training.downloaders.iem import BatchIemDownloader
from training.workspace import TrainingWorkspace

DEFAULT_START = "2026-05-20"
DEFAULT_END = "2026-06-04"


def prepare_training_data(
    run_dir: Path | str = Path(".demo"),
    station_id: str = "KLGA",
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    workers: int = 4,
    dry_run: bool = False,
    skip_hrrr: bool = False,
    skip_iem: bool = False,
    skip_decoder: bool = False,
    f01_downloader_cls: Any = BatchHrrrF01Downloader,
    iem_downloader_cls: Any = BatchIemDownloader,
    decoder_cache_cls: Any = TrainingDecoderCache,
) -> dict[str, Any]:
    """Create isolated weather training data for a date range."""
    root = Path(run_dir)
    workspace = TrainingWorkspace(root)
    data_store_dir = workspace.data_store_dir
    settings = Settings(data_store_dir=data_store_dir)
    station = STATIONS[station_id]
    init_times = _init_times(start_date, end_date)

    plan: dict[str, Any] = {
        "dry_run": dry_run,
        "run_dir": str(root),
        "data_store_dir": str(data_store_dir),
        "station_id": station_id,
        "start_date": start_date,
        "end_date": end_date,
        "init_count": len(init_times),
        "decoder_init_count": len(init_times),
        "decoder_skipped_tail_cap": 0,
        "workers": workers,
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan

    started_at = datetime.now(timezone.utc).isoformat()
    workspace.ensure()
    timings: dict[str, dict[str, Any]] = {}

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
        try:
            decoder_summary = cache.prewarm(
                station, init_times, workers=decoder_workers, progress=True
            )
        except KeyboardInterrupt:
            partial_summary = {
                **plan,
                "interrupted": True,
                "phase": "decoder",
                "hrrr_f01_written": hrrr_f01_written,
                "obs_written": obs_written,
                "decoder": _partial_decoder_summary(
                    settings.training_decoder_cache_dir,
                    station_id,
                    init_times,
                ),
                "timings": timings,
            }
            workspace.write_json(
                root / "prepare_data_summary.partial.json",
                partial_summary,
            )
            raise
        decoder_summary["requested_workers"] = workers
        decoder_summary["effective_workers"] = decoder_workers
        decoder_summary["skipped_tail_cap"] = 0

    summary = {
        **plan,
        "status": "prepared",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "hrrr_f01_written": hrrr_f01_written,
        "obs_written": obs_written,
        "decoder": decoder_summary,
        "timings": timings,
    }
    workspace.write_json(root / "prepare_data_summary.json", summary)
    workspace.record_stage("prepare", summary)
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


def _partial_decoder_summary(
    cache_dir: Path,
    station_id: str,
    init_times: list[pd.Timestamp],
) -> dict[str, Any]:
    requested = {_to_utc(ts) for ts in init_times}
    complete: set[pd.Timestamp] = set()
    station_dir = Path(cache_dir) / station_id
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
                if pd.Timestamp(init_utc) not in requested:
                    continue
                if set(group["fxx"].astype(int)) >= set(range(1, 49)):
                    complete.add(pd.Timestamp(init_utc))
    return {
        "requested_inits": len(requested),
        "complete_inits": len(complete),
        "pending_inits": len(requested) - len(complete),
        "cache_dir": str(cache_dir),
    }


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
