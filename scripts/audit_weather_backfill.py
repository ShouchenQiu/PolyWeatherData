"""Audit local weather backfill completeness without downloading data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.stations import STATIONS
from config.settings import Settings
from data.manager import DataManager
from data.storage.parquet_store import ParquetStore
from model.features import TARGET, TFT_KNOWN_REALS, TFT_UNKNOWN_REALS
from model.training_data import TrainingDecoderCache, TrainingDatasetBuilder


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


def audit_training_readiness(
    data_store: Path | str,
    station_id: str,
    split_ranges: dict[str, tuple[str, str] | list[str]],
    dataset_builder: Any | None = None,
) -> dict[str, Any]:
    root = Path(data_store)
    station = STATIONS[station_id]
    failures: list[str] = []
    split_summaries: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []

    if dataset_builder is None:
        settings = Settings(data_store_dir=root)
        forecast_store = ParquetStore(settings.forecast_store_dir)
        obs_store = ParquetStore(settings.obs_store_dir)
        data_manager = DataManager([], [], forecast_store, obs_store, settings=settings)
        decoder_cache = TrainingDecoderCache(settings.training_decoder_cache_dir)
        dataset_builder = TrainingDatasetBuilder(data_manager, decoder_cache)

    for split_name in ("train", "val", "test"):
        start, end = split_ranges[split_name]
        init_times = _expected_hours(start, end)
        try:
            frame = dataset_builder.build_frame(station, init_times, split=split_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{split_name}: builder failed: {exc}")
            frame = pd.DataFrame()
        frames.append(frame)
        summary = _training_frame_summary(frame, init_times)
        split_summaries[split_name] = summary
        if not summary["non_empty"]:
            failures.append(f"{split_name}: empty split")
        if not summary["init_row_count_ok"]:
            failures.append(f"{split_name}: one or more init windows are not 120 rows")
        if summary["missing_required_columns"]:
            failures.append(
                f"{split_name}: missing required columns "
                f"{summary['missing_required_columns']}"
            )
        if summary["nan_required_columns"]:
            failures.append(
                f"{split_name}: NaN in required columns {summary['nan_required_columns']}"
            )

    data_hash = _hash_training_frames(frames)
    return {
        "ready": not failures,
        "failures": failures,
        "data_store": str(root),
        "station_id": station_id,
        "splits": split_summaries,
        "data_order_hash": data_hash,
        "feature_list_and_order": {
            "known_reals": list(TFT_KNOWN_REALS),
            "unknown_reals": list(TFT_UNKNOWN_REALS),
            "target": TARGET,
        },
    }


def _expected_hours(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date + " 00:00:00", tz="UTC")
    end = pd.Timestamp(end_date + " 23:00:00", tz="UTC")
    return list(pd.date_range(start, end, freq="h"))


def _training_frame_summary(
    frame: pd.DataFrame,
    init_times: list[pd.Timestamp],
) -> dict[str, Any]:
    required = set(TFT_KNOWN_REALS + [TARGET, "valid_time", "init_time_str", "local_idx"])
    if frame.empty:
        return {
            "row_count": 0,
            "sample_count": 0,
            "expected_sample_count": len(init_times),
            "non_empty": False,
            "init_row_count_ok": False,
            "missing_required_columns": sorted(required),
            "nan_required_columns": [],
        }
    missing = sorted(col for col in required if col not in frame.columns)
    nan_cols = []
    for col in sorted(required - set(missing)):
        if frame[col].isna().any():
            nan_cols.append(col)
    counts = frame.groupby("init_time_str").size() if "init_time_str" in frame.columns else pd.Series(dtype=int)
    return {
        "row_count": len(frame),
        "sample_count": int(len(counts)),
        "expected_sample_count": len(init_times),
        "non_empty": True,
        "init_row_count_ok": bool((counts == 120).all()) and len(counts) == len(init_times),
        "missing_required_columns": missing,
        "nan_required_columns": nan_cols,
    }


def _hash_training_frames(frames: list[pd.DataFrame]) -> str:
    hasher = hashlib.sha256()
    cols = ["split", "init_time_str", "local_idx", "valid_time", "fxx", TARGET]
    for frame in frames:
        present = [col for col in cols if col in frame.columns]
        work = frame[present].copy()
        for col in work.columns:
            if pd.api.types.is_datetime64_any_dtype(work[col]):
                work[col] = pd.to_datetime(work[col], utc=True).astype(str)
        hasher.update(work.to_json(orient="records", date_format="iso").encode("utf-8"))
    return hasher.hexdigest()


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
