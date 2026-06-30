"""Persistent, source-validated training frames produced by readiness audit."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.storage.parquet_store import ParquetStore
from model.features import TFT_KNOWN_REALS, TFT_UNKNOWN_REALS
from model.tft import MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH


_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AuditedFrameCache:
    """Store split frames and invalidate them from their relevant source rows."""

    root_dir: Path
    data_store: Path

    def __init__(self, root_dir: Path | str, data_store: Path | str) -> None:
        object.__setattr__(self, "root_dir", Path(root_dir))
        object.__setattr__(self, "data_store", Path(data_store))

    def load(
        self,
        split: str,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, str] | None:
        frame_path = self.frame_path(split)
        manifest_path = self.manifest_path(split)
        if not frame_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, ValueError):
            return None
        fingerprint = self.source_fingerprint(
            station_id,
            start_date,
            end_date,
        )
        expected = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "split": split,
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "source_fingerprint": fingerprint,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        try:
            frame = pd.read_parquet(frame_path)
        except Exception:  # noqa: BLE001
            return None
        if len(frame) != int(manifest.get("row_count", -1)):
            return None
        return frame, fingerprint

    def save(
        self,
        split: str,
        station_id: str,
        start_date: str,
        end_date: str,
        frame: pd.DataFrame,
        source_fingerprint: str | None = None,
    ) -> str:
        fingerprint = source_fingerprint or self.source_fingerprint(
            station_id,
            start_date,
            end_date,
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)
        frame_path = self.frame_path(split)
        frame_tmp = frame_path.with_suffix(".parquet.tmp")
        persisted_frame = _normalize_timestamp_columns(frame)
        persisted_frame.to_parquet(frame_tmp, index=False)
        frame_tmp.replace(frame_path)

        manifest = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "split": split,
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "source_fingerprint": fingerprint,
            "row_count": len(frame),
        }
        manifest_path = self.manifest_path(split)
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="ascii",
        )
        manifest_tmp.replace(manifest_path)
        return fingerprint

    def source_fingerprint(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> str:
        start = pd.Timestamp(f"{start_date} 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{end_date} 23:00:00", tz="UTC")
        encoder_start = start - pd.Timedelta(hours=MAX_ENCODER_LENGTH)
        encoder_end = end - pd.Timedelta(hours=1)
        decoder_end = end + pd.Timedelta(hours=MAX_PREDICTION_LENGTH)

        forecast_store = ParquetStore(self.data_store / "hrrr")
        obs_store = ParquetStore(self.data_store / "obs")
        forecast = forecast_store.read(
            station_id,
            encoder_start - pd.Timedelta(hours=1),
            encoder_end - pd.Timedelta(hours=1),
        )
        obs = obs_store.read(
            station_id,
            encoder_start,
            decoder_end,
        )
        contract = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "encoder_length": MAX_ENCODER_LENGTH,
            "prediction_length": MAX_PREDICTION_LENGTH,
            "known_reals": list(TFT_KNOWN_REALS),
            "unknown_reals": list(TFT_UNKNOWN_REALS),
            "first_hrrr_valid_time": _first_valid_time(
                forecast_store,
                station_id,
                encoder_start - pd.Timedelta(hours=1),
            ),
            "first_obs_valid_time": _first_valid_time(
                obs_store,
                station_id,
                encoder_start,
            ),
        }
        hasher = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        )
        for name, frame in (
            ("hrrr", forecast),
            ("obs", obs),
        ):
            hasher.update(name.encode("ascii"))
            hasher.update(_stable_frame_payload(frame))
        hasher.update(b"decoder")
        hasher.update(self._decoder_payload(station_id, start, end))
        return hasher.hexdigest()

    def frame_path(self, split: str) -> Path:
        return self.root_dir / f"{split}.parquet"

    def manifest_path(self, split: str) -> Path:
        return self.root_dir / f"{split}.manifest.json"

    def _decoder_payload(
        self,
        station_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> bytes:
        station_dir = self.data_store / "_decoder_cache" / station_id
        hasher = hashlib.sha256()
        matched = False
        start_month = pd.Period(start.strftime("%Y-%m"), freq="M")
        end_month = pd.Period(end.strftime("%Y-%m"), freq="M")
        for path in sorted(station_dir.glob("*.parquet")):
            try:
                month = pd.Period(path.stem, freq="M")
            except ValueError:
                continue
            if month < start_month or month > end_month:
                continue
            matched = True
            frame = pd.read_parquet(path)
            if "init_time" in frame.columns:
                init_time = pd.to_datetime(frame["init_time"], utc=True)
                frame = frame[(init_time >= start) & (init_time <= end)]
            hasher.update(path.stem.encode("ascii"))
            hasher.update(_stable_frame_payload(frame.reset_index(drop=True)))
        return hasher.digest() if matched else b"[]"


def _stable_frame_payload(frame: pd.DataFrame) -> bytes:
    if frame.empty:
        return b"[]"
    work = frame.reindex(sorted(frame.columns), axis=1)
    sort_columns = [
        column
        for column in ("init_time", "valid_time", "fxx")
        if column in work.columns
    ]
    if sort_columns:
        work = work.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    schema = {
        "columns": list(work.columns),
        "dtypes": [str(dtype) for dtype in work.dtypes],
        "rows": len(work),
    }
    hasher = hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8"))
    chunk_size = 50_000
    for offset in range(0, len(work), chunk_size):
        chunk = work.iloc[offset : offset + chunk_size]
        try:
            row_hashes = pd.util.hash_pandas_object(
                chunk,
                index=False,
                categorize=True,
            )
        except TypeError:
            normalized = chunk.copy()
            for column in normalized.select_dtypes(include=["object"]).columns:
                normalized[column] = normalized[column].map(repr)
            row_hashes = pd.util.hash_pandas_object(
                normalized,
                index=False,
                categorize=True,
            )
        hasher.update(row_hashes.to_numpy().tobytes())
    return hasher.digest()


def _normalize_timestamp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("init_time", "valid_time"):
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], utc=True)
    return normalized


def _first_valid_time(
    store: ParquetStore,
    station_id: str,
    end: pd.Timestamp,
) -> str | None:
    history = store.read(
        station_id,
        pd.Timestamp("2000-01-01 00:00:00", tz="UTC"),
        end,
    )
    if history.empty or "valid_time" not in history.columns:
        return None
    first = pd.to_datetime(history["valid_time"], utc=True).min()
    return None if pd.isna(first) else pd.Timestamp(first).isoformat()
