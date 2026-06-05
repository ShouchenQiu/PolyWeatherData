"""Training data helpers shared by offline scripts and TFT training.

The canonical HRRR ParquetStore remains f01-only.  This module owns
training-only artifacts that are useful offline but must not be read by
production inference, such as cached full decoder windows.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.forecast.hrrr_zarr import HrrrZarrSource
from data.manager import InsufficientEncoderData
from data.storage.cleaner import Cleaner
from model.features import add_model_features
from model.tft import MAX_ENCODER_LENGTH


@dataclass
class TrainingDecoderCache:
    """File-backed cache for full HRRR decoder windows used in training only."""

    root_dir: Path
    source: Any | None = None

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.source is None:
            self.source = HrrrZarrSource()

    def path_for(self, station_id: str, init_time: pd.Timestamp) -> Path:
        init_utc = _to_utc(init_time)
        station_dir = self.root_dir / station_id
        station_dir.mkdir(parents=True, exist_ok=True)
        return station_dir / f"{init_utc.strftime('%Y-%m')}.parquet"

    def get(self, station: Any, init_time: pd.Timestamp) -> pd.DataFrame:
        """Return cleaned fxx 1..48 decoder rows, fetching and caching on miss."""
        init_utc = _to_utc(init_time)
        cached = self._read_cached_init(station.station_id, init_utc)
        if cached is not None:
            return cached

        df = self._fetch_clean_validate(station, init_utc)
        self._append_month(station.station_id, init_utc, df)
        return df

    def prewarm(
        self,
        station: Any,
        init_times: list[pd.Timestamp],
        workers: int = 12,
    ) -> dict[str, Any]:
        """Fetch and persist missing decoder windows for a list of init times.

        Existing monthly-cache rows are skipped.  Writes are serialized in the
        caller thread so concurrent downloads never race on parquet files.
        """
        started = time.perf_counter()
        init_utcs = [_to_utc(ts) for ts in init_times]
        pending = [
            init_utc for init_utc in init_utcs
            if self._read_cached_init(station.station_id, init_utc) is None
        ]
        rows_written = 0
        errors: list[dict[str, str]] = []
        fetched = 0
        worker_count = max(1, int(workers))

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(self._fetch_clean_validate, station, init_utc): init_utc
                for init_utc in pending
            }
            for fut in as_completed(future_map):
                init_utc = future_map[fut]
                try:
                    df = fut.result()
                    rows_written += self._append_month(
                        station.station_id, init_utc, df
                    )
                    fetched += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append({"init_time": str(init_utc), "error": str(exc)})

        elapsed = time.perf_counter() - started
        return {
            "requested_inits": len(init_utcs),
            "skipped_cached": len(init_utcs) - len(pending),
            "fetched_inits": fetched,
            "rows_written": rows_written,
            "errors": len(errors),
            "error_details": errors,
            "elapsed_seconds": elapsed,
            "init_per_second": fetched / elapsed if elapsed > 0 else 0.0,
            "major_inits": sum(1 for ts in init_utcs if ts.hour in {0, 6, 12, 18}),
            "offhour_inits": sum(1 for ts in init_utcs if ts.hour not in {0, 6, 12, 18}),
        }

    def _read_cached_init(
        self,
        station_id: str,
        init_utc: pd.Timestamp,
    ) -> pd.DataFrame | None:
        path = self.path_for(station_id, init_utc)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty or "init_time" not in df.columns:
            return None
        init_series = pd.to_datetime(df["init_time"], utc=True)
        subset = df[init_series == init_utc].copy()
        if subset.empty:
            return None
        return self._validate(subset, station_id, init_utc)

    def _fetch_clean_validate(self, station: Any, init_utc: pd.Timestamp) -> pd.DataFrame:
        result = self.source.fetch(init_utc, range(0, 49), station)
        df = result.get(station.station_id, pd.DataFrame())
        if df is None or df.empty:
            raise RuntimeError(
                f"TrainingDecoderCache: no decoder rows for "
                f"station={station.station_id} init={init_utc}"
            )
        df = Cleaner.process_hrrr(df)
        return self._validate(df, station.station_id, init_utc)

    def _append_month(
        self,
        station_id: str,
        init_utc: pd.Timestamp,
        df: pd.DataFrame,
    ) -> int:
        path = self.path_for(station_id, init_utc)
        before_keys: set[tuple[pd.Timestamp, int]] = set()
        if path.exists():
            existing = pd.read_parquet(path)
            if not existing.empty:
                existing = existing.copy()
                existing["init_time"] = pd.to_datetime(existing["init_time"], utc=True)
                before_keys = set(
                    zip(existing["init_time"], existing["fxx"].astype(int))
                )
                combined = pd.concat([existing, df], ignore_index=True)
            else:
                combined = df.copy()
        else:
            combined = df.copy()

        combined = combined.copy()
        combined["init_time"] = pd.to_datetime(combined["init_time"], utc=True)
        combined["valid_time"] = pd.to_datetime(combined["valid_time"], utc=True)
        combined["fxx"] = combined["fxx"].astype(int)
        combined = (
            combined.sort_values(["init_time", "fxx"])
            .drop_duplicates(subset=["init_time", "fxx"], keep="last")
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        after_keys = set(zip(combined["init_time"], combined["fxx"].astype(int)))
        return len(after_keys - before_keys)

    @staticmethod
    def _validate(
        df: pd.DataFrame,
        station_id: str,
        init_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        if "fxx" not in df.columns:
            raise ValueError(
                f"decoder cache rows missing fxx for station={station_id} "
                f"init={init_utc}"
            )
        decoder = df[df["fxx"].astype(int).between(1, 48)].copy()
        decoder = decoder.sort_values("fxx").reset_index(drop=True)
        actual = set(decoder["fxx"].astype(int).tolist())
        if actual != set(range(1, 49)):
            raise ValueError(
                f"decoder fxx must be exactly {{1..48}} for "
                f"station={station_id} init={init_utc}; got {sorted(actual)}"
            )
        if "TMP_2m" not in decoder.columns or decoder["TMP_2m"].isna().any():
            raise ValueError(
                f"decoder TMP_2m must be present and non-null for "
                f"station={station_id} init={init_utc}"
            )
        return decoder


class TrainingDatasetBuilder:
    """Build TFT training frames from bootstrap stores plus decoder cache."""

    def __init__(
        self,
        data_manager: Any,
        decoder_cache: TrainingDecoderCache,
        encoder_hours: int = MAX_ENCODER_LENGTH,
    ) -> None:
        self._data_manager = data_manager
        self._decoder_cache = decoder_cache
        self._encoder_hours = encoder_hours
        self.skipped: list[dict[str, Any]] = []

    def build_frame(
        self,
        station: Any,
        init_times: list[pd.Timestamp],
        split: str,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        self.skipped = []
        for init_time in init_times:
            init_utc = _to_utc(init_time)
            status = self._data_manager.inference_readiness(
                station, init_utc, self._encoder_hours
            )
            if not status.ready:
                self.skipped.append(
                    {
                        "init_time": init_utc,
                        "split": split,
                        "reason": status.reason,
                        "first_eligible_init": status.first_eligible_init,
                    }
                )
                continue
            try:
                encoder = self._data_manager.build_encoder_obs_window(
                    station, init_utc, n_hours=self._encoder_hours
                )
                decoder = self._decoder_cache.get(station, init_utc)
                decoder = self._data_manager.build_training_decoder_window(
                    station, decoder
                )
            except InsufficientEncoderData as exc:
                self.skipped.append(
                    {
                        "init_time": init_utc,
                        "split": split,
                        "reason": type(exc).__name__,
                        "first_eligible_init": None,
                    }
                )
                continue

            encoder = add_model_features(
                encoder, start_idx=0, init_utc=init_utc,
                station_id=station.station_id,
            )
            decoder = add_model_features(
                decoder, start_idx=self._encoder_hours, init_utc=init_utc,
                station_id=station.station_id,
            )
            combined = pd.concat([encoder, decoder], ignore_index=True)
            combined["split"] = split
            frames.append(combined)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
