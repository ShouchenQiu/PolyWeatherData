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

from data.forecast.hrrr_fill import MAJOR_HOURS, nearest_prior_major
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
        progress: bool = False,
        print_every: int = 10,
        write_batch_size: int = 24,
    ) -> dict[str, Any]:
        """Fetch and persist missing decoder windows for a list of init times.

        Existing monthly-cache rows are skipped.  Missing windows are downloaded
        in parallel and appended to monthly parquet files in one batched write
        per month, keeping download concurrency separate from file IO.
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
        write_batch_count = max(1, int(write_batch_size))
        frames: list[pd.DataFrame] = []

        if progress:
            print(
                "TrainingDecoderCache: "
                f"{len(init_utcs)} inits total, {len(init_utcs) - len(pending)} "
                f"already cached, {len(pending)} pending.",
                flush=True,
            )

        if pending and self._source_has_window_cache():
            self._warm_fill_dependencies(station, pending, worker_count, progress)

        pool = ThreadPoolExecutor(max_workers=worker_count)
        try:
            future_map = {
                pool.submit(self._fetch_clean_validate, station, init_utc): init_utc
                for init_utc in pending
            }
            for fut in as_completed(future_map):
                init_utc = future_map[fut]
                try:
                    frames.append(fut.result())
                    fetched += 1
                    if len(frames) >= write_batch_count:
                        rows_written += self._append_many(station.station_id, frames)
                        frames = []
                except Exception as exc:  # noqa: BLE001
                    error = {"init_time": str(init_utc), "error": str(exc)}
                    errors.append(error)
                    if progress and len(errors) <= 5:
                        print(
                            f"  ERROR decoder fetch {init_utc}: {exc}",
                            flush=True,
                        )

                if progress and (
                    fetched + len(errors) <= 5
                    or (fetched + len(errors)) % max(1, int(print_every)) == 0
                    or fetched + len(errors) == len(pending)
                ):
                    elapsed_so_far = time.perf_counter() - started
                    completed = fetched + len(errors)
                    rate = completed / max(elapsed_so_far, 0.001)
                    remaining = len(pending) - completed
                    eta_min = remaining / rate / 60.0 if rate > 0 else float("inf")
                    print(
                        f"[decoder {completed}/{len(pending)}] "
                        f"rate={rate:.2f} init/s ETA={eta_min:.1f}m "
                        f"errors={len(errors)}",
                        flush=True,
                    )
        except BaseException:
            if frames:
                rows_written += self._append_many(station.station_id, frames)
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        if frames:
            rows_written += self._append_many(station.station_id, frames)

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

    def _source_has_window_cache(self) -> bool:
        return hasattr(self.source, "_full_window_cache")

    def _warm_fill_dependencies(
        self,
        station: Any,
        pending: list[pd.Timestamp],
        workers: int,
        progress: bool,
    ) -> None:
        major_inits: set[pd.Timestamp] = set()
        for init_utc in pending:
            if init_utc.hour in MAJOR_HOURS:
                major_inits.add(init_utc)
                continue
            nearest = nearest_prior_major(init_utc)
            major_inits.add(nearest)
            major_inits.add(nearest - pd.Timedelta(hours=6))

        ordered = sorted(major_inits)
        if progress:
            print(
                f"TrainingDecoderCache: warming {len(ordered)} major windows "
                "for off-hour fill.",
                flush=True,
            )
        worker_count = max(1, min(int(workers), 4))
        started = time.perf_counter()
        completed = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(self.source.fetch, init_utc, range(0, 49), station): init_utc
                for init_utc in ordered
            }
            for fut in as_completed(future_map):
                init_utc = future_map[fut]
                completed += 1
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    if progress:
                        print(
                            f"  WARN dependency warm failed {init_utc}: {exc}",
                            flush=True,
                        )
                if progress and (
                    completed <= 3
                    or completed % 5 == 0
                    or completed == len(ordered)
                ):
                    elapsed = time.perf_counter() - started
                    rate = completed / max(elapsed, 0.001)
                    remaining = len(ordered) - completed
                    eta_min = remaining / rate / 60.0 if rate > 0 else float("inf")
                    print(
                        f"[decoder warm {completed}/{len(ordered)}] "
                        f"rate={rate:.2f} major/s ETA={eta_min:.1f}m "
                        f"errors={errors}",
                        flush=True,
                    )

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
        return self._append_many(station_id, [df])

    def _append_many(self, station_id: str, frames: list[pd.DataFrame]) -> int:
        if not frames:
            return 0
        combined_new = pd.concat(
            [frame.reset_index(drop=True) for frame in frames],
            ignore_index=True,
            join="outer",
        )
        combined_new = combined_new.copy()
        combined_new["init_time"] = pd.to_datetime(
            combined_new["init_time"], utc=True
        )
        combined_new["valid_time"] = pd.to_datetime(
            combined_new["valid_time"], utc=True
        )
        combined_new["fxx"] = combined_new["fxx"].astype(int)

        rows_written = 0
        for _, month_df in combined_new.groupby(
            combined_new["init_time"].dt.strftime("%Y-%m"), sort=True
        ):
            first_init = pd.Timestamp(month_df["init_time"].iloc[0])
            path = self.path_for(station_id, first_init)
            rows_written += self._append_month_frame(path, month_df)
        return rows_written

    @staticmethod
    def _append_month_frame(path: Path, df: pd.DataFrame) -> int:
        before_keys: set[tuple[pd.Timestamp, int]] = set()
        if path.exists():
            existing = pd.read_parquet(path).reset_index(drop=True)
            if not existing.empty:
                existing = existing.copy()
                existing["init_time"] = pd.to_datetime(existing["init_time"], utc=True)
                before_keys = set(
                    zip(existing["init_time"], existing["fxx"].astype(int))
                )
                combined = pd.concat(
                    [existing, df.reset_index(drop=True)],
                    ignore_index=True,
                    join="outer",
                )
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
