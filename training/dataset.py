"""Training dataset helpers shared by offline preparation and TFT training.

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

from data.forecast.hrrr_fill import HrrrFill, MAJOR_HOURS, nearest_prior_major
from data.manager import InsufficientEncoderData
from data.storage.cleaner import Cleaner
from model.features import add_model_features
from model.tft import MAX_ENCODER_LENGTH

DECODER_WEIGHT_COLUMN = "decoder_weight"


@dataclass
class TrainingDecoderCache:
    """File-backed cache for full HRRR decoder windows used in training only."""

    root_dir: Path
    source: Any | None = None
    fallback_source: Any | None = None
    slow_init_threshold_s: float = 30.0

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_sources(self) -> None:
        if self.source is None:
            from data.forecast.hrrr_zarr import HrrrZarrSource

            self.source = HrrrZarrSource()
        if self.fallback_source is None:
            from data.forecast.hrrr_s3 import HrrrS3Source

            self.fallback_source = HrrrS3Source()

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

        warm_started = time.perf_counter()
        warm_summary = {
            "warm_requested_inits": 0,
            "warm_completed_inits": 0,
            "warm_errors": 0,
        }
        if pending:
            self._ensure_sources()
        if pending and self._source_has_window_cache():
            warm_summary = self._warm_fill_dependencies(
                station, pending, worker_count, progress
            )
        warm_elapsed = time.perf_counter() - warm_started

        fetch_started = time.perf_counter()
        zarr_success = 0
        fallback_success = 0
        hybrid_fallback_success = 0
        full_fallback_success = 0
        fallback_failed = 0
        zarr_timeout_count = 0
        slow_init_samples: list[dict[str, Any]] = []
        pool = ThreadPoolExecutor(max_workers=worker_count)
        try:
            future_map = {
                pool.submit(self._fetch_clean_validate_with_source, station, init_utc): init_utc
                for init_utc in pending
            }
            for fut in as_completed(future_map):
                init_utc = future_map[fut]
                try:
                    df, source_name, primary_exc, fetch_elapsed = fut.result()
                    frames.append(df)
                    if source_name == "primary":
                        zarr_success += 1
                    else:
                        fallback_success += 1
                        if source_name == "hybrid_fallback":
                            hybrid_fallback_success += 1
                        else:
                            full_fallback_success += 1
                        if primary_exc is not None and _looks_like_timeout(primary_exc):
                            zarr_timeout_count += 1
                        if progress:
                            print(
                                self._fallback_progress_line(
                                    init_utc,
                                    source_name,
                                    primary_exc,
                                    fetch_elapsed,
                                ),
                                flush=True,
                            )
                    if fetch_elapsed >= self.slow_init_threshold_s:
                        slow_init_samples.append(
                            {
                                "init_time": str(init_utc),
                                "elapsed_seconds": fetch_elapsed,
                                "source": source_name,
                            }
                        )
                    fetched += 1
                    if len(frames) >= write_batch_count:
                        rows_written += self._append_many(station.station_id, frames)
                        frames = []
                except Exception as exc:  # noqa: BLE001
                    fallback_failed += 1
                    if _looks_like_timeout(exc):
                        zarr_timeout_count += 1
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
                        f"errors={len(errors)} "
                        f"routes=zarr:{zarr_success} "
                        f"hybrid:{hybrid_fallback_success} "
                        f"full_fallback:{full_fallback_success}",
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
        fetch_elapsed = time.perf_counter() - fetch_started
        return {
            "requested_inits": len(init_utcs),
            "skipped_cached": len(init_utcs) - len(pending),
            "fetched_inits": fetched,
            "rows_written": rows_written,
            "errors": len(errors),
            "error_details": errors,
            "elapsed_seconds": elapsed,
            "warm_elapsed_seconds": warm_elapsed,
            "fetch_elapsed_seconds": fetch_elapsed,
            "init_per_second": fetched / elapsed if elapsed > 0 else 0.0,
            "fetch_init_per_second": fetched / fetch_elapsed if fetch_elapsed > 0 else 0.0,
            "zarr_success_inits": zarr_success,
            "fallback_success_inits": fallback_success,
            "hybrid_fallback_success_inits": hybrid_fallback_success,
            "full_fallback_success_inits": full_fallback_success,
            "fallback_failed_inits": fallback_failed,
            "zarr_timeout_count": zarr_timeout_count,
            "slow_init_samples": slow_init_samples[:10],
            **warm_summary,
            "major_inits": sum(1 for ts in init_utcs if ts.hour in {0, 6, 12, 18}),
            "offhour_inits": sum(1 for ts in init_utcs if ts.hour not in {0, 6, 12, 18}),
        }

    def _fallback_progress_line(
        self,
        init_utc: pd.Timestamp,
        source_name: str,
        primary_exc: Exception | None,
        elapsed_seconds: float,
    ) -> str:
        primary_name = type(self.source).__name__
        fallback_name = type(self.fallback_source).__name__
        error = _compact_exception(primary_exc)
        if source_name == "hybrid_fallback":
            nearest_major_init = nearest_prior_major(init_utc)
            route = (
                f"mode=hybrid "
                f"current={fallback_name}[fxx=0..18] "
                f"tail={primary_name}[nearest_major={nearest_major_init}]"
            )
        else:
            route = (
                f"mode=full_fallback "
                f"source={fallback_name}[fxx=0..48]"
            )
        return (
            f"  FALLBACK decoder init={init_utc} {route} "
            f"primary_error={error} elapsed={elapsed_seconds:.1f}s"
        )

    def _source_has_window_cache(self) -> bool:
        return hasattr(self.source, "_full_window_cache")

    def _warm_fill_dependencies(
        self,
        station: Any,
        pending: list[pd.Timestamp],
        workers: int,
        progress: bool,
    ) -> dict[str, int]:
        major_inits: set[pd.Timestamp] = set()
        for init_utc in pending:
            if init_utc.hour in MAJOR_HOURS:
                major_inits.add(init_utc)
                continue
            nearest = nearest_prior_major(init_utc)
            major_inits.add(nearest)

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
        return {
            "warm_requested_inits": len(ordered),
            "warm_completed_inits": completed,
            "warm_errors": errors,
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
        df, _source_name, _primary_exc, _elapsed = self._fetch_clean_validate_with_source(
            station, init_utc
        )
        return df

    def _fetch_clean_validate_with_source(
        self,
        station: Any,
        init_utc: pd.Timestamp,
    ) -> tuple[pd.DataFrame, str, Exception | None, float]:
        self._ensure_sources()
        started = time.perf_counter()
        try:
            df = self._fetch_clean_validate_from_source(
                self.source, station, init_utc
            )
            return df, "primary", None, time.perf_counter() - started
        except Exception as primary_exc:
            if self.fallback_source is None:
                raise
            if init_utc.hour not in MAJOR_HOURS and self._source_has_window_cache():
                try:
                    df = self._fetch_hybrid_non_major_decoder(station, init_utc)
                    return (
                        df,
                        "hybrid_fallback",
                        primary_exc,
                        time.perf_counter() - started,
                    )
                except Exception:
                    pass
            try:
                df = self._fetch_clean_validate_from_source(
                    self.fallback_source, station, init_utc
                )
                return df, "fallback", primary_exc, time.perf_counter() - started
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"TrainingDecoderCache: primary source failed for "
                    f"station={station.station_id} init={init_utc}: "
                    f"{primary_exc}; fallback source failed: {fallback_exc}"
                ) from fallback_exc

    def _fetch_hybrid_non_major_decoder(
        self,
        station: Any,
        init_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        if self.fallback_source is None:
            raise RuntimeError("hybrid decoder fallback requires a fallback source")
        non_major = self._fetch_clean_from_source(
            self.fallback_source, station, init_utc, range(0, 19)
        )
        nearest_major_init = nearest_prior_major(init_utc)
        nearest_major = self._fetch_clean_from_source(
            self.source, station, nearest_major_init, range(0, 49)
        )
        filled = HrrrFill().fill(non_major, nearest_major)
        return self._validate(filled, station.station_id, init_utc)

    def _fetch_clean_validate_from_source(
        self,
        source: Any,
        station: Any,
        init_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        df = self._fetch_clean_from_source(source, station, init_utc, range(0, 49))
        return self._validate(df, station.station_id, init_utc)

    @staticmethod
    def _fetch_clean_from_source(
        source: Any,
        station: Any,
        init_utc: pd.Timestamp,
        fxx_range: range,
    ) -> pd.DataFrame:
        result = source.fetch(init_utc, fxx_range, station)
        df = result.get(station.station_id, pd.DataFrame())
        if df is None or df.empty:
            raise RuntimeError(
                f"TrainingDecoderCache: no HRRR rows for "
                f"station={station.station_id} init={init_utc}"
            )
        return Cleaner.process_hrrr(df)

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
        if "is_extrapolated" not in decoder.columns:
            decoder["is_extrapolated"] = False
        decoder["is_extrapolated"] = (
            decoder["is_extrapolated"].fillna(False).astype(bool)
        )
        decoder["extrapolation_step"] = 0
        fallback_fxx = sorted(
            decoder.loc[decoder["is_extrapolated"], "fxx"].astype(int).tolist()
        )
        if fallback_fxx:
            expected = list(range(fallback_fxx[0], 49))
            if fallback_fxx != expected:
                raise ValueError(
                    "decoder is_extrapolated rows must form a contiguous tail "
                    f"for station={station_id} init={init_utc}; got {fallback_fxx}"
                )
            decoder.loc[
                decoder["is_extrapolated"],
                "extrapolation_step",
            ] = range(1, len(fallback_fxx) + 1)
        decoder["extrapolation_step"] = decoder["extrapolation_step"].astype(int)
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
            encoder[DECODER_WEIGHT_COLUMN] = 1.0
            extrapolated = (
                decoder["is_extrapolated"].fillna(False).astype(bool)
                if "is_extrapolated" in decoder.columns
                else pd.Series(False, index=decoder.index)
            )
            decoder[DECODER_WEIGHT_COLUMN] = (~extrapolated).astype("float32")
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


def _looks_like_timeout(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text


def _compact_exception(exc: BaseException | None, limit: int = 180) -> str:
    if exc is None:
        return "unknown"
    text = " ".join(str(exc).split())
    rendered = f"{type(exc).__name__}: {text}"
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit - 3]}..."
