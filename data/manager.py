"""DataManager: priority-ordered data fetching with ParquetStore caching.

Coordinates multiple HRRR forecast sources (S3 -> NOMADS -> Zarr ->
Herbie) and multiple obs sources (IEM -> NOAA file) so that callers
never need to know which source is actually serving the data.

Each successful fetch is appended to the corresponding ParquetStore,
enabling fast historical replay without re-downloading.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
from loguru import logger

from config.settings import Settings as _Settings
from data.storage.cleaner import Cleaner

if TYPE_CHECKING:
    from config.stations import StationConfig
    from data.forecast.base import BaseForecastSource
    from data.obs.base import BaseObsSource
    from data.storage.parquet_store import ParquetStore

_DEFAULT_FXX_RANGE = range(0, 49)

# Sentinel start date for historical store reads in inference_readiness().
# Far enough back to pre-date any real station data.
_EARLIEST_DATA_START = pd.Timestamp("2020-01-01", tz="UTC")


class InsufficientEncoderData(RuntimeError):
    """Raised when the encoder window cannot be fully populated from the Parquet store.

    Callers (ForecastLoop, BacktestEngine) should call
    ``DataManager.inference_readiness()`` before ``build_encoder_obs_window()``
    to avoid this exception in steady-state operation.
    """


class ReadinessStatus(NamedTuple):
    """Return value of ``DataManager.inference_readiness()``.

    Attributes
    ----------
    ready:
        True only when the full encoder window is populated and all
        required columns are present.  The caller may safely call
        ``build_encoder_obs_window()`` without expecting
        ``InsufficientEncoderData``.
    reason:
        One of:
          ``"ok"``                    -- encoder window is complete.
          ``"cold_start"``            -- not enough historical data yet;
                                         ``first_eligible_init`` is the
                                         earliest init that *will* be ready.
          ``"incomplete_hrrr_window"`` -- HRRR rows < encoder_hours.
          ``"incomplete_obs_window"``  -- obs rows < encoder_hours.
          ``"missing_required_columns"`` -- columns absent after window read.
    first_eligible_init:
        Estimated earliest ``init_time`` for which readiness would return
        ``ready=True`` for this station.  Derived as
        ``first_clean_row_ts + encoder_hours``.  ``None`` when the store
        is completely empty and no estimate is possible.
    """
    ready:               bool
    reason:              str
    first_eligible_init: "pd.Timestamp | None"


class DataManager:
    """Fetch HRRR forecasts and surface observations with automatic fallback.

    Parameters
    ----------
    forecast_sources:
        HRRR sources tried in order from index 0 (highest priority, e.g.
        S3) to the last entry (lowest priority, e.g. Herbie).
    obs_sources:
        Obs sources tried in order (e.g. IEM first, then NOAA file).
    forecast_store:
        ParquetStore used to cache and retrieve HRRR data.
    obs_store:
        ParquetStore used to cache and retrieve obs data.
    settings:
        Optional Settings instance.  When None, a default Settings() is
        constructed at init time.  Inject explicitly in tests that need to
        control cold_start_hours.
    """

    def __init__(
        self,
        forecast_sources: "list[BaseForecastSource]",
        obs_sources:      "list[BaseObsSource]",
        forecast_store:   "ParquetStore",
        obs_store:        "ParquetStore",
        settings:         "_Settings | None" = None,
    ) -> None:
        self._forecast_sources = forecast_sources
        self._obs_sources      = obs_sources
        self._forecast_store   = forecast_store
        self._obs_store        = obs_store
        self._settings         = settings if settings is not None else _Settings()

    # ------------------------------------------------------------------
    # HRRR forecast
    # ------------------------------------------------------------------

    def fetch_hrrr(
        self,
        init_time: pd.Timestamp,
        station:   "StationConfig",
    ) -> pd.DataFrame:
        """Fetch HRRR data, trying each source in priority order.

        On success the result is appended to forecast_store for future
        offline replay.  Always fetches fxx 0..48 (the full decoder window).

        Parameters
        ----------
        init_time:
            HRRR model initialisation time (UTC).
        station:
            Target station configuration.

        Returns
        -------
        DataFrame with columns init_time, fxx, valid_time + HRRR vars.

        Raises
        ------
        RuntimeError
            If every source fails.
        """
        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )

        last_exc: Exception | None = None
        for src in self._forecast_sources:
            src_name = type(src).__name__
            try:
                result = src.fetch(init_utc, _DEFAULT_FXX_RANGE, station)
                # Sources return dict[station_id -> DataFrame]
                df = result.get(station.station_id, pd.DataFrame())
                if df is not None and not df.empty:
                    if not self._hrrr_is_complete(df):
                        logger.warning(
                            "HRRR source {} returned incomplete data for "
                            "init={}, trying next source",
                            src_name, init_utc,
                        )
                        continue
                    df = Cleaner.process_hrrr(df)
                    logger.debug(
                        "HRRR fetch: source={} init={} rows={}",
                        src_name, init_utc, len(df),
                    )
                    self._forecast_store.append(station.station_id, df)
                    return df
            except Exception as exc:
                logger.warning(
                    "HRRR source {} failed for init={}: {}", src_name, init_utc, exc
                )
                last_exc = exc
                continue
        raise RuntimeError(
            f"All HRRR sources failed for init={init_utc}. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _hrrr_is_complete(df: pd.DataFrame) -> bool:
        """Return True if df covers the full decoder window (fxx 1..48, TMP_2m non-null).

        A non-MAJOR splice that failed to obtain the nearest MAJOR run returns
        only fxx 0..18.  This check rejects that short window so the source
        fallback loop can try the next ForecastSource instead of propagating
        a truncated decoder frame to ModelPipeline.
        """
        if "fxx" not in df.columns or "TMP_2m" not in df.columns:
            return False
        decoder_rows = df[df["fxx"].between(1, 48)]
        missing_steps = set(range(1, 49)) - set(decoder_rows["fxx"].unique())
        if missing_steps:
            return False
        return bool(decoder_rows["TMP_2m"].notna().all())

    # ------------------------------------------------------------------
    # Surface observations
    # ------------------------------------------------------------------

    def fetch_iem(
        self,
        station: "StationConfig",
        start:   pd.Timestamp,
        end:     pd.Timestamp,
        now_utc: "pd.Timestamp | None" = None,
    ) -> pd.DataFrame:
        """Fetch obs data, trying each source in priority order.

        On success the result is appended to obs_store.

        Parameters
        ----------
        station:
            Target station configuration.
        start:
            Inclusive start of the requested window (UTC).
        end:
            Inclusive end of the requested window (UTC).
        now_utc:
            When provided (tick mode), IEM source uses only the :51 reading
            for H-1 instead of the full [start, end] window.

        Returns
        -------
        DataFrame with columns valid_time + obs variables.

        Raises
        ------
        RuntimeError
            If every source fails.
        """
        last_exc: Exception | None = None
        for src in self._obs_sources:
            src_name = type(src).__name__
            try:
                df = src.fetch(station, start, end, now_utc)
                if df is not None and not df.empty:
                    df = Cleaner.process_iem(df)
                    if not self._obs_is_complete(df):
                        logger.warning(
                            "Obs source {} returned incomplete data for "
                            "station={}, trying next source",
                            src_name, station.station_id,
                        )
                        continue
                    logger.debug(
                        "Obs fetch: source={} station={} rows={}",
                        src_name, station.station_id, len(df),
                    )
                    self._obs_store.append(station.station_id, df)
                    return df
            except Exception as exc:
                logger.warning(
                    "Obs source {} failed for station={}: {}",
                    src_name, station.station_id, exc,
                )
                last_exc = exc
                continue
        raise RuntimeError(
            f"All obs sources failed for station={station.station_id} "
            f"{start}..{end}. Last error: {last_exc}"
        )

    def fetch_latest_iem_tick(
        self,
        station: "StationConfig",
        now_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        """Append the single H-1 :51 obs reading for a live tick cycle.

        Delegates to fetch_iem with now_utc set so that IemSource uses the
        narrow tick window instead of pulling the full historical range.
        """
        return self.fetch_iem(station, now_utc, now_utc, now_utc=now_utc)

    @staticmethod
    def _obs_is_complete(df: pd.DataFrame) -> bool:
        """Return True if at least one row has all required fields non-null simultaneously.

        Must be called on fully processed data (after ``Cleaner.process_iem()``,
        which chains ``clean_iem()`` + ``build_features_iem()``).  After
        ``process_iem()``, ``skyc1``, ``p01i``, and wind columns are
        zero-filled; the meaningful per-row check is whether
        ``raw_tmpc_tenths`` is non-null in the same row as the other
        required columns.

        Wind column resolution (row-level):
        - ``obs_WS`` alone satisfies the wind requirement (scalar speed
          produced by ``build_features_iem()``).
        - Otherwise ``obs_U`` AND ``obs_V`` must both be non-null in the
          same row (``obs_U`` alone is not sufficient).
        """
        required = ["raw_tmpc_tenths", "skyc1", "p01i"]
        if not all(c in df.columns for c in required):
            return False

        has_ws = "obs_WS" in df.columns
        has_uv = "obs_U" in df.columns and "obs_V" in df.columns
        if not has_ws and not has_uv:
            return False

        if has_ws:
            wind_mask = df["obs_WS"].notna()
        else:
            wind_mask = df["obs_U"].notna() & df["obs_V"].notna()

        row_mask = (
            df["raw_tmpc_tenths"].notna()
            & df["skyc1"].notna()
            & df["p01i"].notna()
            & wind_mask
        )
        return bool(row_mask.any())

    # ------------------------------------------------------------------
    # Encoder window builder
    # ------------------------------------------------------------------

    def build_encoder_obs_window(
        self,
        station: "StationConfig",
        init_time: pd.Timestamp,
        n_hours: int = 72,
    ) -> pd.DataFrame:
        """Read the last n_hours of f01 HRRR and IEM obs; merge on valid_time.

        Reads directly from the ParquetStores -- no network fetch.

        The window is **exactly** ``[init_time - n_hours, init_time - 1h]``
        (i.e. ``n_hours`` hourly rows).  No cold-start extension is applied;
        cold-start burn-in is an offline / batch concern handled by
        ``Cleaner.drop_cold_start()``.  Live callers should gate on
        ``inference_readiness()`` before calling this method.

        Parameters
        ----------
        station:
            Target station.
        init_time:
            Model initialisation time (UTC).  The window covers
            ``[init_time - n_hours, init_time - 1h]``, i.e.
            ``n_hours`` hourly rows ending one hour before init.
        n_hours:
            Length of the lookback window in hours (default 72,
            matching ``MAX_ENCODER_LENGTH`` [ADR-0001]).

        Returns
        -------
        Merged DataFrame with at least valid_time, TMP_2m, raw_tmpc_tenths,
        and temp_error columns.  Rows are sorted by valid_time.

        Raises
        ------
        InsufficientEncoderData
            When fewer than n_hours rows are available in either store,
            or when the merged window is too short after the asof join and
            temp_error NaN removal.
        RuntimeError
            When required columns (valid_time, TMP_2m, raw_tmpc_tenths) are
            absent from the store data, or when temp_error cannot be computed.
        """
        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )
        window_start = init_utc - pd.Timedelta(hours=n_hours)
        window_end   = init_utc - pd.Timedelta(hours=1)

        # .copy() immediately after store reads to avoid shared-reference
        # side effects when the caller or Cleaner mutates the DataFrame.
        hrrr_raw = self._forecast_store.read(
            station.station_id, window_start, window_end
        ).copy()
        obs_raw = self._obs_store.read(
            station.station_id, window_start, window_end
        ).copy()

        if len(hrrr_raw) < n_hours:
            raise InsufficientEncoderData(
                f"build_encoder_obs_window: insufficient HRRR rows for "
                f"station={station.station_id} "
                f"(have {len(hrrr_raw)}, need {n_hours})"
            )
        if len(obs_raw) < n_hours:
            raise InsufficientEncoderData(
                f"build_encoder_obs_window: insufficient obs rows for "
                f"station={station.station_id} "
                f"(have {len(obs_raw)}, need {n_hours})"
            )

        # Required-column guard: raise early with an actionable message.
        missing_hrrr = [c for c in ("valid_time", "TMP_2m") if c not in hrrr_raw.columns]
        missing_obs  = [c for c in ("valid_time", "raw_tmpc_tenths") if c not in obs_raw.columns]
        if missing_hrrr or missing_obs:
            raise RuntimeError(
                f"build_encoder_obs_window: required columns missing for "
                f"station={station.station_id} -- "
                f"HRRR missing: {missing_hrrr}, obs missing: {missing_obs}"
            )

        hrrr_raw["valid_time"] = pd.to_datetime(hrrr_raw["valid_time"], utc=True)
        obs_raw["valid_time"]  = pd.to_datetime(obs_raw["valid_time"],  utc=True)

        hrrr_sorted = hrrr_raw.sort_values("valid_time").reset_index(drop=True)
        obs_sorted  = obs_raw.sort_values("valid_time").reset_index(drop=True)

        # Asof join with 20-minute tolerance: HRRR rows where no obs lands within
        # 20 min get NaN obs columns; temp_error dropna below discards those rows.
        merged = pd.merge_asof(
            hrrr_sorted,
            obs_sorted,
            on="valid_time",
            tolerance=pd.Timedelta(minutes=20),
            direction="nearest",
            suffixes=("_hrrr", "_obs"),
        )
        merged = merged.sort_values("valid_time").reset_index(drop=True)

        if len(merged) < n_hours:
            raise InsufficientEncoderData(
                f"build_encoder_obs_window: merged window too short for "
                f"station={station.station_id} "
                f"(have {len(merged)}, need {n_hours})"
            )

        if "TMP_2m" in merged.columns and (
            "raw_tmpc_tenths" in merged.columns or "tmpc" in merged.columns
        ):
            merged = Cleaner.compute_temp_error(merged)
            merged = merged.dropna(subset=["temp_error"]).reset_index(drop=True)

        if "temp_error" not in merged.columns:
            raise RuntimeError(
                f"build_encoder_obs_window: temp_error not produced for "
                f"station={station.station_id}; check that TMP_2m and "
                f"raw_tmpc_tenths are both present after merge"
            )

        if len(merged) < n_hours:
            raise InsufficientEncoderData(
                f"build_encoder_obs_window: temp_error NaN removal left too few rows for "
                f"station={station.station_id} "
                f"(have {len(merged)}, need {n_hours})"
            )

        return merged.tail(n_hours).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Inference readiness gate
    # ------------------------------------------------------------------

    def inference_readiness(
        self,
        station:       "StationConfig",
        init_time:     pd.Timestamp,
        encoder_hours: int = 72,
    ) -> ReadinessStatus:
        """Check whether the encoder window for this station/init_time is ready.

        Reads the exact encoder window ``[init_time - encoder_hours, init_time - 1h]``
        from both stores.  When either window is short it performs a secondary
        historical read to determine whether the shortfall is a cold-start state
        (not enough data has been collected yet) or a mid-history gap.

        Does NOT raise; always returns a ``ReadinessStatus``.  Callers that only
        need a boolean may use the thinner ``is_inference_ready()`` wrapper.

        Does NOT call ``Cleaner.drop_cold_start()``; cold-start burn-in is an
        offline / batch concern and must not be applied here to avoid
        double-counting (ADR-0001, grill-tasks-v2.md #3).

        Parameters
        ----------
        station:
            Target station.
        init_time:
            Model initialisation time (UTC).
        encoder_hours:
            Encoder lookback window length in hours (default 72).

        Returns
        -------
        ReadinessStatus with ready, reason, and first_eligible_init.
        """
        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )
        window_start = init_utc - pd.Timedelta(hours=encoder_hours)
        window_end   = init_utc - pd.Timedelta(hours=1)

        hrrr_window = self._forecast_store.read(
            station.station_id, window_start, window_end
        )
        obs_window = self._obs_store.read(
            station.station_id, window_start, window_end
        )

        # --- Check required columns when data is present ---
        # Both valid_time and the temperature column must be present;
        # absence of either would cause build_encoder_obs_window to fail.
        _hrrr_required = {"valid_time", "TMP_2m"}
        _obs_required  = {"valid_time", "raw_tmpc_tenths"}
        if not hrrr_window.empty and not _hrrr_required.issubset(hrrr_window.columns):
            return ReadinessStatus(
                ready=False,
                reason="missing_required_columns",
                first_eligible_init=None,
            )
        if not obs_window.empty and not _obs_required.issubset(obs_window.columns):
            return ReadinessStatus(
                ready=False,
                reason="missing_required_columns",
                first_eligible_init=None,
            )

        cold_start = self._settings.cold_start_hours

        # --- Check HRRR encoder window coverage ---
        if len(hrrr_window) < encoder_hours:
            first_eligible = self._estimate_first_eligible(
                self._forecast_store, station.station_id, window_start,
                encoder_hours, current_window=hrrr_window,
                cold_start_hours=cold_start,
            )
            if first_eligible is None or first_eligible > init_utc:
                return ReadinessStatus(
                    ready=False, reason="cold_start",
                    first_eligible_init=first_eligible,
                )
            return ReadinessStatus(
                ready=False, reason="incomplete_hrrr_window",
                first_eligible_init=first_eligible,
            )

        # --- Check obs encoder window coverage ---
        if len(obs_window) < encoder_hours:
            first_eligible = self._estimate_first_eligible(
                self._obs_store, station.station_id, window_start,
                encoder_hours, current_window=obs_window,
                cold_start_hours=cold_start,
            )
            if first_eligible is None or first_eligible > init_utc:
                return ReadinessStatus(
                    ready=False, reason="cold_start",
                    first_eligible_init=first_eligible,
                )
            return ReadinessStatus(
                ready=False, reason="incomplete_obs_window",
                first_eligible_init=first_eligible,
            )

        # --- Both windows are sufficient ---
        # Enforce cold-start gate based on the absolute first row in the full
        # store, not just the encoder-window min.  When data collection started
        # fewer than cold_start_hours + encoder_hours ago, the encoder window may
        # be full (exactly 72 rows) even though burn-in is not complete.
        first_hrrr = self._estimate_first_eligible(
            self._forecast_store, station.station_id, window_start,
            encoder_hours, current_window=hrrr_window,
            cold_start_hours=cold_start,
        )
        first_obs = self._estimate_first_eligible(
            self._obs_store, station.station_id, window_start,
            encoder_hours, current_window=obs_window,
            cold_start_hours=cold_start,
        )
        if first_hrrr is not None and first_obs is not None:
            first_eligible = max(first_hrrr, first_obs)
        elif first_hrrr is not None:
            first_eligible = first_hrrr
        elif first_obs is not None:
            first_eligible = first_obs
        else:
            first_eligible = None

        if first_eligible is not None and init_utc < first_eligible:
            return ReadinessStatus(
                ready=False, reason="cold_start",
                first_eligible_init=first_eligible,
            )
        return ReadinessStatus(ready=True, reason="ok", first_eligible_init=first_eligible)

    def is_inference_ready(
        self,
        station:       "StationConfig",
        init_time:     pd.Timestamp,
        encoder_hours: int = 72,
    ) -> bool:
        """Thin boolean wrapper over ``inference_readiness()``.

        Returns True only when the encoder window is ready to use.
        """
        return self.inference_readiness(station, init_time, encoder_hours).ready

    # ------------------------------------------------------------------
    # Private helpers for inference_readiness
    # ------------------------------------------------------------------

    @staticmethod
    def _min_valid_time(df: pd.DataFrame) -> "pd.Timestamp | None":
        """Return the minimum valid_time in df as a UTC Timestamp, or None."""
        if df.empty or "valid_time" not in df.columns:
            return None
        try:
            return pd.to_datetime(df["valid_time"], utc=True).min()
        except Exception:
            return None

    @staticmethod
    def _estimate_first_eligible(
        store:            "ParquetStore",
        station_id:       str,
        window_start:     pd.Timestamp,
        encoder_hours:    int,
        current_window:   pd.DataFrame,
        cold_start_hours: int = 0,
    ) -> "pd.Timestamp | None":
        """Return first_eligible_init = first_store_row_ts + cold_start_hours + encoder_hours.

        Combines two data sources to find the absolute earliest available row:
          1. Secondary historical read: ``_EARLIEST_DATA_START .. window_start - 1h``.
          2. ``current_window``: the partial encoder-window DataFrame already read
             by the caller.  This covers the typical cold-start shape where the
             very first rows are *inside* the encoder window (< 72 rows collected so
             far, so the historical range is empty but partial data exists).

        ``cold_start_hours`` is the live-store burn-in addend.  Because
        ``fetch_hrrr()`` / ``fetch_iem()`` do NOT call
        ``Cleaner.drop_cold_start()`` before appending, the first
        ``cold_start_hours`` rows in the store are unreliable.  The callers
        should pass ``Settings.cold_start_hours`` (default 72) so that
        ``first_eligible_init`` correctly requires
        ``first_store_row + cold_start_hours + encoder_hours`` of
        accumulated data before live inference is permitted.

        Returns None when neither the historical read nor the current
        window contains any valid_time data.
        """
        hist = store.read(
            station_id,
            _EARLIEST_DATA_START,
            window_start - pd.Timedelta(hours=1),
        )
        frames = [
            f for f in (hist, current_window)
            if not f.empty and "valid_time" in f.columns
        ]
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        try:
            first_ts = pd.to_datetime(combined["valid_time"], utc=True).min()
        except Exception:
            return None
        if pd.isna(first_ts):
            return None
        return first_ts + pd.Timedelta(hours=cold_start_hours + encoder_hours)
