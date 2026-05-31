"""DataManager: priority-ordered data fetching with ParquetStore caching.

Coordinates multiple HRRR forecast sources (S3 -> NOMADS -> Zarr ->
Herbie) and multiple obs sources (IEM -> NOAA file) so that callers
never need to know which source is actually serving the data.

Each successful fetch is appended to the corresponding ParquetStore,
enabling fast historical replay without re-downloading.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

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
    """

    def __init__(
        self,
        forecast_sources: "list[BaseForecastSource]",
        obs_sources:      "list[BaseObsSource]",
        forecast_store:   "ParquetStore",
        obs_store:        "ParquetStore",
    ) -> None:
        self._forecast_sources = forecast_sources
        self._obs_sources      = obs_sources
        self._forecast_store   = forecast_store
        self._obs_store        = obs_store

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
                    if not self._obs_is_complete(df):
                        logger.warning(
                            "Obs source {} returned incomplete data for "
                            "station={}, trying next source",
                            src_name, station.station_id,
                        )
                        continue
                    df = Cleaner.process_iem(df)
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
        """Return True if df has at least one non-null reading for each required field.

        Required fields: raw_tmpc_tenths, obs_WS or (obs_U and obs_V), skyc1, p01i.
        A single NOAA row that only contains temperature must not pass this gate.
        """
        if "raw_tmpc_tenths" not in df.columns or not df["raw_tmpc_tenths"].notna().any():
            return False

        has_wind = (
            ("obs_WS" in df.columns and df["obs_WS"].notna().any())
            or (
                "obs_U" in df.columns and "obs_V" in df.columns
                and df["obs_U"].notna().any() and df["obs_V"].notna().any()
            )
        )
        if not has_wind:
            return False

        if "skyc1" not in df.columns or not df["skyc1"].notna().any():
            return False

        if "p01i" not in df.columns or not df["p01i"].notna().any():
            return False

        return True

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

        Parameters
        ----------
        station:
            Target station.
        init_time:
            Model initialisation time (UTC).  The window covers
            [init_time - n_hours, init_time - 1h], i.e. hours -n_hours..-1
            relative to init.  For the default n_hours=72 this is hours
            -72..-1 (72 hourly rows).  The IEM observation at init_time
            itself is not included because it has not yet been fully
            quality-controlled when the model runs.
        n_hours:
            Length of the lookback window in hours (default 72).

        Returns
        -------
        Merged DataFrame with at least valid_time, TMP_2m, and raw_tmpc_tenths
        columns.  Rows are sorted by valid_time.

        Raises
        ------
        RuntimeError
            When fewer than n_hours rows are available in either the HRRR
            or obs store after the merge.
        """
        settings = _Settings()
        cold_start = settings.cold_start_hours

        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )
        # Read cold_start_hours extra so that after dropping the cold-start
        # period the remaining window is at least n_hours deep.
        window_start = init_utc - pd.Timedelta(hours=n_hours + cold_start)
        window_end   = init_utc - pd.Timedelta(hours=1)

        hrrr_raw = self._forecast_store.read(
            station.station_id, window_start, window_end
        )
        obs_raw = self._obs_store.read(
            station.station_id, window_start, window_end
        )

        # Drop cold-start rows: the first cold_start_hours rows of the
        # station's history are unreliable and must not reach the model.
        hrrr_raw = Cleaner.drop_cold_start(hrrr_raw, cold_start)
        obs_raw  = Cleaner.drop_cold_start(obs_raw,  cold_start)

        if len(hrrr_raw) < n_hours:
            raise RuntimeError(
                f"build_encoder_obs_window: insufficient HRRR rows for "
                f"station={station.station_id} "
                f"(have {len(hrrr_raw)}, need {n_hours})"
            )
        if len(obs_raw) < n_hours:
            raise RuntimeError(
                f"build_encoder_obs_window: insufficient obs rows for "
                f"station={station.station_id} "
                f"(have {len(obs_raw)}, need {n_hours})"
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
            raise RuntimeError(
                f"build_encoder_obs_window: merged window too short for "
                f"station={station.station_id} "
                f"(have {len(merged)}, need {n_hours})"
            )

        if "TMP_2m" in merged.columns and (
            "raw_tmpc_tenths" in merged.columns or "tmpc" in merged.columns
        ):
            merged = Cleaner.compute_temp_error(merged)
            merged = merged.dropna(subset=["temp_error"]).reset_index(drop=True)

        if len(merged) < n_hours:
            raise RuntimeError(
                f"build_encoder_obs_window: temp_error NaN removal left too few rows for "
                f"station={station.station_id} "
                f"(have {len(merged)}, need {n_hours})"
            )

        return merged.tail(n_hours).reset_index(drop=True)
