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
        fxx_range: range = _DEFAULT_FXX_RANGE,
    ) -> pd.DataFrame:
        """Fetch HRRR data, trying each source in priority order.

        On success the result is appended to forecast_store for future
        offline replay.

        Parameters
        ----------
        init_time:
            HRRR model initialisation time (UTC).
        station:
            Target station configuration.
        fxx_range:
            Forecast hours to retrieve (default 0..48 inclusive).

        Returns
        -------
        DataFrame with columns init_time, fxx, valid_time + HRRR vars.

        Raises
        ------
        RuntimeError
            If every source fails.
        """
        last_exc: Exception | None = None
        for src in self._forecast_sources:
            src_name = type(src).__name__
            try:
                result = src.fetch(init_time, fxx_range, station)
                # Sources return dict[station_id -> DataFrame]
                df = result.get(station.station_id, pd.DataFrame())
                if df is not None and not df.empty:
                    if not self._hrrr_is_complete(df):
                        logger.warning(
                            "HRRR source {} returned incomplete data for "
                            "init={}, trying next source",
                            src_name, init_time,
                        )
                        continue
                    logger.debug(
                        "HRRR fetch: source={} init={} rows={}",
                        src_name, init_time, len(df),
                    )
                    self._forecast_store.append(station.station_id, df)
                    return df
            except Exception as exc:
                logger.warning(
                    "HRRR source {} failed for init={}: {}", src_name, init_time, exc
                )
                last_exc = exc
                continue
        raise RuntimeError(
            f"All HRRR sources failed for init={init_time}. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _hrrr_is_complete(df: pd.DataFrame) -> bool:
        """Return True if the DataFrame has at least one valid TMP_2m reading."""
        if "TMP_2m" not in df.columns:
            return False
        return bool(df["TMP_2m"].notna().any())

    # ------------------------------------------------------------------
    # Surface observations
    # ------------------------------------------------------------------

    def fetch_iem(
        self,
        station: "StationConfig",
        start:   pd.Timestamp,
        end:     pd.Timestamp,
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
                df = src.fetch(station, start, end)
                if df is not None and not df.empty:
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

    @staticmethod
    def _obs_is_complete(df: pd.DataFrame) -> bool:
        """Return True if the DataFrame has at least one valid temperature reading."""
        if "raw_tmpc_tenths" not in df.columns:
            return False
        return bool(df["raw_tmpc_tenths"].notna().any())

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
            [init_time - n_hours, init_time).
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
        window_start = init_utc - pd.Timedelta(hours=n_hours)
        window_end   = init_utc - pd.Timedelta(hours=1)

        hrrr_raw = self._forecast_store.read(
            station.station_id, window_start, window_end
        )
        obs_raw = self._obs_store.read(
            station.station_id, window_start, window_end
        )

        # Filter to fxx=1 only (ADR-0003 stores fxx 0 and 1; we want f01)
        if not hrrr_raw.empty and "fxx" in hrrr_raw.columns:
            hrrr_raw = hrrr_raw[hrrr_raw["fxx"] == 1].reset_index(drop=True)

        if len(hrrr_raw) < cold_start:
            raise RuntimeError(
                f"build_encoder_obs_window: insufficient HRRR rows for "
                f"station={station.station_id} "
                f"(have {len(hrrr_raw)}, need {cold_start})"
            )
        if len(obs_raw) < cold_start:
            raise RuntimeError(
                f"build_encoder_obs_window: insufficient obs rows for "
                f"station={station.station_id} "
                f"(have {len(obs_raw)}, need {cold_start})"
            )

        hrrr_raw["valid_time"] = pd.to_datetime(hrrr_raw["valid_time"], utc=True)
        obs_raw["valid_time"]  = pd.to_datetime(obs_raw["valid_time"],  utc=True)

        merged = pd.merge(hrrr_raw, obs_raw, on="valid_time", how="inner", suffixes=("_hrrr", "_obs"))
        merged = merged.sort_values("valid_time").reset_index(drop=True)

        if len(merged) < cold_start:
            raise RuntimeError(
                f"build_encoder_obs_window: merged window too short for "
                f"station={station.station_id} "
                f"(have {len(merged)}, need {cold_start})"
            )

        return merged

