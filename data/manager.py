"""DataManager: IEM observation fetching with ParquetStore caching.

Each successful fetch is appended to the obs ParquetStore,
enabling fast historical replay without re-downloading.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from config.settings import Settings as _Settings

if TYPE_CHECKING:
    from config.stations import StationConfig
    from data.obs.base import BaseObsSource
    from data.storage.parquet_store import ParquetStore


class DataManager:
    """Fetch surface observations with automatic fallback."""

    def __init__(
        self,
        obs_sources: "list[BaseObsSource]",
        obs_store:   "ParquetStore",
        settings:    "_Settings | None" = None,
    ) -> None:
        self._obs_sources = obs_sources
        self._obs_store   = obs_store
        self._settings    = settings if settings is not None else _Settings()

    def fetch_iem(
        self,
        station: "StationConfig",
        start:   pd.Timestamp,
        end:     pd.Timestamp,
        now_utc: "pd.Timestamp | None" = None,
    ) -> pd.DataFrame:
        """Fetch IEM observations, trying each source in priority order.

        On success the result is appended to obs_store for future replay.
        """
        for src in self._obs_sources:
            try:
                df = src.fetch(station, start, end, now_utc=now_utc)
                if df is not None and not df.empty:
                    logger.debug("OBS fetch: source={} station={} rows={}",
                                 type(src).__name__, station.station_id, len(df))
                    self._obs_store.append(station.station_id, df)
                    return df
            except Exception as exc:
                logger.warning("OBS source {} failed: {}",
                               type(src).__name__, exc)
                continue
        raise RuntimeError(
            f"All obs sources failed for station={station.station_id}"
        )

    @property
    def obs_store(self) -> "ParquetStore":
        return self._obs_store
