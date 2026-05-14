from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from config.stations import StationConfig


class BaseForecastSource(ABC):
    """Abstract base for HRRR forecast data sources."""

    @abstractmethod
    def fetch(
        self,
        init_time: pd.Timestamp,
        fxx_range: range,
        station: "StationConfig | None" = None,
    ) -> "dict[str, pd.DataFrame]":
        """Fetch HRRR data for one init cycle.

        Returns a dict mapping station_id to a DataFrame with columns:
            init_time, fxx, valid_time, <36 HRRR variable columns>

        Rows are indexed by (init_time, fxx).  Missing values are NaN.

        Parameters
        ----------
        init_time:
            UTC timestamp of the HRRR model initialisation hour.
        fxx_range:
            Forecast hours to fetch, e.g. range(0, 49).
        station:
            Optional single-station override.  When provided, only that
            station's data is returned.  When None, all configured STATIONS
            are returned (production default for GRIB-based sources).
        """
