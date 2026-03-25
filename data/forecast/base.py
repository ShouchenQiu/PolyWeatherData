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
        station: "StationConfig",
    ) -> pd.DataFrame:
        """Fetch HRRR data for one init cycle.

        Returns a DataFrame with columns:
            init_time, fxx, valid_time, <36 HRRR variable columns>

        Rows are indexed by (init_time, fxx).  Missing values are NaN.

        Parameters
        ----------
        init_time:
            UTC timestamp of the HRRR model initialisation hour.
        fxx_range:
            Forecast hours to fetch, e.g. range(0, 49).
        station:
            Target station configuration (supplies lat/lon and grid indices).
        """
