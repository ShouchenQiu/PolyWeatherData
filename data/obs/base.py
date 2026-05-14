from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from config.stations import StationConfig


class BaseObsSource(ABC):
    """Abstract base for observation data sources."""

    @abstractmethod
    def fetch(
        self,
        station: "StationConfig",
        start: pd.Timestamp,
        end: pd.Timestamp,
        now_utc: "pd.Timestamp | None" = None,
    ) -> pd.DataFrame:
        """Fetch surface observations for a station over a time window.

        Returns a DataFrame with columns:
            valid_time (UTC-aware), raw_tmpc_tenths, raw_dwpc_tenths,
            raw_wind_spd_kt, raw_wind_gust_kt, obs_U, obs_V,
            raw_vis_sm, raw_alti_inhg, skyc1, skyl1, p01i,
            is_rain, is_snow

        Missing values are NaN.  Rows are in ascending valid_time order.

        Parameters
        ----------
        station:
            Target station configuration.
        start:
            Inclusive start of the requested window (UTC).
        end:
            Inclusive end of the requested window (UTC).
        """
