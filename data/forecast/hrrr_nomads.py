"""HRRR NOMADS forecast source.

Fallback (second in priority) when S3 is unavailable.
Uses the official NOAA NOMADS server which has only ~2-3 days of history.

URL pattern mirrors S3; only the base URL differs.
"""
from __future__ import annotations

from data.forecast._grib_source import BaseGribSource

_NOMADS_BASE = (
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
)


class HrrrNomadsSource(BaseGribSource):
    """Fetch HRRR forecasts from the NOAA NOMADS server.

    URL pattern:
        https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/
            hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFX.grib2

    Notes
    -----
    - Near-real-time only: NOMADS retains ~2-3 days of model output.
    - No concurrency restrictions, but bandwidth is lower than S3.
    - May be marginally more up-to-date than S3 (no sync delay).
    """

    def _build_url(self, date_str: str, init_hh: str, fxx_str: str) -> str:
        return (
            f"{_NOMADS_BASE}/hrrr.{date_str}/conus/"
            f"hrrr.t{init_hh}z.wrfsfcf{fxx_str}.grib2"
        )
