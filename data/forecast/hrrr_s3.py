"""HRRR S3 forecast source.

Primary data source.  Downloads GRIB2 files via byte-range requests from
the NOAA public S3 bucket (noaa-hrrr-bdp-pds.s3.amazonaws.com).

Covers full history (Dec 2024 onwards).
"""
from __future__ import annotations

from data.forecast._grib_source import BaseGribSource

_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


class HrrrS3Source(BaseGribSource):
    """Fetch HRRR forecasts from the NOAA public S3 bucket.

    URL pattern:
        https://noaa-hrrr-bdp-pds.s3.amazonaws.com/
            hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFX.grib2
    """

    def _build_url(self, date_str: str, init_hh: str, fxx_str: str) -> str:
        return (
            f"{_S3_BASE}/hrrr.{date_str}/conus/"
            f"hrrr.t{init_hh}z.wrfsfcf{fxx_str}.grib2"
        )
