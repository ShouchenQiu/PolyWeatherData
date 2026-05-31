"""HRRR S3 forecast source.

Primary data source.  Downloads GRIB2 files via byte-range requests from
the NOAA public S3 bucket (noaa-hrrr-bdp-pds.s3.amazonaws.com).

Covers full history (Dec 2024 onwards).
"""
from __future__ import annotations

from data.forecast._grib_source import BaseGribSource


class HrrrS3Source(BaseGribSource):
    """Fetch HRRR forecasts from the NOAA public S3 bucket.

    URL pattern:
        https://noaa-hrrr-bdp-pds.s3.amazonaws.com/
            hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFX.grib2

    Parameters
    ----------
    base_url:
        Override the S3 base URL (useful in tests).  When None, the URL is
        constructed from ``settings.hrrr_s3_bucket``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        super().__init__()
        if base_url is not None:
            self._base_url = base_url.rstrip("/")
        else:
            from config.settings import Settings
            bucket = Settings().hrrr_s3_bucket
            self._base_url = f"https://{bucket}.s3.amazonaws.com"

    def _build_url(self, date_str: str, init_hh: str, fxx_str: str) -> str:
        return (
            f"{self._base_url}/hrrr.{date_str}/conus/"
            f"hrrr.t{init_hh}z.wrfsfcf{fxx_str}.grib2"
        )
