from __future__ import annotations

from typing import Any

from data.forecast.base import BaseForecastSource

__all__ = [
    "BaseForecastSource",
    "HrrrS3Source",
    "HrrrNomadsSource",
    "HrrrZarrSource",
]


def __getattr__(name: str) -> Any:
    if name == "HrrrS3Source":
        from data.forecast.hrrr_s3 import HrrrS3Source

        return HrrrS3Source
    if name == "HrrrNomadsSource":
        from data.forecast.hrrr_nomads import HrrrNomadsSource

        return HrrrNomadsSource
    if name == "HrrrZarrSource":
        from data.forecast.hrrr_zarr import HrrrZarrSource

        return HrrrZarrSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
