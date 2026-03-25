# config/stations.py - Station configuration dataclass and registry.
# STATIONS is the single source of truth for all supported weather stations.
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StationConfig:
    station_id: str
    lat: float
    lon: float
    timezone: str
    tz_offset: int
    hrrr_grid_row_zarr: int
    hrrr_grid_col_zarr: int
    iem_network: str
    slug_pattern: str


STATIONS: dict[str, StationConfig] = {
    "KLGA": StationConfig(
        station_id="KLGA",
        lat=40.7769,
        lon=-73.8740,
        timezone="America/New_York",
        tz_offset=-5,
        hrrr_grid_row_zarr=698,
        hrrr_grid_col_zarr=1556,
        iem_network="NY_ASOS",
        slug_pattern=r"will-the-high.*lga",
    ),
}
