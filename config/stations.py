# config/stations.py - Station configuration dataclass and registry.
# STATIONS is the single source of truth for all supported weather stations.
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StationConfig:
    station_id: str
    lat: float
    lon: float
    timezone: str
    tz_offset: int
    iem_network: str
    slug_pattern: str
    slug_aliases: list[str] = field(default_factory=list)


STATIONS: dict[str, StationConfig] = {
    "KLGA": StationConfig(
        station_id="KLGA",
        lat=40.7769,
        lon=-73.8740,
        timezone="America/New_York",
        tz_offset=-5,
        iem_network="NY_ASOS",
        slug_pattern=r"will-the-high.*lga",
        slug_aliases=[
            "klga", "new-york-lga", "new-york-city", "laguardia", "nyc",
        ],
    ),
    "KORD": StationConfig(
        station_id="KORD",
        lat=41.9742,
        lon=-87.9073,
        timezone="America/Chicago",
        tz_offset=-6,
        iem_network="IL_ASOS",
        slug_pattern=r"will-the-high.*(ord|chicago)",
        slug_aliases=[
            "kord", "chicago-ohare", "chicago", "ord",
        ],
    ),
    "KATL": StationConfig(
        station_id="KATL",
        lat=33.6407,
        lon=-84.4277,
        timezone="America/New_York",
        tz_offset=-5,
        iem_network="GA_ASOS",
        slug_pattern=r"will-the-high.*(atl|atlanta)",
        slug_aliases=[
            "katl", "atlanta", "atl",
        ],
    ),
    "KSEA": StationConfig(
        station_id="KSEA",
        lat=47.4490,
        lon=-122.3090,
        timezone="America/Los_Angeles",
        tz_offset=-8,
        iem_network="WA_ASOS",
        slug_pattern=r"will-the-high.*(sea|seattle)",
        slug_aliases=[
            "ksea", "seattle", "sea",
        ],
    ),
    "KDFW": StationConfig(
        station_id="KDFW",
        lat=32.8998,
        lon=-97.0403,
        timezone="America/Chicago",
        tz_offset=-6,
        iem_network="TX_ASOS",
        slug_pattern=r"will-the-high.*(dfw|dallas)",
        slug_aliases=[
            "kdfw", "dallas-fort-worth", "dallas", "dfw",
        ],
    ),
}
