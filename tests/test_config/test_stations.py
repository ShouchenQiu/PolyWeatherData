# tests/test_config/test_stations.py - Unit tests for config.stations.
import re

import pytest

from config.stations import STATIONS, StationConfig


def test_klga_exists():
    assert "KLGA" in STATIONS


def test_klga_is_station_config():
    assert isinstance(STATIONS["KLGA"], StationConfig)


def test_klga_lat_in_range():
    lat = STATIONS["KLGA"].lat
    # LaGuardia Airport is roughly 40-41 N
    assert 40.0 < lat < 41.5, f"Unexpected lat: {lat}"


def test_klga_lon_in_range():
    lon = STATIONS["KLGA"].lon
    # LaGuardia Airport is roughly -74 to -73 E
    assert -75.0 < lon < -73.0, f"Unexpected lon: {lon}"


def test_slug_pattern_compiles():
    pattern = STATIONS["KLGA"].slug_pattern
    compiled = re.compile(pattern)
    assert compiled is not None


def test_slug_pattern_matches_example():
    pattern = STATIONS["KLGA"].slug_pattern
    assert re.search(pattern, "will-the-high-temp-at-lga-exceed-90f")


def test_timezone_nonempty():
    assert STATIONS["KLGA"].timezone == "America/New_York"


def test_hrrr_grid_values():
    st = STATIONS["KLGA"]
    assert st.hrrr_grid_row_zarr == 698
    assert st.hrrr_grid_col_zarr == 1556


# ---------------------------------------------------------------------------
# V2V01 -- slug_aliases
# ---------------------------------------------------------------------------

def test_klga_has_slug_aliases():
    """StationConfig must have a slug_aliases list (non-empty for KLGA)."""
    st = STATIONS["KLGA"]
    assert hasattr(st, "slug_aliases")
    assert isinstance(st.slug_aliases, list)
    assert len(st.slug_aliases) >= 1


def test_klga_slug_aliases_include_icao():
    """'klga' must appear in the aliases so _infer_station finds it."""
    aliases = [a.lower() for a in STATIONS["KLGA"].slug_aliases]
    assert "klga" in aliases
