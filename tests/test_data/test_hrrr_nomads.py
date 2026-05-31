"""Tests for HrrrNomadsSource -- URL construction and is_extrapolated column.

All network calls are mocked; no real HTTP requests are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.forecast.hrrr_nomads import HrrrNomadsSource
from data.forecast.vars import DATA_COLS


_SAMPLE_IDX = """\
1:0:d=2026010100:TMP:2 m above ground:1 hour fcst:
2:12345:d=2026010100:DPT:2 m above ground:1 hour fcst:
"""

_STATION = MagicMock()
_STATION.station_id = "KLGA"
_STATION.lat = 40.7769
_STATION.lon = -73.8740
_STATION.hrrr_grid_row_zarr = 698
_STATION.hrrr_grid_col_zarr = 1556


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def test_nomads_build_url_major():
    """URL for a MAJOR init matches the expected NOMADS pattern."""
    src = HrrrNomadsSource()
    url = src._build_url("20260101", "00", "06")
    assert "nomads.ncep.noaa.gov" in url
    assert "hrrr.20260101/conus" in url
    assert "hrrr.t00z.wrfsfcf06.grib2" in url


def test_nomads_build_url_non_major():
    """URL for a non-MAJOR init follows the same pattern (fxx 03)."""
    src = HrrrNomadsSource()
    url = src._build_url("20260101", "03", "09")
    assert "hrrr.t03z.wrfsfcf09.grib2" in url


def test_nomads_base_url_override():
    """Constructor accepts base_url override for testing."""
    src = HrrrNomadsSource(base_url="https://test-nomads.example.com")
    url = src._build_url("20260101", "06", "12")
    assert url.startswith("https://test-nomads.example.com")


# ---------------------------------------------------------------------------
# is_extrapolated column
# ---------------------------------------------------------------------------


def test_nomads_fetch_output_has_is_extrapolated_false():
    """MAJOR init fetch() output includes is_extrapolated=False on all rows."""
    src = HrrrNomadsSource()
    with (
        patch.object(src, "_fetch_idx", return_value=_SAMPLE_IDX),
        patch.object(src, "_fetch_range", return_value=None),
    ):
        result = src.fetch(
            pd.Timestamp("2026-01-01 06:00", tz="UTC"),
            range(0, 3),
            _STATION,
        )
    df = result["KLGA"]
    assert "is_extrapolated" in df.columns
    assert (df["is_extrapolated"] == False).all()  # noqa: E712
