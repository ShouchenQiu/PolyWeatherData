"""Tests for HrrrS3Source (and shared BaseGribSource logic).

All network calls are mocked; no real HTTP requests are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.forecast.hrrr_s3 import HrrrS3Source, _S3_BASE
from data.forecast.vars import DATA_COLS, HRRR_SEARCHES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_IDX = """\
1:0:d=2026010100:TMP:2 m above ground:1 hour fcst:
2:12345:d=2026010100:DPT:2 m above ground:1 hour fcst:
3:24690:d=2026010100:RH:2 m above ground:1 hour fcst:
4:37035:d=2026010100:GUST:surface:1 hour fcst:
"""

_STATION = MagicMock()
_STATION.station_id = "KLGA"
_STATION.lat = 40.7769
_STATION.lon = -73.8740
_STATION.hrrr_grid_row_zarr = 698
_STATION.hrrr_grid_col_zarr = 1556


# ---------------------------------------------------------------------------
# Unit: URL construction
# ---------------------------------------------------------------------------


def test_build_url_format():
    src = HrrrS3Source()
    url = src._build_url("20260101", "06", "12")
    expected = f"{_S3_BASE}/hrrr.20260101/conus/hrrr.t06z.wrfsfcf12.grib2"
    assert url == expected


def test_build_url_zero_padded_fxx():
    src = HrrrS3Source()
    url = src._build_url("20260101", "00", "03")
    assert "wrfsfcf03" in url


# ---------------------------------------------------------------------------
# Unit: .idx parsing
# ---------------------------------------------------------------------------


def test_parse_idx_extracts_correct_cols():
    src = HrrrS3Source()
    targets = src._parse_idx(_SAMPLE_IDX)
    col_names = [t[2] for t in targets]
    assert "TMP_2m" in col_names
    assert "DPT_2m" in col_names
    assert "RH_2m" in col_names
    assert "GUST_sfc" in col_names


def test_parse_idx_byte_ranges():
    src = HrrrS3Source()
    targets = src._parse_idx(_SAMPLE_IDX)
    by_col = {t[2]: (t[0], t[1]) for t in targets}
    # TMP_2m starts at byte 0, ends at (12345 - 1)
    assert by_col["TMP_2m"] == (0, 12344)
    # DPT_2m starts at 12345, ends at (24690 - 1)
    assert by_col["DPT_2m"] == (12345, 24689)


def test_parse_idx_last_msg_large_end():
    src = HrrrS3Source()
    targets = src._parse_idx(_SAMPLE_IDX)
    by_col = {t[2]: (t[0], t[1]) for t in targets}
    # GUST_sfc is the last message: end should be start + 5_000_000
    start, end = by_col["GUST_sfc"]
    assert end == start + 5_000_000


def test_parse_idx_empty_returns_empty():
    src = HrrrS3Source()
    assert src._parse_idx("") == []


def test_parse_idx_unknown_vars_skipped():
    idx = "1:0:d=2026010100:UNKNOWN_VAR:surface:1 hour fcst:\n"
    src = HrrrS3Source()
    targets = src._parse_idx(idx)
    assert targets == []


# ---------------------------------------------------------------------------
# Unit: Range request header format
# ---------------------------------------------------------------------------


def test_fetch_range_header_format():
    src = HrrrS3Source()
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["headers"] = headers
        resp = MagicMock()
        resp.status_code = 206
        resp.content = b"\x00" * 10
        return resp

    src._session.get = fake_get
    result = src._fetch_range("https://example.com/file.grib2", 100, 999)
    assert result == b"\x00" * 10
    assert captured["headers"] == {"Range": "bytes=100-999"}


def test_fetch_range_returns_none_on_failure():
    src = HrrrS3Source()

    def bad_get(*args, **kwargs):
        raise ConnectionError("network down")

    src._session.get = bad_get
    result = src._fetch_range("https://example.com/file.grib2", 0, 99)
    assert result is None


# ---------------------------------------------------------------------------
# Integration: fetch() returns correct shape with mocked HTTP
# ---------------------------------------------------------------------------


def _make_fake_idx_resp(idx_text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = idx_text
    return resp


def test_fetch_returns_dict_keyed_by_station():
    """fetch() returns a dict with station_id keys."""
    src = HrrrS3Source()
    with (
        patch.object(src, "_fetch_idx", return_value=_SAMPLE_IDX),
        patch.object(src, "_fetch_range", return_value=None),
    ):
        result = src.fetch(
            pd.Timestamp("2026-01-01 06:00", tz="UTC"),
            range(0, 3),
            _STATION,
        )
    assert isinstance(result, dict)
    assert "KLGA" in result


def test_fetch_returns_dataframe_with_all_cols():
    """fetch() result for each station has init_time/fxx/valid_time and all 36 data columns."""
    src = HrrrS3Source()

    # Make _fetch_idx return a minimal idx, _fetch_range return None (simulate
    # download failure -> all NaN) -- the row must still be built.
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
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "init_time" in df.columns
    assert "fxx" in df.columns
    assert "valid_time" in df.columns
    for col in DATA_COLS:
        assert col in df.columns


def test_fetch_all_nan_when_idx_unavailable():
    """When ALL idx fetches fail the source raises RuntimeError so DataManager
    can fall back to the next source."""
    src = HrrrS3Source()
    with patch.object(src, "_fetch_idx", return_value=None):
        with pytest.raises(RuntimeError, match="all idx fetches failed"):
            src.fetch(
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                range(0, 2),
                _STATION,
            )


def test_fetch_fxx_values_correct():
    """fxx column values match the requested range even when range-bytes fail."""
    src = HrrrS3Source()
    with (
        patch.object(src, "_fetch_idx", return_value=_SAMPLE_IDX),
        patch.object(src, "_fetch_range", return_value=None),
    ):
        result = src.fetch(
            pd.Timestamp("2026-01-01 12:00", tz="UTC"),
            range(0, 5),
            _STATION,
        )
    assert list(result["KLGA"]["fxx"]) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Unit: HRRR_SEARCHES covers all 36 data columns
# ---------------------------------------------------------------------------


def test_hrrr_searches_covers_all_data_cols():
    search_cols = {col for _, col in HRRR_SEARCHES}
    missing = set(DATA_COLS) - search_cols
    assert not missing, f"Columns missing from HRRR_SEARCHES: {missing}"


# ---------------------------------------------------------------------------
# Multi-station: fetch(station=None) returns all configured STATIONS
# ---------------------------------------------------------------------------


def test_fetch_station_none_returns_all_configured_stations():
    """fetch(station=None) keys the result dict by every station in STATIONS."""
    from unittest.mock import MagicMock
    import config.stations as _cfg_stations

    klga = _cfg_stations.STATIONS["KLGA"]
    kord = _cfg_stations.StationConfig(
        station_id="KORD",
        lat=41.9742,
        lon=-87.9073,
        timezone="America/Chicago",
        tz_offset=-6,
        hrrr_grid_row_zarr=700,
        hrrr_grid_col_zarr=600,
        iem_network="IL_ASOS",
        slug_pattern=r"will-the-high.*ord",
    )
    two_stations = {"KLGA": klga, "KORD": kord}

    src = HrrrS3Source()
    with (
        patch("config.stations.STATIONS", two_stations),
        patch.object(src, "_fetch_idx", return_value=_SAMPLE_IDX),
        patch.object(src, "_fetch_range", return_value=None),
    ):
        result = src.fetch(
            pd.Timestamp("2026-01-01 06:00", tz="UTC"),
            range(0, 2),
            station=None,
        )

    assert set(result.keys()) == {"KLGA", "KORD"}
    assert len(result["KLGA"]) == 2
    assert len(result["KORD"]) == 2
