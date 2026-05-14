"""Tests for IemSource and NoaaFileSource.

All HTTP calls are mocked; no real network requests are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.obs.iem import IemSource, _SKY_MAP, _f_to_c, _t_group_to_c
from data.obs.noaa_file import NoaaFileSource


# ---------------------------------------------------------------------------
# Station stub
# ---------------------------------------------------------------------------

_STATION = MagicMock()
_STATION.station_id = "KLGA"

# ---------------------------------------------------------------------------
# Sample IEM CSV response (mirrors real asos.py output)
# ---------------------------------------------------------------------------

_IEM_CSV = (
    "# this is a comment\n"
    "station,valid,tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,"
    "skyc1,skyc2,skyc3,wxcodes,metar\n"
    "LGA,2026-01-01 00:00,35.1,24.1,62,270,10,0.0,29.92,,,10,FEW,,,,"
    "KLGA 010000Z AUTO 27010KT 10SM FEW025 02/M04 A2992 RMK AO2 SLP134"
    " T00170039\n"
    "LGA,2026-01-01 01:00,37.0,26.1,63,280,8,0.0,29.95,,,10,OVC,,,,"
    "KLGA 010100Z AUTO 28008KT 10SM OVC030 03/M03 A2995 RMK AO2 SLP147"
    " T00280028\n"
)

# ---------------------------------------------------------------------------
# Unit: temperature helpers
# ---------------------------------------------------------------------------


def test_f_to_c_freezing():
    assert _f_to_c(32.0) == pytest.approx(0.0, abs=1e-6)


def test_f_to_c_boiling():
    assert _f_to_c(212.0) == pytest.approx(100.0, abs=1e-6)


def test_t_group_to_c_positive():
    # T00170039 -> 1.7 C / 3.9 C
    assert _t_group_to_c("0", "017") == pytest.approx(1.7, abs=1e-6)


def test_t_group_to_c_negative():
    # T10050... -> -0.5 C
    assert _t_group_to_c("1", "005") == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Unit: CSV parsing
# ---------------------------------------------------------------------------


def test_parse_csv_returns_dataframe():
    src = IemSource()
    df = src._parse_csv(_IEM_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_parse_csv_valid_time_utc():
    src = IemSource()
    df = src._parse_csv(_IEM_CSV)
    assert df["valid_time"].dt.tz is not None


def test_parse_csv_t_group_precision():
    # Row 1 METAR has T00170039 -> raw_tmpc_tenths should be 1.7 C
    src = IemSource()
    df = src._parse_csv(_IEM_CSV)
    assert df["raw_tmpc_tenths"].iloc[0] == pytest.approx(1.7, abs=0.01)


def test_parse_csv_t_group_dewpoint():
    # Row 1 METAR has T00170039 -> raw_dwpc_tenths should be 3.9 C
    src = IemSource()
    df = src._parse_csv(_IEM_CSV)
    assert df["raw_dwpc_tenths"].iloc[0] == pytest.approx(3.9, abs=0.01)


def test_parse_csv_wind_fields():
    src = IemSource()
    df = src._parse_csv(_IEM_CSV)
    assert df["raw_wind_spd_kt"].iloc[0] == pytest.approx(10.0)
    # gust falls back to speed when not present in CSV
    assert not pd.isna(df["raw_wind_spd_kt"].iloc[0])


def test_parse_csv_sky_cover_numeric():
    src = IemSource()
    df = src._parse_csv(_IEM_CSV)
    # "FEW" -> 15.0
    assert df["skyc1"].iloc[0] == pytest.approx(15.0)
    # "OVC" -> 100.0
    assert df["skyc1"].iloc[1] == pytest.approx(100.0)


def test_parse_csv_missing_column_is_nan():
    # CSV without metar column -> T-group not available; tmpf used as fallback
    no_metar_csv = (
        "station,valid,tmpf,dwpf,drct,sknt,p01i,alti,vsby,gust,skyc1,wxcodes\n"
        "LGA,2026-01-01 00:00,35.6,26.6,270,10,0.0,29.92,10,,CLR,\n"
    )
    src = IemSource()
    df = src._parse_csv(no_metar_csv)
    assert len(df) == 1
    expected_tmpc = _f_to_c(35.6)
    assert df["raw_tmpc_tenths"].iloc[0] == pytest.approx(expected_tmpc, abs=0.05)


def test_parse_csv_empty_text_returns_empty():
    src = IemSource()
    df = src._parse_csv("")
    assert len(df) == 0
    assert "valid_time" in df.columns


def test_parse_csv_comment_only_returns_empty():
    src = IemSource()
    df = src._parse_csv("# no data\n# still nothing\n")
    assert len(df) == 0


def test_parse_csv_ovc_sky_map():
    assert _SKY_MAP["OVC"] == 100.0
    assert _SKY_MAP["CLR"] == 0.0
    assert _SKY_MAP["SCT"] == 40.0


# ---------------------------------------------------------------------------
# Integration: fetch() calls _download_csv and maps fields
# ---------------------------------------------------------------------------


def test_fetch_calls_download_and_parses():
    src = IemSource()
    with patch.object(src, "_download_csv", return_value=_IEM_CSV):
        df = src.fetch(
            _STATION,
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
        )
    assert len(df) == 2
    assert set(["valid_time", "raw_tmpc_tenths", "raw_wind_spd_kt"]).issubset(
        df.columns
    )


def test_fetch_returns_empty_on_download_failure():
    src = IemSource()
    with patch.object(src, "_download_csv", return_value=None):
        df = src.fetch(
            _STATION,
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
        )
    assert len(df) == 0


# ---------------------------------------------------------------------------
# NoaaFileSource: METAR TXT parsing
# ---------------------------------------------------------------------------

_NOAA_TXT = (
    "2026/01/01 14:53\n"
    "KLGA 011453Z 22012G18KT 10SM OVC055 14/03 A2979 "
    "RMK AO2 SLP097 T01390028 10144 20039 52012\n"
)


def test_noaa_parse_txt_valid_time():
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row is not None
    vt = row["valid_time"]
    assert vt.hour == 14
    assert vt.minute == 53


def test_noaa_parse_txt_t_group_temp():
    # T01390028 -> 13.9 C / 2.8 C
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row["raw_tmpc_tenths"] == pytest.approx(13.9, abs=0.01)
    assert row["raw_dwpc_tenths"] == pytest.approx(2.8, abs=0.01)


def test_noaa_parse_txt_wind():
    # 22012G18KT -> dir=220, spd=12, gust=18
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row["raw_wind_spd_kt"]  == pytest.approx(12.0)
    assert row["raw_wind_gust_kt"] == pytest.approx(18.0)


def test_noaa_parse_txt_altimeter():
    # A2979 -> 29.79 inHg
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row["raw_alti_inhg"] == pytest.approx(29.79, abs=0.01)


def test_noaa_parse_txt_sky_ovc():
    # OVC055 -> 100 % at 5500 ft
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row["skyc1"]  == pytest.approx(100.0)
    assert row["skyl1"]  == pytest.approx(5500.0)


def test_noaa_parse_txt_visibility():
    # 10SM -> 10.0
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row["raw_vis_sm"] == pytest.approx(10.0)


def test_noaa_parse_txt_no_precip_flag():
    src = NoaaFileSource()
    row = src._parse_txt(_NOAA_TXT)
    assert row["is_rain"] == 0.0
    assert row["is_snow"] == 0.0


def test_noaa_parse_txt_rain_flag():
    txt = "2026/01/01 14:53\nKLGA 011453Z 18010KT 5SM -RA OVC020 12/10 A2980\n"
    src = NoaaFileSource()
    row = src._parse_txt(txt)
    assert row["is_rain"] == 1.0
    assert row["is_snow"] == 0.0


def test_noaa_fetch_outside_window_returns_empty():
    src = NoaaFileSource()
    with patch.object(src._session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _NOAA_TXT
        mock_resp.raise_for_status = lambda: None
        mock_get.return_value = mock_resp

        df = src.fetch(
            _STATION,
            pd.Timestamp("2026-01-02", tz="UTC"),   # METAR is 2026-01-01
            pd.Timestamp("2026-01-03", tz="UTC"),
        )
    assert len(df) == 0


def test_noaa_fetch_inside_window_returns_row():
    src = NoaaFileSource()
    with patch.object(src._session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _NOAA_TXT
        mock_resp.raise_for_status = lambda: None
        mock_get.return_value = mock_resp

        df = src.fetch(
            _STATION,
            pd.Timestamp("2026-01-01 14:00", tz="UTC"),
            pd.Timestamp("2026-01-01 15:00", tz="UTC"),
        )
    assert len(df) == 1
    assert df["raw_tmpc_tenths"].iloc[0] == pytest.approx(13.9, abs=0.01)


def test_noaa_parse_fractional_visibility():
    # "1 1/2SM" -> 1.5 SM
    txt = "2026/01/01 14:53\nKLGA 011453Z 18010KT 1 1/2SM OVC020 12/10 A2980\n"
    src = NoaaFileSource()
    row = src._parse_txt(txt)
    assert row["raw_vis_sm"] == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# V04: IemSource tick mode
# ---------------------------------------------------------------------------


def test_fetch_tick_mode_uses_51_timestamp():
    """Tick mode computes start/end around :51 of H-1."""
    src = IemSource()
    captured_params: dict = {}

    def mock_download(params: dict) -> str | None:
        captured_params.update(params)
        return None   # empty response is fine; we only check params

    with patch.object(src, "_download_csv", side_effect=mock_download):
        src.fetch(
            _STATION,
            start=pd.Timestamp("2026-01-01", tz="UTC"),   # ignored in tick mode
            end=pd.Timestamp("2026-01-02", tz="UTC"),     # ignored in tick mode
            now_utc=pd.Timestamp("2026-01-01 14:00", tz="UTC"),
        )

    # now_utc=14:00, H-1=13:00, target=13:51, window=[13:50:30, 13:51:30]
    # IEM params use day1/day2 only (minute precision not passed)
    assert captured_params.get("year1") == "2026"
    assert captured_params.get("month1") == "1"
    assert captured_params.get("day1") == "1"


def test_fetch_tick_mode_ignores_start_end():
    """Tick mode overrides start/end, not just narrows them."""
    src = IemSource()
    calls: list[dict] = []

    def mock_download(params: dict) -> str | None:
        calls.append(dict(params))
        return None

    far_future = pd.Timestamp("2030-12-31", tz="UTC")
    now = pd.Timestamp("2026-01-01 14:00", tz="UTC")

    with patch.object(src, "_download_csv", side_effect=mock_download):
        src.fetch(_STATION, start=far_future, end=far_future, now_utc=now)

    # The year should be 2026 (from now_utc), not 2030 (from start/end)
    assert calls[0]["year1"] == "2026"


def test_fetch_tick_mode_no_now_utc_uses_start_end():
    """Without now_utc, fetch() uses start/end as before."""
    src = IemSource()
    calls: list[dict] = []

    def mock_download(params: dict) -> str | None:
        calls.append(dict(params))
        return None

    with patch.object(src, "_download_csv", side_effect=mock_download):
        src.fetch(
            _STATION,
            start=pd.Timestamp("2026-03-25", tz="UTC"),
            end=pd.Timestamp("2026-03-26", tz="UTC"),
        )

    assert calls[0]["year1"] == "2026"
    assert calls[0]["month1"] == "3"
    assert calls[0]["day1"] == "25"


# ---------------------------------------------------------------------------
# V04: IemSource T-group fallback WARNING
# ---------------------------------------------------------------------------


def test_fetch_warns_on_tgroup_fallback():
    """A WARNING is emitted when tmpf is used instead of T-group."""
    no_tgroup_csv = (
        "station,valid,tmpf,dwpf,drct,sknt,p01i,alti,vsby,gust,skyc1,wxcodes,metar\n"
        "LGA,2026-01-01 00:00,35.6,26.6,270,10,0.0,29.92,10,,CLR,,KLGA 010000Z CLR\n"
    )
    src = IemSource()
    messages: list[str] = []
    from loguru import logger as _logger
    sink_id = _logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        src._parse_csv(no_tgroup_csv)
    finally:
        _logger.remove(sink_id)

    assert any("T-group" in m for m in messages), "Expected T-group fallback WARNING"


def test_fetch_no_warning_when_tgroup_present():
    """No T-group WARNING when all rows have T-group in METAR."""
    src = IemSource()
    messages: list[str] = []
    from loguru import logger as _logger
    sink_id = _logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        src._parse_csv(_IEM_CSV)   # _IEM_CSV rows have T-group in metar
    finally:
        _logger.remove(sink_id)

    tgroup_warnings = [m for m in messages if "T-group" in m]
    assert not tgroup_warnings, f"Unexpected T-group WARNINGs: {tgroup_warnings}"


# ---------------------------------------------------------------------------
# V04: NoaaFileSource robustness
# ---------------------------------------------------------------------------


def test_noaa_incomplete_metar_no_crash():
    """A truncated file (1 line only) returns a partial row, not None."""
    src = NoaaFileSource()
    row = src._parse_txt("2026/01/01 14:53")
    # Should return a dict (not None) with NaN for unparseable fields
    assert row is not None
    assert "valid_time" in row
    import numpy as np
    assert np.isnan(row["raw_tmpc_tenths"])


def test_noaa_missing_temp_logs_warning():
    """METAR without temp field logs a WARNING."""
    no_temp_txt = (
        "2026/01/01 14:53\n"
        "KLGA 011453Z 22012KT 10SM OVC055 A2979 RMK AO2 SLP097\n"
    )
    src = NoaaFileSource()
    messages: list[str] = []
    from loguru import logger as _logger
    sink_id = _logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        src._parse_txt(no_temp_txt)
    finally:
        _logger.remove(sink_id)

    assert any("temperature unavailable" in m for m in messages)
