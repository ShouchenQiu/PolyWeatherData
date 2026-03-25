import numpy as np
import pandas as pd
import pytest

from data.storage.cleaner import Cleaner, _KELVIN_OFFSET, _KT_PER_MS


# ---------------------------------------------------------------------------
# clean_hrrr -- Kelvin -> Celsius
# ---------------------------------------------------------------------------


def test_kelvin_to_celsius_tmp2m():
    df = pd.DataFrame({"TMP_2m": [273.15, 300.0, 250.0]})
    result = Cleaner.clean_hrrr(df)
    expected = [0.0, 300.0 - 273.15, 250.0 - 273.15]
    assert result["TMP_2m"].tolist() == pytest.approx(expected, abs=1e-6)


def test_kelvin_to_celsius_all_temp_cols():
    # 280 K -> 6.85 C, which is within the plausibility bounds of every column
    # including TMP_500mb whose upper bound is 10 C
    cols = ["TMP_2m", "DPT_2m", "TMP_sfc", "TMP_850mb", "TMP_500mb"]
    data = {c: [280.0] for c in cols}
    result = Cleaner.clean_hrrr(pd.DataFrame(data))
    for col in cols:
        assert result[col].iloc[0] == pytest.approx(280.0 - 273.15, abs=1e-6)


# ---------------------------------------------------------------------------
# clean_hrrr -- m/s -> knots
# ---------------------------------------------------------------------------


def test_wind_ms_to_kt():
    df = pd.DataFrame({"UGRD_10m": [10.0], "GUST_sfc": [15.0]})
    result = Cleaner.clean_hrrr(df)
    assert result["UGRD_10m"].iloc[0] == pytest.approx(10.0 * _KT_PER_MS, rel=1e-5)
    assert result["GUST_sfc"].iloc[0] == pytest.approx(15.0 * _KT_PER_MS, rel=1e-5)


# ---------------------------------------------------------------------------
# clean_hrrr -- plausibility clipping
# ---------------------------------------------------------------------------


def test_rh_clipped_to_0_103():
    df = pd.DataFrame({"RH_2m": [-5.0, 50.0, 110.0]})
    result = Cleaner.clean_hrrr(df)
    assert result["RH_2m"].tolist() == pytest.approx([0.0, 50.0, 103.0])


def test_visibility_clipped():
    df = pd.DataFrame({"VIS": [-100.0, 50_000.0, 200_000.0]})
    result = Cleaner.clean_hrrr(df)
    assert result["VIS"].iloc[0] == 0.0
    assert result["VIS"].iloc[2] == 100_000.0


def test_tmp2m_clip_after_kelvin_conversion():
    # 150 K -> -123.15 C, which is below -80 clip bound
    df = pd.DataFrame({"TMP_2m": [150.0]})
    result = Cleaner.clean_hrrr(df)
    assert result["TMP_2m"].iloc[0] == -80.0


# ---------------------------------------------------------------------------
# clean_hrrr -- dedup
# ---------------------------------------------------------------------------


def test_hrrr_drops_duplicate_rows():
    df = pd.DataFrame({"TMP_2m": [290.0, 290.0], "RH_2m": [50.0, 50.0]})
    result = Cleaner.clean_hrrr(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# clean_iem -- valid_time parsing
# ---------------------------------------------------------------------------


def test_iem_valid_time_becomes_utc():
    df = pd.DataFrame({"valid_time": ["2026-01-01 00:00:00"]})
    result = Cleaner.clean_iem(df)
    assert result["valid_time"].dt.tz is not None


# ---------------------------------------------------------------------------
# clean_iem -- gust fill
# ---------------------------------------------------------------------------


def test_iem_gust_filled_from_speed():
    df = pd.DataFrame({
        "valid_time": ["2026-01-01 00:00:00"],
        "raw_wind_spd_kt": [10.0],
        "raw_wind_gust_kt": [float("nan")],
    })
    result = Cleaner.clean_iem(df)
    assert result["raw_wind_gust_kt"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# clean_iem -- zero-fill flags
# ---------------------------------------------------------------------------


def test_iem_zero_fill_rain_snow():
    df = pd.DataFrame({
        "valid_time": ["2026-01-01 00:00:00"],
        "is_rain": [float("nan")],
        "is_snow": [float("nan")],
        "p01i": [float("nan")],
    })
    result = Cleaner.clean_iem(df)
    assert result["is_rain"].iloc[0] == 0.0
    assert result["is_snow"].iloc[0] == 0.0
    assert result["p01i"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# clean_iem -- hourly dedup
# ---------------------------------------------------------------------------


def test_iem_hourly_dedup_keeps_first():
    df = pd.DataFrame({
        "valid_time": [
            "2026-01-01 00:05:00",  # rounds to 00:00
            "2026-01-01 00:50:00",  # rounds to 01:00
            "2026-01-01 00:55:00",  # also rounds to 01:00 -> drop
        ],
        "raw_tmpc_tenths": [1.0, 2.0, 3.0],
    })
    result = Cleaner.clean_iem(df)
    assert len(result) == 2
    assert result["raw_tmpc_tenths"].tolist() == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# clean_iem -- sort order
# ---------------------------------------------------------------------------


def test_iem_sorted_by_valid_time():
    df = pd.DataFrame({
        "valid_time": ["2026-01-01 02:00:00", "2026-01-01 01:00:00"],
        "raw_tmpc_tenths": [2.0, 1.0],
    })
    result = Cleaner.clean_iem(df)
    assert result["valid_time"].iloc[0] < result["valid_time"].iloc[1]
