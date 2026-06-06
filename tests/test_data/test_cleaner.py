import numpy as np
import pandas as pd
import pytest

from data.storage.cleaner import Cleaner


# ---------------------------------------------------------------------------
# clean_hrrr -- Kelvin -> Celsius
# ---------------------------------------------------------------------------


def test_kelvin_to_celsius_tmp2m():
    df = pd.DataFrame({"TMP_2m": [0.0, 280.0, 300.0, 250.0]})
    result = Cleaner.clean_hrrr(df)
    expected = [0.0, 280.0 - 273.15, 300.0 - 273.15, 250.0 - 273.15]
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
# clean_hrrr -- wind stays in m/s
# ---------------------------------------------------------------------------


def test_wind_stays_mps():
    df = pd.DataFrame({"UGRD_10m": [10.0], "GUST_sfc": [15.0]})
    result = Cleaner.clean_hrrr(df)
    assert result["UGRD_10m"].iloc[0] == pytest.approx(10.0)
    assert result["GUST_sfc"].iloc[0] == pytest.approx(15.0)


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
    df = pd.DataFrame({"TMP_2m": [-100.0]})
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


def test_iem_valid_time_rounds_to_representative_hour():
    df = pd.DataFrame({"valid_time": ["2026-01-01 13:51:00"]})
    result = Cleaner.clean_iem(df)
    assert result["valid_time"].iloc[0] == pd.Timestamp(
        "2026-01-01 14:00:00+00:00"
    )


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


def test_iem_skyc1_forward_filled_then_zero_filled():
    df = pd.DataFrame({
        "valid_time": [
            "2026-01-01 00:00:00",
            "2026-01-01 01:00:00",
            "2026-01-01 02:00:00",
        ],
        "skyc1": [float("nan"), 70.0, float("nan")],
    })
    result = Cleaner.clean_iem(df)
    assert result["skyc1"].tolist() == pytest.approx([0.0, 70.0, 70.0])


def test_iem_obs_uv_forward_backward_filled_then_zero_filled():
    df = pd.DataFrame({
        "valid_time": [
            "2026-01-01 00:00:00",
            "2026-01-01 01:00:00",
            "2026-01-01 02:00:00",
        ],
        "obs_U": [float("nan"), 3.0, float("nan")],
        "obs_V": [float("nan"), 4.0, float("nan")],
    })
    result = Cleaner.clean_iem(df)
    assert result["obs_U"].tolist() == pytest.approx([3.0, 3.0, 3.0])
    assert result["obs_V"].tolist() == pytest.approx([4.0, 4.0, 4.0])


# ---------------------------------------------------------------------------
# clean_iem -- hourly dedup
# ---------------------------------------------------------------------------


def test_iem_hourly_dedup_prefers_51_representative_observation():
    df = pd.DataFrame({
        "valid_time": [
            "2026-01-01 00:05:00",  # rounds to 00:00
            "2026-01-01 00:50:00",  # rounds to 01:00
            "2026-01-01 00:51:00",  # preferred representative for 01:00
        ],
        "raw_tmpc_tenths": [1.0, 2.0, 3.0],
    })
    result = Cleaner.clean_iem(df)
    assert len(result) == 2
    assert result["raw_tmpc_tenths"].tolist() == pytest.approx([1.0, 3.0])


def test_iem_hourly_dedup_rejects_implausible_temperature_candidate():
    df = pd.DataFrame({
        "valid_time": [
            "2026-05-20 20:51:00",
            "2026-05-20 21:10:00",
            "2026-05-20 21:15:00",
        ],
        "raw_tmpc_tenths": [34.4, -61.0, 35.0],
        "raw_dwpc_tenths": [16.7, -61.0, 17.0],
    })
    messages: list[str] = []
    from loguru import logger as _logger
    sink_id = _logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        result = Cleaner.clean_iem(df)
    finally:
        _logger.remove(sink_id)

    assert len(result) == 1
    assert result["valid_time"].iloc[0] == pd.Timestamp("2026-05-20 21:00:00+00:00")
    assert result["raw_tmpc_tenths"].iloc[0] == pytest.approx(34.4)
    assert any("implausible IEM temperature" in m for m in messages)


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


# ---------------------------------------------------------------------------
# V03: build_features_hrrr -- wind decomposition + log1p
# ---------------------------------------------------------------------------


def test_build_features_hrrr_wind_decomp():
    df = Cleaner.clean_hrrr(pd.DataFrame({
        "UGRD_10m": [5.0],
        "VGRD_10m": [3.0],
    }))
    result = Cleaner.build_features_hrrr(df)
    assert "WS_10m" in result.columns
    assert "WD_sin_10m" in result.columns
    assert "WD_cos_10m" in result.columns
    expected_ws = np.sqrt(5.0 ** 2 + 3.0 ** 2)
    expected_wd = np.arctan2(-5.0, -3.0)
    assert result["WS_10m"].iloc[0] == pytest.approx(expected_ws, rel=1e-5)
    assert result["WD_sin_10m"].iloc[0] == pytest.approx(np.sin(expected_wd), rel=1e-5)
    assert result["WD_cos_10m"].iloc[0] == pytest.approx(np.cos(expected_wd), rel=1e-5)


def test_build_features_hrrr_all_wind_levels():
    df = Cleaner.clean_hrrr(pd.DataFrame({
        "UGRD_10m": [1.0],
        "VGRD_10m": [2.0],
        "UGRD_80m": [3.0],
        "VGRD_80m": [4.0],
        "UGRD_500mb": [5.0],
        "VGRD_500mb": [12.0],
    }))
    result = Cleaner.build_features_hrrr(df)
    assert result["WS_10m"].iloc[0] == pytest.approx(np.sqrt(5.0))
    assert result["WS_80m"].iloc[0] == pytest.approx(5.0)
    assert result["WS_500mb"].iloc[0] == pytest.approx(13.0)


def test_build_features_hrrr_calm_wind_direction_zero():
    df = Cleaner.clean_hrrr(pd.DataFrame({"UGRD_10m": [0.0], "VGRD_10m": [0.0]}))
    result = Cleaner.build_features_hrrr(df)
    assert result["WS_10m"].iloc[0] == pytest.approx(0.0)
    assert result["WD_sin_10m"].iloc[0] == pytest.approx(0.0)
    assert result["WD_cos_10m"].iloc[0] == pytest.approx(0.0)


def test_build_features_hrrr_log1p_created():
    df = Cleaner.clean_hrrr(pd.DataFrame({"GUST_sfc": [10.0], "PRATE": [0.002]}))
    result = Cleaner.build_features_hrrr(df)
    assert "log1p_GUST_sfc" in result.columns
    assert "log1p_PRATE" in result.columns
    assert result["log1p_GUST_sfc"].iloc[0] == pytest.approx(np.log1p(10.0), rel=1e-6)
    assert result["log1p_PRATE"].iloc[0] == pytest.approx(np.log1p(0.002), rel=1e-6)


def test_build_features_hrrr_original_cols_preserved():
    df = Cleaner.clean_hrrr(pd.DataFrame({"UGRD_10m": [5.0], "VGRD_10m": [3.0], "DSWRF": [50.0]}))
    result = Cleaner.build_features_hrrr(df)
    assert "UGRD_10m" in result.columns
    assert "DSWRF" in result.columns


def test_build_features_hrrr_no_wind_cols_no_crash():
    df = Cleaner.clean_hrrr(pd.DataFrame({"TMP_2m": [290.0]}))
    result = Cleaner.build_features_hrrr(df)
    assert "WS_10m" not in result.columns
    assert "WD_sin_10m" not in result.columns


# ---------------------------------------------------------------------------
# V03: build_features_iem -- canonical raw obs columns
# ---------------------------------------------------------------------------


def test_build_features_iem_keeps_raw_temperature_columns():
    df = Cleaner.clean_iem(pd.DataFrame({
        "valid_time": ["2026-01-01 00:00:00"],
        "raw_tmpc_tenths": [5.5],
        "raw_dwpc_tenths": [2.0],
    }))
    result = Cleaner.build_features_iem(df)
    assert "raw_tmpc_tenths" in result.columns
    assert result["raw_tmpc_tenths"].iloc[0] == pytest.approx(5.5)
    assert "raw_dwpc_tenths" in result.columns
    assert result["raw_dwpc_tenths"].iloc[0] == pytest.approx(2.0)
    assert "tmpc" not in result.columns
    assert "dwpc" not in result.columns


def test_build_features_iem_raw_cols_still_present():
    df = Cleaner.clean_iem(pd.DataFrame({
        "valid_time": ["2026-01-01 00:00:00"],
        "raw_tmpc_tenths": [5.5],
    }))
    result = Cleaner.build_features_iem(df)
    assert "raw_tmpc_tenths" in result.columns


def test_build_features_iem_observed_wind_decomp():
    df = Cleaner.clean_iem(pd.DataFrame({
        "valid_time": ["2026-01-01 00:00:00"],
        "obs_U": [3.0],
        "obs_V": [4.0],
    }))
    result = Cleaner.build_features_iem(df)
    expected_wd = np.arctan2(-3.0, -4.0)
    assert result["obs_WS"].iloc[0] == pytest.approx(5.0)
    assert result["obs_WD_sin"].iloc[0] == pytest.approx(np.sin(expected_wd), rel=1e-5)
    assert result["obs_WD_cos"].iloc[0] == pytest.approx(np.cos(expected_wd), rel=1e-5)


# ---------------------------------------------------------------------------
# V03: compute_temp_error
# ---------------------------------------------------------------------------


def test_compute_temp_error_prefers_raw_tmpc_tenths():
    df = pd.DataFrame({
        "TMP_2m": [10.0],
        "tmpc": [12.5],
        "raw_tmpc_tenths": [8.5],
    })
    result = Cleaner.compute_temp_error(df)
    assert result["temp_error"].iloc[0] == pytest.approx(-1.5)


def test_compute_temp_error_falls_back_to_tmpc_for_legacy_frames():
    df = pd.DataFrame({"TMP_2m": [10.0], "tmpc": [12.5]})
    result = Cleaner.compute_temp_error(df)
    assert result["temp_error"].iloc[0] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# V03: drop_cold_start
# ---------------------------------------------------------------------------


def test_drop_cold_start_removes_first_n():
    df = pd.DataFrame({"x": list(range(10))})
    result = Cleaner.drop_cold_start(df, 3)
    assert len(result) == 7
    assert result["x"].iloc[0] == 3


def test_drop_cold_start_zero_is_noop():
    df = pd.DataFrame({"x": list(range(5))})
    result = Cleaner.drop_cold_start(df, 0)
    assert len(result) == 5


def test_drop_cold_start_empty_df():
    result = Cleaner.drop_cold_start(pd.DataFrame(), 10)
    assert result.empty


# ---------------------------------------------------------------------------
# V2V03: clean_iem -- sort BEFORE fills so ffill is time-directed (#57)
# ---------------------------------------------------------------------------


def test_iem_ffill_is_time_directed_when_rows_arrive_unsorted():
    """Rows out of time order must be sorted before ffill so fills go forward.

    If sort comes after ffill, a row at t=02:00 (NaN) would see the NaN at
    t=00:00 to its left and fill incorrectly.  The correct behavior is:
      t=00:00 -> NaN   (no prior value -> should stay NaN or bfill from 01:00)
      t=01:00 -> 3.0
      t=02:00 -> NaN   (after t=01:00 -> ffilled to 3.0)
    """
    df = pd.DataFrame({
        "valid_time": [
            "2026-01-01 02:00:00",   # arrives first but is later
            "2026-01-01 01:00:00",   # arrives second but is earlier
            "2026-01-01 00:00:00",   # arrives third and is earliest
        ],
        "obs_U": [float("nan"), 3.0, float("nan")],
        "obs_V": [float("nan"), 4.0, float("nan")],
    })
    result = Cleaner.clean_iem(df)

    # After sort + ffill + bfill:
    #   t=00:00: no prior -> bfilled from t=01:00 -> 3.0 / 4.0
    #   t=01:00: 3.0 / 4.0 (original)
    #   t=02:00: ffilled from t=01:00 -> 3.0 / 4.0
    assert result["valid_time"].is_monotonic_increasing, "rows must be sorted ascending"
    # All three rows end up at 3.0 / 4.0 due to ffill+bfill
    assert result["obs_U"].tolist() == pytest.approx([3.0, 3.0, 3.0])
    assert result["obs_V"].tolist() == pytest.approx([4.0, 4.0, 4.0])
