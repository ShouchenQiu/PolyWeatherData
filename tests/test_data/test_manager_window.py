"""Tests for DataManager.build_encoder_obs_window (V03)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.manager import DataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_station(station_id: str = "KLGA") -> MagicMock:
    s = MagicMock()
    s.station_id = station_id
    return s


def _make_hrrr_rows(n: int, start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for i in range(n):
        vt = start + pd.Timedelta(hours=i)
        rows.append({
            "init_time":  str(start),
            "fxx":        1,
            "valid_time": str(vt),
            "TMP_2m":     5.0 + i * 0.1,
        })
    return pd.DataFrame(rows)


def _make_obs_rows(n: int, start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for i in range(n):
        vt = start + pd.Timedelta(hours=i)
        rows.append({
            "valid_time":      str(vt),
            "raw_tmpc_tenths": 6.0 + i * 0.1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 72-row success path
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_returns_merged_72_rows():
    """Returns a merged DataFrame with 72 rows when both stores have enough data."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # Provide cold_start(72) + n_hours(72) = 144 rows so that after
    # drop_cold_start(72) exactly 72 rows remain for the merge.
    hrrr_df = _make_hrrr_rows(144, window_start - pd.Timedelta(hours=72))
    obs_df  = _make_obs_rows(144, window_start - pd.Timedelta(hours=72))

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = obs_df

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        result = mgr.build_encoder_obs_window(station, init, n_hours=72)

    assert len(result) == 72
    assert "TMP_2m" in result.columns
    assert "raw_tmpc_tenths" in result.columns
    assert "temp_error" in result.columns
    # temp_error = raw_tmpc_tenths - TMP_2m; first row: 6.0 - 5.0 = 1.0
    assert result["temp_error"].iloc[0] == pytest.approx(1.0, abs=0.01)
    assert result["valid_time"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Raises when HRRR rows < cold_start_hours
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_raises_on_insufficient_hrrr():
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # 10 rows -> after drop_cold_start(72) -> 0 rows < 72 threshold
    short_hrrr = _make_hrrr_rows(10, window_start)
    obs_df     = _make_obs_rows(144, window_start - pd.Timedelta(hours=72))

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = short_hrrr
    obs_store.read.return_value = obs_df

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        with pytest.raises(RuntimeError, match="insufficient HRRR rows"):
            mgr.build_encoder_obs_window(station, init, n_hours=72)


# ---------------------------------------------------------------------------
# Raises when obs rows < cold_start_hours
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_raises_on_insufficient_obs():
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_df   = _make_hrrr_rows(144, window_start - pd.Timedelta(hours=72))
    # 5 rows -> after drop_cold_start(72) -> 0 rows < 72 threshold
    short_obs = _make_obs_rows(5, window_start)

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = short_obs

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        with pytest.raises(RuntimeError, match="insufficient obs rows"):
            mgr.build_encoder_obs_window(station, init, n_hours=72)


# ---------------------------------------------------------------------------
# Raises when merged result < cold_start_hours (valid_time mismatch)
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_raises_on_insufficient_merged():
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    base_start = init - pd.Timedelta(hours=144)
    # HRRR: 144 rows starting at base_start (valid_times base+0h .. base+143h)
    # After drop_cold_start(72): rows base+72h .. base+143h
    # Obs: 144 rows starting 80 hours after base_start
    # After drop_cold_start(72): rows base+80+72=base+152h .. base+80+143h
    # -> no temporal overlap with HRRR post-drop range -> merged is empty
    obs_start = base_start + pd.Timedelta(hours=80)

    hrrr_df    = _make_hrrr_rows(144, base_start)
    offset_obs = _make_obs_rows(144, obs_start)

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = offset_obs

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        with pytest.raises(RuntimeError, match="too few rows|merged window too short"):
            mgr.build_encoder_obs_window(station, init, n_hours=72)


# ---------------------------------------------------------------------------
# Only fxx=1 rows from HRRR store are used
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_uses_only_fxx1():
    # Store guarantees fxx=1 only (ADR-0003); verify build_encoder_obs_window
    # returns the correct row count without any internal fxx filter.
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    base_start = init - pd.Timedelta(hours=144)

    hrrr_df = _make_hrrr_rows(144, base_start)  # all fxx=1 (store invariant)
    obs_df  = _make_obs_rows(144, base_start)

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = obs_df

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        result = mgr.build_encoder_obs_window(station, init, n_hours=72)

    assert len(result) == 72
    if "fxx" in result.columns:
        assert (result["fxx"] == 1).all()


# ---------------------------------------------------------------------------
# Cold-start rows are dropped before merge
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_drops_cold_start_rows():
    """The first cold_start_hours rows of the wider read are never in the result."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    base_start = init - pd.Timedelta(hours=144)   # 144 rows each

    # Mark cold-start HRRR rows with a sentinel: TMP_2m = -999.0
    hrrr_df = _make_hrrr_rows(144, base_start)
    hrrr_df.loc[:71, "TMP_2m"] = -999.0           # first 72 = cold-start

    obs_df = _make_obs_rows(144, base_start)

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = obs_df

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        result = mgr.build_encoder_obs_window(station, init, n_hours=72)

    assert len(result) == 72
    assert (-999.0 not in result["TMP_2m"].values), (
        "Cold-start sentinel rows must not appear in the encoder window"
    )


# ---------------------------------------------------------------------------
# Raises when dropna(temp_error) shrinks merged below n_hours
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_raises_when_temp_error_nan_leaves_too_few():
    """72 merged rows with one NaN TMP_2m -> dropna leaves 71 -> RuntimeError."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    base_start = init - pd.Timedelta(hours=144)

    hrrr_df = _make_hrrr_rows(144, base_start)
    obs_df  = _make_obs_rows(144, base_start)

    # Inject NaN into TMP_2m at one row that will survive into the merged window
    # (row 72 is the first post-cold-start row in the 144-row set)
    hrrr_df.loc[72, "TMP_2m"] = float("nan")

    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = obs_df

    station = _make_station()
    mgr = DataManager([], [], forecast_store, obs_store)

    with patch("config.settings.Settings") as MockSettings:
        MockSettings.return_value.cold_start_hours = 72
        with pytest.raises(RuntimeError, match="temp_error NaN removal left too few rows"):
            mgr.build_encoder_obs_window(station, init, n_hours=72)


# ---------------------------------------------------------------------------
# V2V03: DataManager._obs_is_complete -- multi-field completeness gate (#54/#56)
# ---------------------------------------------------------------------------


def _complete_obs_row() -> dict:
    """A row that satisfies all completeness criteria."""
    return {
        "valid_time":      pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        "raw_tmpc_tenths": 5.0,
        "obs_U":           -3.0,
        "obs_V":            4.0,
        "skyc1":            0.0,
        "p01i":             0.0,
    }


def test_obs_is_complete_accepts_full_row():
    df = pd.DataFrame([_complete_obs_row()])
    assert DataManager._obs_is_complete(df) is True


def test_obs_is_complete_rejects_temperature_only():
    """A single-column temperature DataFrame must not pass the gate. (#54)"""
    df = pd.DataFrame([{"raw_tmpc_tenths": 5.0}])
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_rejects_missing_wind():
    row = _complete_obs_row()
    del row["obs_U"]
    del row["obs_V"]
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_accepts_obs_ws_as_wind():
    """obs_WS (scalar wind speed) is sufficient to satisfy the wind requirement."""
    row = _complete_obs_row()
    del row["obs_U"]
    del row["obs_V"]
    row["obs_WS"] = 5.0
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is True


def test_obs_is_complete_rejects_missing_skyc1():
    row = _complete_obs_row()
    row["skyc1"] = float("nan")
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_rejects_missing_p01i():
    row = _complete_obs_row()
    row["p01i"] = float("nan")
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["raw_tmpc_tenths", "obs_U", "obs_V", "skyc1", "p01i"])
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_accepts_zero_precip():
    """p01i=0 (no rain) is a valid reading, not missing."""
    row = _complete_obs_row()
    row["p01i"] = 0.0
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is True


def test_obs_is_complete_vrb_wind_passes_after_cleaning():
    """VRB wind (obs_U/V filled to 0 by process_iem) must pass the completeness gate."""
    row = _complete_obs_row()
    row["obs_U"] = 0.0   # process_iem fills VRB to 0
    row["obs_V"] = 0.0
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is True


def test_obs_is_complete_rejects_cross_row_partial_data():
    """Values spread across rows where no single row has all required fields must fail."""
    rows = [
        # Row A: has temp but no wind
        {
            "raw_tmpc_tenths": 5.0,
            "obs_U": float("nan"),
            "obs_V": float("nan"),
            "skyc1": 0.0,
            "p01i": 0.0,
        },
        # Row B: has wind but no temp
        {
            "raw_tmpc_tenths": float("nan"),
            "obs_U": -3.0,
            "obs_V": 4.0,
            "skyc1": 0.0,
            "p01i": 0.0,
        },
    ]
    df = pd.DataFrame(rows)
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_rejects_obs_u_without_obs_v():
    """obs_U alone (no obs_V column) must NOT satisfy the wind requirement."""
    row = {
        "raw_tmpc_tenths": 5.0,
        "obs_U": 3.0,
        "skyc1": 0.0,
        "p01i": 0.0,
    }
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is False


def test_obs_is_complete_requires_both_uv_components():
    """Both obs_U and obs_V must be non-null in the same row to satisfy the wind gate."""
    row = {
        "raw_tmpc_tenths": 5.0,
        "obs_U": 3.0,
        "obs_V": float("nan"),
        "skyc1": 0.0,
        "p01i": 0.0,
    }
    df = pd.DataFrame([row])
    assert DataManager._obs_is_complete(df) is False
