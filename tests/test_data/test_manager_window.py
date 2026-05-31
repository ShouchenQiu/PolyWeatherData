"""Tests for DataManager.build_encoder_obs_window and inference_readiness (V04)."""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from data.manager import DataManager, InsufficientEncoderData, ReadinessStatus


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


def _make_manager(hrrr_df: pd.DataFrame, obs_df: pd.DataFrame) -> tuple:
    """Return (mgr, forecast_store, obs_store) with stubbed reads."""
    forecast_store = MagicMock()
    obs_store = MagicMock()
    forecast_store.read.return_value = hrrr_df
    obs_store.read.return_value = obs_df
    mgr = DataManager([], [], forecast_store, obs_store)
    return mgr, forecast_store, obs_store


# ---------------------------------------------------------------------------
# V2V04: build_encoder_obs_window -- exact window, no cold_start extension
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_returns_merged_72_rows():
    """Returns a merged DataFrame with 72 rows when both stores serve the exact window."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # V2V04: stores need to return exactly n_hours rows (no cold_start extension).
    hrrr_df = _make_hrrr_rows(72, window_start)
    obs_df  = _make_obs_rows(72, window_start)
    mgr, _, _ = _make_manager(hrrr_df, obs_df)

    result = mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)

    assert len(result) == 72
    assert "TMP_2m" in result.columns
    assert "raw_tmpc_tenths" in result.columns
    assert "temp_error" in result.columns
    # temp_error = raw_tmpc_tenths - TMP_2m; first row approx 6.0 - 5.0 = 1.0
    assert result["temp_error"].iloc[0] == pytest.approx(1.0, abs=0.01)
    assert result["valid_time"].is_monotonic_increasing


def test_build_encoder_obs_window_reads_exact_window_from_stores():
    """Store.read() is called with exactly [init - n_hours, init - 1h] -- no extra rows."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    expected_start = init - pd.Timedelta(hours=72)
    expected_end   = init - pd.Timedelta(hours=1)

    hrrr_df = _make_hrrr_rows(72, expected_start)
    obs_df  = _make_obs_rows(72, expected_start)
    mgr, fc_store, obs_store = _make_manager(hrrr_df, obs_df)

    mgr.build_encoder_obs_window(_make_station("KLGA"), init, n_hours=72)

    fc_store.read.assert_called_once_with("KLGA", expected_start, expected_end)
    obs_store.read.assert_called_once_with("KLGA", expected_start, expected_end)


def test_build_encoder_obs_window_copies_store_results():
    """Mutating the returned DataFrame must not corrupt the store's copy."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_df = _make_hrrr_rows(72, window_start)
    obs_df  = _make_obs_rows(72, window_start)

    original_hrrr_values = hrrr_df["TMP_2m"].copy()

    mgr, _, _ = _make_manager(hrrr_df, obs_df)
    result = mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)

    # Mutate the result; the original DataFrame must be unaffected.
    result["TMP_2m"] = -999.0
    assert (hrrr_df["TMP_2m"] == original_hrrr_values).all(), (
        ".copy() must be called after store reads to prevent shared-reference mutations"
    )


# ---------------------------------------------------------------------------
# Raises InsufficientEncoderData (not RuntimeError) for short windows
# ---------------------------------------------------------------------------

def test_build_encoder_obs_window_raises_on_insufficient_hrrr():
    """InsufficientEncoderData when HRRR store has fewer than n_hours rows."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    short_hrrr = _make_hrrr_rows(10, window_start)
    obs_df     = _make_obs_rows(72, window_start)
    mgr, _, _  = _make_manager(short_hrrr, obs_df)

    with pytest.raises(InsufficientEncoderData, match="insufficient HRRR rows"):
        mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)


def test_build_encoder_obs_window_raises_on_insufficient_obs():
    """InsufficientEncoderData when obs store has fewer than n_hours rows."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_df   = _make_hrrr_rows(72, window_start)
    short_obs = _make_obs_rows(5, window_start)
    mgr, _, _ = _make_manager(hrrr_df, short_obs)

    with pytest.raises(InsufficientEncoderData, match="insufficient obs rows"):
        mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)


def test_build_encoder_obs_window_raises_on_insufficient_merged():
    """InsufficientEncoderData when HRRR and obs timestamps do not overlap."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)
    # Obs starts 80 hours after HRRR window_start -- no overlap within window.
    obs_start = window_start + pd.Timedelta(hours=80)

    hrrr_df    = _make_hrrr_rows(72, window_start)
    offset_obs = _make_obs_rows(72, obs_start)
    mgr, _, _  = _make_manager(hrrr_df, offset_obs)

    with pytest.raises(InsufficientEncoderData, match="too few rows|merged window too short"):
        mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)


def test_build_encoder_obs_window_raises_on_missing_required_columns():
    """RuntimeError when HRRR store omits TMP_2m."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # HRRR rows without TMP_2m column
    rows = [{"init_time": str(window_start), "fxx": 1,
             "valid_time": str(window_start + pd.Timedelta(hours=i))}
            for i in range(72)]
    hrrr_no_tmp = pd.DataFrame(rows)
    obs_df = _make_obs_rows(72, window_start)
    mgr, _, _ = _make_manager(hrrr_no_tmp, obs_df)

    with pytest.raises(RuntimeError, match="required columns missing"):
        mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)


def test_build_encoder_obs_window_raises_when_temp_error_nan_leaves_too_few():
    """InsufficientEncoderData when temp_error NaN removal shrinks result below n_hours."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_df = _make_hrrr_rows(72, window_start)
    obs_df  = _make_obs_rows(72, window_start)

    # Inject NaN into TMP_2m -- temp_error becomes NaN for that row,
    # dropna removes it, leaving 71 < 72 required rows.
    hrrr_df.loc[0, "TMP_2m"] = float("nan")
    mgr, _, _ = _make_manager(hrrr_df, obs_df)

    with pytest.raises(InsufficientEncoderData, match="temp_error NaN removal left too few rows"):
        mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)


def test_build_encoder_obs_window_uses_only_fxx1():
    """Store guarantees fxx=1 only (ADR-0003); result contains only fxx=1 rows."""
    init = pd.Timestamp("2026-01-04 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_df = _make_hrrr_rows(72, window_start)
    obs_df  = _make_obs_rows(72, window_start)
    mgr, _, _ = _make_manager(hrrr_df, obs_df)

    result = mgr.build_encoder_obs_window(_make_station(), init, n_hours=72)

    assert len(result) == 72
    if "fxx" in result.columns:
        assert (result["fxx"] == 1).all()


# ---------------------------------------------------------------------------
# V2V04: inference_readiness (#3, #27)
# ---------------------------------------------------------------------------

def _make_manager_with_side_effects(
    window_hrrr: pd.DataFrame,
    window_obs:  pd.DataFrame,
    hist_hrrr:   pd.DataFrame | None = None,
    hist_obs:    pd.DataFrame | None = None,
) -> DataManager:
    """Build a DataManager whose stores return different data per call.

    First call to read() returns the window DataFrame (encoder window check).
    If hist_* is provided, a second call returns the historical DataFrame
    (first-eligible-init computation).
    """
    forecast_store = MagicMock()
    obs_store      = MagicMock()

    if hist_hrrr is not None:
        fc_side  = [window_hrrr, hist_hrrr]
        forecast_store.read.side_effect = fc_side
    else:
        forecast_store.read.return_value = window_hrrr

    if hist_obs is not None:
        obs_side = [window_obs, hist_obs]
        obs_store.read.side_effect = obs_side
    else:
        obs_store.read.return_value = window_obs

    return DataManager([], [], forecast_store, obs_store)


def test_inference_readiness_ok_when_full_window():
    """Returns ready=True and reason='ok' when both windows are fully populated and
    store has data back at least cold_start_hours + encoder_hours (144 h) before init.
    """
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_window = _make_hrrr_rows(72, window_start)
    obs_window  = _make_obs_rows(72, window_start)

    # Historical data starts at init - 144h so that:
    # first_ts = init - 144h
    # first_eligible = (init - 144h) + 72 + 72 = init -> not > init -> ready=True
    hist_start = init - pd.Timedelta(hours=144)
    hist_hrrr  = _make_hrrr_rows(1, hist_start)
    hist_obs   = _make_obs_rows(1, hist_start)

    mgr = _make_manager_with_side_effects(
        window_hrrr=hrrr_window,
        window_obs=obs_window,
        hist_hrrr=hist_hrrr,
        hist_obs=hist_obs,
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is True
    assert status.reason == "ok"
    assert status.first_eligible_init is not None


def test_inference_readiness_cold_start_even_with_full_window():
    """Returns cold_start even when encoder window has 72 rows but store is too young.

    When data collection started only 72h ago, the encoder window is full but
    the cold-start burn-in period has not elapsed.  The fix ensures that
    _estimate_first_eligible() is called even for the "both windows full" path.
    """
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_window = _make_hrrr_rows(72, window_start)
    obs_window  = _make_obs_rows(72, window_start)

    # No historical data before window_start (store started at init - 72h)
    empty_hist = pd.DataFrame()

    mgr = _make_manager_with_side_effects(
        window_hrrr=hrrr_window,
        window_obs=obs_window,
        hist_hrrr=empty_hist,
        hist_obs=empty_hist,
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "cold_start"
    assert status.first_eligible_init is not None
    assert status.first_eligible_init > init


def test_inference_readiness_cold_start_when_no_data():
    """Returns ready=False, reason='cold_start' when the encoder window is empty."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")

    empty = pd.DataFrame()
    mgr = _make_manager_with_side_effects(
        window_hrrr=empty,
        window_obs=empty,
        hist_hrrr=empty,   # no historical data either
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "cold_start"
    assert status.first_eligible_init is None


def test_inference_readiness_cold_start_when_first_row_too_recent():
    """Returns cold_start when first available row is < encoder_hours before init_time."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # Only 40 rows in the encoder window (not enough)
    short_hrrr = _make_hrrr_rows(40, window_start + pd.Timedelta(hours=32))

    # Historical read (before window_start) finds the very first row at
    # init - 60h.  With cold_start_hours=72 (Settings default),
    # first_eligible = (init - 60h) + 72 + 72 = init + 84h > init -> cold_start.
    hist_first_ts = init - pd.Timedelta(hours=60)
    hist_hrrr = _make_hrrr_rows(1, hist_first_ts)

    mgr = _make_manager_with_side_effects(
        window_hrrr=short_hrrr,
        window_obs=_make_obs_rows(72, window_start),
        hist_hrrr=hist_hrrr,
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "cold_start"
    # first_eligible_init should be > init (cold_start_hours + encoder_hours pushes it forward)
    assert status.first_eligible_init is not None
    assert status.first_eligible_init > init


def test_inference_readiness_incomplete_hrrr_window_when_past_cold_start():
    """Returns incomplete_hrrr_window when warm, but this window is short."""
    init = pd.Timestamp("2026-06-01 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # Only 50 rows in window (short by 22)
    short_hrrr = _make_hrrr_rows(50, window_start)

    # Historical read finds a row from 200h before init -- we're past cold start.
    hist_hrrr = _make_hrrr_rows(1, init - pd.Timedelta(hours=200))

    mgr = _make_manager_with_side_effects(
        window_hrrr=short_hrrr,
        window_obs=_make_obs_rows(72, window_start),
        hist_hrrr=hist_hrrr,
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "incomplete_hrrr_window"
    # first_eligible is in the past (we already passed cold start)
    assert status.first_eligible_init is not None
    assert status.first_eligible_init <= init


def test_inference_readiness_incomplete_obs_window():
    """Returns incomplete_obs_window when HRRR is OK but obs is short."""
    init = pd.Timestamp("2026-06-01 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_df     = _make_hrrr_rows(72, window_start)
    short_obs   = _make_obs_rows(30, window_start)
    hist_obs    = _make_hrrr_rows(1, init - pd.Timedelta(hours=200))

    mgr = _make_manager_with_side_effects(
        window_hrrr=hrrr_df,
        window_obs=short_obs,
        hist_obs=hist_obs,
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "incomplete_obs_window"


def test_inference_readiness_cold_start_first_data_in_encoder_window():
    """Returns cold_start when all available data is inside the encoder window (typical early stage).

    The typical cold-start scenario: we just started collecting data, so the
    store has e.g. 40 rows all inside [window_start, window_end].  The
    historical read (before window_start) returns nothing.  _estimate_first_eligible
    must use the current_window rows to compute first_eligible.
    """
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # All data is inside the encoder window -- no historical data before window_start
    early_rows = _make_hrrr_rows(40, window_start + pd.Timedelta(hours=5))
    # early_rows[0].valid_time = window_start + 5h = init - 67h
    # With cold_start_hours=72 (default): first_eligible = (init-67h) + 72 + 72 = init+77h > init

    empty_hist = pd.DataFrame()  # nothing before window_start

    mgr = _make_manager_with_side_effects(
        window_hrrr=early_rows,
        window_obs=_make_obs_rows(72, window_start),
        hist_hrrr=empty_hist,
    )

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "cold_start"
    assert status.first_eligible_init is not None
    assert status.first_eligible_init > init


def test_inference_readiness_missing_required_columns():
    """Returns missing_required_columns when TMP_2m is absent from HRRR window."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # HRRR without TMP_2m column
    rows = [{"valid_time": str(window_start + pd.Timedelta(hours=i)), "fxx": 1}
            for i in range(72)]
    hrrr_no_col = pd.DataFrame(rows)
    obs_df = _make_obs_rows(72, window_start)

    mgr, _, _ = _make_manager(hrrr_no_col, obs_df)

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "missing_required_columns"


def test_inference_readiness_missing_valid_time_column():
    """Returns missing_required_columns when valid_time is absent from HRRR window."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    # HRRR without valid_time column -- _min_valid_time would return None;
    # inference_readiness must catch this before reporting ready=True.
    hrrr_no_vt = pd.DataFrame([{"TMP_2m": 5.0 + i} for i in range(72)])
    obs_df = _make_obs_rows(72, window_start)

    mgr, _, _ = _make_manager(hrrr_no_vt, obs_df)

    status = mgr.inference_readiness(_make_station(), init, encoder_hours=72)

    assert status.ready is False
    assert status.reason == "missing_required_columns"


def test_is_inference_ready_returns_bool():
    """is_inference_ready is a thin boolean wrapper over inference_readiness."""
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    window_start = init - pd.Timedelta(hours=72)

    hrrr_window = _make_hrrr_rows(72, window_start)
    obs_window  = _make_obs_rows(72, window_start)

    hist_start = init - pd.Timedelta(hours=144)
    hist_hrrr  = _make_hrrr_rows(1, hist_start)
    hist_obs   = _make_obs_rows(1, hist_start)

    mgr = _make_manager_with_side_effects(
        window_hrrr=hrrr_window,
        window_obs=obs_window,
        hist_hrrr=hist_hrrr,
        hist_obs=hist_obs,
    )

    result = mgr.is_inference_ready(_make_station(), init, encoder_hours=72)
    assert result is True


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
