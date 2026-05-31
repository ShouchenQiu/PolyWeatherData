"""Tests for HrrrZarrSource -- V2V02 hardening.

zarr and fsspec are mocked; no real S3 requests are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.forecast.vars import DATA_COLS, ZARR_VARS


_STATION = MagicMock()
_STATION.station_id = "KLGA"
_STATION.lat = 40.7769
_STATION.lon = -73.8740
_STATION.hrrr_grid_row_zarr = 698
_STATION.hrrr_grid_col_zarr = 1556


def _station_no_grid():
    """Station config missing zarr grid attributes."""
    s = MagicMock(spec=["station_id", "lat", "lon"])
    s.station_id = "KBOS"
    s.lat = 42.3601
    s.lon = -71.0589
    return s


def _make_zarr_mocks(
    n_fcst: int = 48,
    row_shape: int = 1000,
    col_shape: int = 1800,
    tmp2m_fail: bool = False,
):
    """Build minimal zarr group mocks for testing."""
    anl_arr = MagicMock()
    anl_arr.__getitem__ = lambda self, idx: 280.0

    fcst_arr = MagicMock()
    fcst_arr.shape = (n_fcst, row_shape, col_shape)
    fcst_arr.__getitem__ = lambda self, idx: np.full(n_fcst, 280.0)

    def make_navigate(fail_tmp2m: bool):
        def _navigate(root, path):
            if fail_tmp2m and "TMP" in path and "fcst" in str(root):
                raise RuntimeError("zarr read error")
            if "fcst" in str(root):
                return fcst_arr
            return anl_arr
        return _navigate

    z_anl = MagicMock()
    z_fcst = MagicMock()
    z_anl.__str__ = lambda self: "z_anl"
    z_fcst.__str__ = lambda self: "z_fcst"

    return z_anl, z_fcst, make_navigate(tmp2m_fail)


# ---------------------------------------------------------------------------
# Module-level DATA_COLS / ZARR_VARS consistency
# ---------------------------------------------------------------------------


def test_data_cols_zarr_vars_consistency():
    """DATA_COLS and ZARR_VARS must have the same set of column names."""
    assert set(DATA_COLS) == set(ZARR_VARS.keys()), (
        f"Mismatch: DATA_COLS - ZARR_VARS = {set(DATA_COLS) - set(ZARR_VARS.keys())}, "
        f"ZARR_VARS - DATA_COLS = {set(ZARR_VARS.keys()) - set(DATA_COLS)}"
    )


# ---------------------------------------------------------------------------
# Station grid config validation
# ---------------------------------------------------------------------------


def test_missing_grid_config_raises_value_error():
    """ValueError when station_config is missing hrrr_grid_row_zarr/col."""
    from data.forecast.hrrr_zarr import HrrrZarrSource
    src = HrrrZarrSource()
    with pytest.raises(ValueError, match="hrrr_grid_row_zarr"):
        src.fetch(
            pd.Timestamp("2026-01-01 06:00", tz="UTC"),
            range(0, 5),
            _station_no_grid(),
        )


# ---------------------------------------------------------------------------
# Grid bounds check
# ---------------------------------------------------------------------------


def test_grid_bounds_check_raises_on_out_of_bounds():
    """ValueError when station row/col index exceeds zarr array shape."""
    import data.forecast.hrrr_zarr as _mod

    z_anl, z_fcst, navigate_fn = _make_zarr_mocks(row_shape=100, col_shape=100)

    # Station has row=698 but array is only 100x100 -> out of bounds
    station = MagicMock()
    station.station_id = "KBIG"
    station.lat = 40.0
    station.lon = -73.0
    station.hrrr_grid_row_zarr = 698
    station.hrrr_grid_col_zarr = 50

    with (
        patch.object(_mod, "_zarr", MagicMock()),
        patch.object(_mod, "_fsspec", MagicMock()),
        patch.object(_mod, "_navigate", navigate_fn),
    ):
        _mod._zarr.open_group.side_effect = [z_anl, z_fcst]
        src = _mod.HrrrZarrSource()
        with pytest.raises(ValueError, match="out of bounds"):
            src.fetch(
                pd.Timestamp("2026-01-01 06:00", tz="UTC"),
                range(0, 5),
                station,
            )


# ---------------------------------------------------------------------------
# n_fcst_steps probe failure
# ---------------------------------------------------------------------------


def test_n_fcst_steps_probe_failure_raises():
    """RuntimeError when TMP_2m shape exploration fails (no fallback)."""
    import data.forecast.hrrr_zarr as _mod

    z_anl, z_fcst, navigate_fn = _make_zarr_mocks(tmp2m_fail=True)

    with (
        patch.object(_mod, "_zarr", MagicMock()),
        patch.object(_mod, "_fsspec", MagicMock()),
        patch.object(_mod, "_navigate", navigate_fn),
    ):
        _mod._zarr.open_group.side_effect = [z_anl, z_fcst]
        src = _mod.HrrrZarrSource()
        with pytest.raises(RuntimeError, match="TMP_2m"):
            src.fetch(
                pd.Timestamp("2026-01-01 06:00", tz="UTC"),
                range(0, 5),
                _STATION,
            )


# ---------------------------------------------------------------------------
# MAJOR init: is_extrapolated=False + base_url override
# ---------------------------------------------------------------------------


def test_zarr_base_url_override():
    """Constructor accepts base_url override for testing."""
    from data.forecast.hrrr_zarr import HrrrZarrSource
    src = HrrrZarrSource(base_url="https://custom-zarr.example.com/sfc")
    assert src._base_url == "https://custom-zarr.example.com/sfc"


# ---------------------------------------------------------------------------
# Non-MAJOR nearest-fail: is_extrapolated must still be present
# ---------------------------------------------------------------------------


def test_zarr_non_major_nearest_fail_has_is_extrapolated():
    """When nearest MAJOR fetch fails, non-MAJOR fallback DataFrame has is_extrapolated=False.

    The non-MAJOR path opens z_anl/z_fcst for the 03Z init (open_group calls
    1 and 2), then recursively calls self.fetch() for the nearest MAJOR run
    which tries open_group again (call 3).  We raise on call 3 to simulate an
    unavailable MAJOR store, triggering the early return.  The returned
    DataFrame must carry is_extrapolated=False.
    """
    import data.forecast.hrrr_zarr as _mod

    z_anl, z_fcst, navigate_fn = _make_zarr_mocks()

    with (
        patch.object(_mod, "_zarr", MagicMock()),
        patch.object(_mod, "_fsspec", MagicMock()),
        patch.object(_mod, "_navigate", navigate_fn),
    ):
        open_calls = [0]

        def open_group_side(mapper, mode="r"):
            open_calls[0] += 1
            if open_calls[0] == 1:
                return z_anl   # non-MAJOR anl store
            if open_calls[0] == 2:
                return z_fcst  # non-MAJOR fcst store
            raise RuntimeError("nearest MAJOR store unavailable")

        _mod._zarr.open_group.side_effect = open_group_side

        src = _mod.HrrrZarrSource()
        result = src.fetch(
            pd.Timestamp("2026-01-01 03:00", tz="UTC"),
            range(0, 5),
            _STATION,
        )

    df = result["KLGA"]
    assert "is_extrapolated" in df.columns
    assert (df["is_extrapolated"] == False).all()  # noqa: E712
