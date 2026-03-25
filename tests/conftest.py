"""
Shared pytest fixtures for PolyWeatherBot test suite.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Column definitions (mirrored from Herbie-main/tft_step6_build_72h.py)
# ---------------------------------------------------------------------------

IEM_COLS = [
    "raw_tmpc_tenths",
    "raw_dwpc_tenths",
    "raw_wind_spd_kt",
    "raw_wind_gust_kt",
    "obs_U",
    "obs_V",
    "raw_vis_sm",
    "raw_alti_inhg",
    "skyc1",
    "skyl1",
    "p01i",
    "is_rain",
    "is_snow",
]

HRRR_COLS = [
    "TMP_2m",     "DPT_2m",     "RH_2m",      "TMP_sfc",
    "TMP_850mb",  "TMP_500mb",
    "UGRD_10m",   "VGRD_10m",   "GUST_sfc",
    "UGRD_80m",   "VGRD_80m",
    "UGRD_500mb", "VGRD_500mb",
    "RH_0C",      "RH_frz",
    "DSWRF",      "DLWRF",      "USWRF",      "ULWRF",
    "TCDC",       "LCDC",       "MCDC",       "HCDC",
    "PRATE",      "APCP",       "REFC",       "PWAT",
    "CAPE",       "CIN",
    "HPBL",       "SHTFL",      "LHTFL",
    "PRES_sfc",   "MSLMA",      "VIS",        "SNOD",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db():
    """Return an in-memory SQLite connection."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Alias for pytest's built-in tmp_path."""
    return tmp_path


@pytest.fixture()
def sample_hrrr_df() -> pd.DataFrame:
    """
    Small HRRR-style DataFrame: 3 init_times x 48 forecast hours = 144 rows.
    Contains all 36 HRRR variables plus init_time, fxx, valid_time.
    """
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    rows = []
    for i in range(3):
        init_time = base + pd.Timedelta(hours=i * 6)
        for fxx in range(48):
            valid_time = init_time + pd.Timedelta(hours=fxx)
            row = {"init_time": init_time, "fxx": fxx, "valid_time": valid_time}
            for col in HRRR_COLS:
                row[col] = float(rng.normal(0.0, 1.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


@pytest.fixture()
def sample_iem_df() -> pd.DataFrame:
    """
    Small IEM-style DataFrame: 48 hourly observations.
    Contains all 13 IEM variables plus valid_time.
    """
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    rows = []
    for h in range(48):
        row = {"valid_time": base + pd.Timedelta(hours=h)}
        for col in IEM_COLS:
            row[col] = float(rng.normal(0.0, 1.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
