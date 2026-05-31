"""Canonical feature column definitions shared by model training and inference.

This module must not import from any other project package (config, core,
data, db, events, loops, model, snipers, ui) to remain a leaf dependency
usable from scripts/ and model/ alike.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# HRRR-derived known-real features
# These appear in both the encoder and decoder windows (known at forecast time).
# ---------------------------------------------------------------------------

HRRR_KNOWN_REALS: list[str] = [
    "TMP_2m",    "DPT_2m",    "RH_2m",    "TMP_sfc",
    "TMP_850mb", "TMP_500mb",
    "WS_10m",    "WD_sin_10m",  "WD_cos_10m",
    "WS_80m",    "WD_sin_80m",  "WD_cos_80m",
    "WS_500mb",  "WD_sin_500mb","WD_cos_500mb",
    "log1p_GUST_sfc",
    "RH_0C",  "RH_frz",
    "DSWRF", "DLWRF", "USWRF", "ULWRF",
    "TCDC",  "LCDC",  "MCDC",  "HCDC",
    "log1p_PRATE", "APCP", "REFC", "PWAT",
    "CAPE",  "CIN",
    "HPBL", "SHTFL", "LHTFL",
    "MSLMA", "VIS", "SNOD",
]

# ---------------------------------------------------------------------------
# Observation-only unknown-real features
# These are available in the encoder window only (not known in the decoder).
# ---------------------------------------------------------------------------

OBS_UNKNOWN_REALS: list[str] = [
    "raw_tmpc_tenths",
    "raw_dwpc_tenths",  "raw_wind_spd_kt", "raw_wind_gust_kt",
    "obs_WS",           "obs_WD_sin",      "obs_WD_cos",
    "raw_vis_sm",       "raw_alti_inhg",
    "skyc1",            "skyl1",
    "p01i",             "is_rain",         "is_snow",
]

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------

TARGET: str = "temp_error"

# ---------------------------------------------------------------------------
# Derived column sets
# ---------------------------------------------------------------------------

REQUIRED_ENCODER_SOURCE_COLUMNS: list[str] = (
    HRRR_KNOWN_REALS + OBS_UNKNOWN_REALS + [TARGET]
)

REQUIRED_DECODER_SOURCE_COLUMNS: list[str] = HRRR_KNOWN_REALS

# ---------------------------------------------------------------------------
# Model-generated columns (added by the pipeline, not sourced from raw data)
# ---------------------------------------------------------------------------

MODEL_GENERATED_COLUMNS: list[str] = [
    "time_idx",
    "local_idx",
    "month",
    "day_of_year",
    "hour_of_day",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
    "init_time_str",
    "Node_ID",
]
