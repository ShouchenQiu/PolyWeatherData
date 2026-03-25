from __future__ import annotations

import numpy as np
import pandas as pd

# Temperature columns that arrive in Kelvin from GRIB2 / eccodes
_KELVIN_COLS = [
    "TMP_2m", "DPT_2m", "TMP_sfc", "TMP_850mb", "TMP_500mb",
]

# Wind columns decoded in m/s that must become knots (1 m/s = 1.94384 kt)
_MS_TO_KT_COLS = [
    "UGRD_10m", "VGRD_10m", "GUST_sfc",
    "UGRD_80m", "VGRD_80m",
    "UGRD_500mb", "VGRD_500mb",
]

# Physical plausibility bounds (Kelvin already converted to Celsius before clip)
_HRRR_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "TMP_2m":    (-80.0, 60.0),
    "DPT_2m":    (-80.0, 50.0),
    "TMP_sfc":   (-80.0, 60.0),
    "TMP_850mb": (-80.0, 40.0),
    "TMP_500mb": (-80.0, 10.0),
    "RH_2m":     (0.0, 103.0),
    "RH_0C":     (0.0, 103.0),
    "RH_frz":    (0.0, 103.0),
    "TCDC":      (0.0, 100.0),
    "LCDC":      (0.0, 100.0),
    "MCDC":      (0.0, 100.0),
    "HCDC":      (0.0, 100.0),
    "VIS":       (0.0, 100_000.0),
    "SNOD":      (0.0, 5.0),
}

# IEM numeric columns that should be zero-filled when missing
_IEM_ZERO_FILL = ["is_rain", "is_snow", "p01i"]

_KT_PER_MS = 1.94384
_KELVIN_OFFSET = 273.15


class Cleaner:
    """Data cleaning utilities for HRRR forecast and IEM observation DataFrames."""

    @staticmethod
    def clean_hrrr(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a raw HRRR forecast DataFrame.

        Steps applied in order:
        1. Convert Kelvin columns to Celsius.
        2. Convert wind/gust columns from m/s to knots.
        3. Clip numeric columns to physical plausibility bounds.
        4. Drop fully-duplicate rows.
        """
        df = df.copy()

        # 1. Kelvin -> Celsius
        for col in _KELVIN_COLS:
            if col in df.columns:
                df[col] = df[col] - _KELVIN_OFFSET

        # 2. m/s -> knots
        for col in _MS_TO_KT_COLS:
            if col in df.columns:
                df[col] = df[col] * _KT_PER_MS

        # 3. Clip to plausibility bounds
        for col, (lo, hi) in _HRRR_CLIP_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        # 4. Remove exact duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @staticmethod
    def clean_iem(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a raw IEM METAR observation DataFrame.

        Steps applied in order:
        1. Parse valid_time to UTC-aware Timestamp.
        2. Fill missing wind gust with wind speed.
        3. Zero-fill accumulation/flag columns.
        4. Deduplicate by rounded hourly valid_time (keep first).
        5. Sort by valid_time.
        """
        df = df.copy()

        # 1. Ensure valid_time is datetime
        if "valid_time" in df.columns:
            df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

        # 2. Fill missing gust with speed
        if "raw_wind_gust_kt" in df.columns and "raw_wind_spd_kt" in df.columns:
            df["raw_wind_gust_kt"] = df["raw_wind_gust_kt"].fillna(
                df["raw_wind_spd_kt"]
            )

        # 3. Zero-fill rain/snow flags and accumulation
        for col in _IEM_ZERO_FILL:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 4. Hourly dedup: round to nearest hour, keep first observation
        if "valid_time" in df.columns:
            df["_rounded"] = df["valid_time"].dt.round("h")
            df = df.drop_duplicates(subset="_rounded", keep="first")
            df = df.drop(columns=["_rounded"])

        # 5. Sort
        if "valid_time" in df.columns:
            df = df.sort_values("valid_time").reset_index(drop=True)

        return df
