from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# Temperature columns that arrive in Kelvin from GRIB2 / eccodes
_KELVIN_COLS = [
    "TMP_2m", "DPT_2m", "TMP_sfc", "TMP_850mb", "TMP_500mb",
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
_IEM_TEMP_BOUNDS_C = (-40.0, 55.0)
_IEM_REPRESENTATIVE_TOLERANCE_S = 20 * 60

_KELVIN_OFFSET = 273.15
# IEM temperatures arrive as Celsius tenths (raw_tmpc_tenths) in tenths of a degree
# No unit conversion is needed -- the value is already Celsius.


class Cleaner:
    """Data cleaning utilities for HRRR forecast and IEM observation DataFrames."""

    @staticmethod
    def _wind_components(
        u: pd.Series | np.ndarray,
        v: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return wind speed and meteorological direction sin/cos from U/V."""
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        ws = np.sqrt(u_arr ** 2 + v_arr ** 2)
        wd = np.arctan2(-u_arr, -v_arr)
        calm = ws == 0.0
        sin_wd = np.where(calm, 0.0, np.sin(wd)).astype("float32")
        cos_wd = np.where(calm, 0.0, np.cos(wd)).astype("float32")
        return ws.astype("float32"), sin_wd, cos_wd

    @staticmethod
    def clean_hrrr(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a raw HRRR forecast DataFrame.

        Steps applied in order:
        1. Drop duplicate columns from upstream merges.
        2. Convert Kelvin temperature values (>200 K) to Celsius.
        3. Clip numeric columns to physical plausibility bounds.
        4. Drop fully-duplicate rows.
        """
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]

        # 1. Kelvin -> Celsius, preserving cold-start/sentinel zeros.
        for col in _KELVIN_COLS:
            if col in df.columns:
                k_mask = df[col] > 200.0
                df.loc[k_mask, col] = df.loc[k_mask, col] - _KELVIN_OFFSET

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
        2. Drop implausible observed temperatures.
        3. Round valid_time to the representative whole hour.
        4. Sort by representative hour, candidate quality, and raw time so
           fills are time-directed and hourly dedup keeps the best candidate.
        5. Fill missing wind gust with wind speed.
        6. Zero-fill accumulation/flag columns and fill sky cover.
        7. Fill observed wind components from neighbouring rows.
        8. Deduplicate by hourly valid_time (keep first).
        """
        df = df.copy()

        # 1. Ensure valid_time is datetime. A :51 METAR represents the
        # nearest whole hour, e.g. 13:51 is stored as 14:00.
        if "valid_time" in df.columns:
            df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
            original_valid_time = df["valid_time"].copy()

            if "raw_tmpc_tenths" in df.columns:
                lo, hi = _IEM_TEMP_BOUNDS_C
                temp = pd.to_numeric(df["raw_tmpc_tenths"], errors="coerce")
                bad_temp = temp.notna() & ((temp < lo) | (temp > hi))
                if bad_temp.any():
                    logger.warning(
                        "cleaner: dropped {} rows with implausible IEM temperature "
                        "outside [{}, {}] C",
                        int(bad_temp.sum()),
                        lo,
                        hi,
                    )
                    df = df.loc[~bad_temp].copy()
                    original_valid_time = original_valid_time.loc[df.index].copy()

            df["valid_time"] = df["valid_time"].dt.round("h")
            representative_target = df["valid_time"] - pd.Timedelta(minutes=9)
            df["_iem_candidate_distance_s"] = (
                original_valid_time - representative_target
            ).abs().dt.total_seconds()
            df = df[
                df["_iem_candidate_distance_s"] <= _IEM_REPRESENTATIVE_TOLERANCE_S
            ].copy()
            original_valid_time = original_valid_time.loc[df.index].copy()
            df["_iem_original_valid_time"] = original_valid_time
            if "_raw_tmpc_source" in df.columns:
                df["_iem_tmpc_tgroup"] = (
                    df["_raw_tmpc_source"].fillna("").astype(str) == "tgroup"
                )
            else:
                df["_iem_tmpc_tgroup"] = False
            quality_cols = [
                c for c in df.columns
                if c not in {
                    "valid_time",
                    "_iem_candidate_distance_s",
                    "_iem_candidate_complete",
                    "_iem_original_valid_time",
                    "_iem_tmpc_tgroup",
                    "_raw_tmpc_source",
                    "_raw_dwpc_source",
                }
            ]
            if quality_cols:
                df["_iem_candidate_complete"] = df[quality_cols].notna().sum(axis=1)
            else:
                df["_iem_candidate_complete"] = 0

        # 2. Sort by valid_time before any fills so ffill/bfill are
        # time-directed regardless of row arrival order.
        if "valid_time" in df.columns:
            sort_cols = ["valid_time"]
            ascending = [True]
            if "_iem_candidate_distance_s" in df.columns:
                sort_cols.extend([
                    "_iem_tmpc_tgroup",
                    "_iem_candidate_distance_s",
                    "_iem_candidate_complete",
                    "_iem_original_valid_time",
                ])
                ascending.extend([False, True, False, True])
            df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        # 3. Fill missing gust with speed
        if "raw_wind_gust_kt" in df.columns and "raw_wind_spd_kt" in df.columns:
            df["raw_wind_gust_kt"] = df["raw_wind_gust_kt"].fillna(
                df["raw_wind_spd_kt"]
            )

        # 4. Zero-fill rain/snow flags and accumulation
        for col in _IEM_ZERO_FILL:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        if "skyc1" in df.columns:
            df["skyc1"] = df["skyc1"].ffill().fillna(0.0)

        # 5. Fill observed wind components after source parsing has preserved
        # missing/variable wind direction as NaN.
        for col in ["obs_U", "obs_V"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0.0)

        # 6. Hourly dedup: valid_time is already canonicalized to the hour.
        if "valid_time" in df.columns:
            df = df.drop_duplicates(subset="valid_time", keep="first")

        df = df.drop(
            columns=[
                c for c in [
                    "_iem_candidate_distance_s",
                    "_iem_candidate_complete",
                    "_iem_original_valid_time",
                    "_iem_tmpc_tgroup",
                    "_raw_tmpc_source",
                    "_raw_dwpc_source",
                ]
                if c in df.columns
            ]
        )

        return df.reset_index(drop=True)

    @staticmethod
    def build_features_hrrr(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived feature columns to a cleaned HRRR DataFrame.

        Steps applied in order (requires clean_hrrr to have been called first):
        1. Wind vector merge: U/V -> speed + sin/cos direction for 10m, 80m,
           and 500mb levels.
        2. log1p smoothing of heavy-tail gust/precip columns.

        The original columns are preserved alongside the new ones.
        """
        df = df.copy()

        # 1. Wind merging
        for u_col, v_col, prefix in [
            ("UGRD_10m", "VGRD_10m", "10m"),
            ("UGRD_80m", "VGRD_80m", "80m"),
            ("UGRD_500mb", "VGRD_500mb", "500mb"),
        ]:
            if u_col in df.columns and v_col in df.columns:
                u = df[u_col].fillna(0.0).values
                v = df[v_col].fillna(0.0).values
                ws, wds, wdc = Cleaner._wind_components(u, v)
                df[f"WS_{prefix}"] = ws
                df[f"WD_sin_{prefix}"] = wds
                df[f"WD_cos_{prefix}"] = wdc

        # 2. log1p on model heavy-tail columns
        if "GUST_sfc" in df.columns:
            df["log1p_GUST_sfc"] = np.log1p(
                df["GUST_sfc"].clip(lower=0.0)
            ).astype("float32")
        if "PRATE" in df.columns:
            df["log1p_PRATE"] = np.log1p(
                df["PRATE"].clip(lower=0.0)
            ).astype("float32")

        return df

    @staticmethod
    def build_features_iem(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived feature columns to a cleaned IEM observation DataFrame.

        Raw temperature/dew point columns are retained as canonical model
        inputs. Observed wind components are converted to speed + direction.
        """
        df = df.copy()
        if "obs_U" in df.columns and "obs_V" in df.columns:
            u = df["obs_U"].fillna(0.0).values
            v = df["obs_V"].fillna(0.0).values
            ws, wds, wdc = Cleaner._wind_components(u, v)
            df["obs_WS"] = ws
            df["obs_WD_sin"] = wds
            df["obs_WD_cos"] = wdc
        return df

    @staticmethod
    def compute_temp_error(df: pd.DataFrame) -> pd.DataFrame:
        """Add temp_error = observed temperature - HRRR TMP_2m to a merged frame."""
        df = df.copy()
        if "TMP_2m" not in df.columns:
            return df
        obs_col = "raw_tmpc_tenths" if "raw_tmpc_tenths" in df.columns else "tmpc"
        if obs_col not in df.columns:
            return df
        df["temp_error"] = (df[obs_col] - df["TMP_2m"]).astype("float32")
        return df

    @staticmethod
    def drop_cold_start(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Drop the first n rows per dataset (cold-start period).

        Parameters
        ----------
        df:
            Input DataFrame sorted by its primary time column.
        n:
            Number of leading rows to discard.

        Returns
        -------
        DataFrame with the first n rows removed, index reset.
        """
        if n <= 0 or df.empty:
            return df.reset_index(drop=True)
        return df.iloc[n:].reset_index(drop=True)

    @staticmethod
    def process_hrrr(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and feature-engineer a raw HRRR DataFrame for storage.

        Applies clean_hrrr then build_features_hrrr in the correct order.
        Use this instead of calling the two steps separately.
        """
        return Cleaner.build_features_hrrr(Cleaner.clean_hrrr(df))

    @staticmethod
    def process_iem(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and feature-engineer a raw IEM obs DataFrame for storage.

        Applies clean_iem then build_features_iem in the correct order.
        Use this instead of calling the two steps separately.
        """
        return Cleaner.build_features_iem(Cleaner.clean_iem(df))
