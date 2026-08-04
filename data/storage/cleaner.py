from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# IEM numeric columns that should be zero-filled when missing
_IEM_ZERO_FILL = ["is_rain", "is_snow", "p01i"]
_IEM_TEMP_BOUNDS_C = (-40.0, 55.0)
_IEM_REPRESENTATIVE_TOLERANCE_S = 20 * 60


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
    def build_features_iem(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived feature columns to a cleaned IEM observation DataFrame.

        Raw temperature/dew point columns are retained as canonical model
        inputs. Observed wind components are converted to speed + direction,
        then sparse model-input gaps are repaired with provenance flags.
        """
        df = df.copy()
        if "obs_U" in df.columns and "obs_V" in df.columns:
            u = df["obs_U"].fillna(0.0).values
            v = df["obs_V"].fillna(0.0).values
            ws, wds, wdc = Cleaner._wind_components(u, v)
            df["obs_WS"] = ws
            df["obs_WD_sin"] = wds
            df["obs_WD_cos"] = wdc
        return Cleaner.repair_iem_model_inputs(df)

    @staticmethod
    def repair_iem_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
        """Repair sparse IEM wind/visibility gaps without hiding provenance.

        Missing raw wind speed is recovered from the already-derived vector
        speed, missing gust falls back to that repaired speed, and isolated
        visibility gaps are interpolated in time. Longer visibility outages
        remain missing so the training audit can reject them.
        """
        work = df.copy()
        original_index = work.index
        if "valid_time" in work.columns:
            work = work.assign(_repair_order=np.arange(len(work))).sort_values(
                "valid_time"
            )

        speed_missing = (
            work["raw_wind_spd_kt"].isna()
            if "raw_wind_spd_kt" in work.columns
            else pd.Series(False, index=work.index)
        )
        if speed_missing.any() and "obs_WS" in work.columns:
            work.loc[speed_missing, "raw_wind_spd_kt"] = work.loc[
                speed_missing, "obs_WS"
            ]

        gust_missing = (
            work["raw_wind_gust_kt"].isna()
            if "raw_wind_gust_kt" in work.columns
            else pd.Series(False, index=work.index)
        )
        if (
            gust_missing.any()
            and "raw_wind_spd_kt" in work.columns
        ):
            work.loc[gust_missing, "raw_wind_gust_kt"] = work.loc[
                gust_missing, "raw_wind_spd_kt"
            ]

        visibility_missing = (
            work["raw_vis_sm"].isna()
            if "raw_vis_sm" in work.columns
            else pd.Series(False, index=work.index)
        )
        if visibility_missing.any():
            visibility = pd.to_numeric(work["raw_vis_sm"], errors="coerce")
            work["raw_vis_sm"] = visibility.interpolate(
                method="linear",
                limit=2,
                limit_direction="both",
                limit_area="inside",
            )

        for column, missing in (
            ("raw_wind_spd_kt", speed_missing),
            ("raw_wind_gust_kt", gust_missing),
            ("raw_vis_sm", visibility_missing),
        ):
            if column not in work.columns:
                continue
            flag = f"{column}_imputed"
            previous = (
                work[flag].fillna(False).astype(bool)
                if flag in work.columns
                else pd.Series(False, index=work.index)
            )
            newly_filled = missing & work[column].notna()
            work[flag] = previous | newly_filled

        if "_repair_order" in work.columns:
            work = work.sort_values("_repair_order").drop(columns="_repair_order")
            work.index = original_index
        return work

    @staticmethod
    def process_iem(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and feature-engineer a raw IEM obs DataFrame for storage.

        Applies clean_iem then build_features_iem in the correct order.
        Use this instead of calling the two steps separately.
        """
        return Cleaner.build_features_iem(Cleaner.clean_iem(df))
