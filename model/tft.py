"""TFT model wrapper.

Loads a pytorch-forecasting TemporalFusionTransformer checkpoint and
provides a clean predict() interface that returns per-fxx quantiles.

pytorch_forecasting is an optional heavy dependency; the module still
imports cleanly in test environments that lack it.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Heavy optional imports -- lazy so the module can be imported in tests
# without torch / pytorch_forecasting installed.
try:
    import torch as _torch
    from pytorch_forecasting import (  # type: ignore[import-untyped]
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )
    _PF_AVAILABLE = True
except ImportError:
    _PF_AVAILABLE = False

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Column definitions (must match the training dataset exactly)
# ---------------------------------------------------------------------------

_TEMP_K_COLS = ["TMP_2m", "DPT_2m", "TMP_sfc", "TMP_850mb", "TMP_500mb"]

KNOWN_REALS = [
    "local_idx",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
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

UNKNOWN_REALS = [
    "raw_tmpc_tenths",
    "raw_dwpc_tenths",  "raw_wind_spd_kt", "raw_wind_gust_kt",
    "obs_WS",           "obs_WD_sin",      "obs_WD_cos",
    "raw_vis_sm",       "raw_alti_inhg",
    "skyc1",            "skyl1",
    "p01i",             "is_rain",         "is_snow",
]

TARGET = "temp_error"

MAX_ENCODER_LENGTH    = 24
MAX_PREDICTION_LENGTH = 48


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _wind_components(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ws    = np.sqrt(u ** 2 + v ** 2).astype("float32")
    wd    = np.arctan2(-u, -v)
    calm  = ws == 0.0
    sin_wd = np.where(calm, 0.0, np.sin(wd)).astype("float32")
    cos_wd = np.where(calm, 0.0, np.cos(wd)).astype("float32")
    return ws, sin_wd, cos_wd


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-engineer a 72-row encoder+decoder DataFrame.

    Steps
    -----
    1. Convert Kelvin temperature columns to Celsius where value > 200.
    2. Decompose HRRR and obs wind U/V into speed + sin/cos direction.
    3. Log-transform GUST and PRATE.
    4. Compute temp_error = raw_tmpc_tenths - TMP_2m  (residual target).
    5. Compute local_idx = fxx + 71  (matches training window).
    6. Add calendar cyclical features from valid_time.
    7. Ensure all model columns exist and are float32.
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    # 1. Kelvin -> Celsius
    for col in _TEMP_K_COLS:
        if col in df.columns:
            mask = df[col] > 200.0
            df.loc[mask, col] = (df.loc[mask, col] - 273.15).astype("float32")

    # 2. HRRR wind decomposition
    for u_col, v_col, tag in [
        ("UGRD_10m",   "VGRD_10m",   "10m"),
        ("UGRD_80m",   "VGRD_80m",   "80m"),
        ("UGRD_500mb", "VGRD_500mb", "500mb"),
    ]:
        u = df.get(u_col, pd.Series(0.0, index=df.index)).fillna(0).values
        v = df.get(v_col, pd.Series(0.0, index=df.index)).fillna(0).values
        ws, wds, wdc = _wind_components(u, v)
        df[f"WS_{tag}"]      = ws
        df[f"WD_sin_{tag}"]  = wds
        df[f"WD_cos_{tag}"]  = wdc

    # Obs wind decomposition
    ou = df.get("obs_U", pd.Series(0.0, index=df.index)).fillna(0).values
    ov = df.get("obs_V", pd.Series(0.0, index=df.index)).fillna(0).values
    ws, wds, wdc = _wind_components(ou, ov)
    df["obs_WS"]     = ws
    df["obs_WD_sin"] = wds
    df["obs_WD_cos"] = wdc

    # 3. Log transforms
    df["log1p_GUST_sfc"] = np.log1p(
        df.get("GUST_sfc", pd.Series(0.0, index=df.index)).clip(lower=0)
    ).astype("float32")
    df["log1p_PRATE"] = np.log1p(
        df.get("PRATE", pd.Series(0.0, index=df.index)).clip(lower=0)
    ).astype("float32")

    # 4. temp_error residual
    tmp2m = df.get("TMP_2m", pd.Series(0.0, index=df.index)).fillna(0)
    tmpc  = df.get("raw_tmpc_tenths", pd.Series(0.0, index=df.index)).fillna(0)
    df[TARGET] = (tmpc - tmp2m).astype("float32")

    # 5. local_idx
    df["local_idx"] = (df["fxx"] + 71).astype(int)

    # 6. Calendar cyclical features from valid_time
    if "valid_time" in df.columns:
        vt = pd.to_datetime(df["valid_time"], utc=True)
        hour = vt.dt.hour.values
        doy  = vt.dt.day_of_year.values
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype("float32")
        df["sin_doy"]  = np.sin(2 * np.pi * doy  / 365).astype("float32")
        df["cos_doy"]  = np.cos(2 * np.pi * doy  / 365).astype("float32")
    else:
        for col in ("sin_hour", "cos_hour", "sin_doy", "cos_doy"):
            df[col] = 0.0

    # 7. Ensure all model columns are float32
    for col in KNOWN_REALS + UNKNOWN_REALS + [TARGET]:
        if col in df.columns and col != "local_idx":
            df[col] = df[col].astype("float32")
        elif col not in df.columns and col != "local_idx":
            df[col] = np.float32(0.0)

    # Required group / metadata columns
    if "init_time_str" not in df.columns:
        df["init_time_str"] = (
            df["init_time"].astype(str) if "init_time" in df.columns
            else "unknown"
        )
    if "Node_ID" not in df.columns:
        df["Node_ID"] = "KLGA"

    return df


# ---------------------------------------------------------------------------
# TftModel
# ---------------------------------------------------------------------------

class TftModel:
    """Load a pytorch-forecasting TFT checkpoint and run inference.

    Parameters
    ----------
    checkpoint_path:
        Path to the .ckpt file produced by pytorch-lightning.
    training_dataset_pkl:
        Path to the pickled TimeSeriesDataSet used during training.
        Required so that feature normalisers are loaded consistently.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        training_dataset_pkl: Path,
    ) -> None:
        if not _PF_AVAILABLE:
            raise ImportError(
                "pytorch_forecasting and torch are required for TftModel. "
                "Install with: pip install pytorch-forecasting torch"
            )
        self._ckpt_path   = Path(checkpoint_path)
        self._pkl_path    = Path(training_dataset_pkl)
        self._tft         = None  # loaded lazily
        self._training_ds = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._tft is not None:
            return
        with open(self._pkl_path, "rb") as f:
            self._training_ds = pickle.load(f)
        self._tft = TemporalFusionTransformer.load_from_checkpoint(
            str(self._ckpt_path)
        )
        self._tft.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        encoder_df: pd.DataFrame,
        decoder_df: pd.DataFrame,
    ) -> dict[int, dict[str, float]]:
        """Run TFT inference for one init cycle.

        Parameters
        ----------
        encoder_df:
            24 rows, fxx = -23..0  (contains IEM observations + HRRR zeroed)
        decoder_df:
            48 rows, fxx = 1..48   (contains HRRR forecast values)

        Returns
        -------
        {fxx: {"p10": float, "p50": float, "p90": float}}
            Quantile predictions in degC (post-processed, before MC).
            TFT output is temp_error (residual); caller should add TMP_2m
            if absolute temperature is needed.
        """
        self._ensure_loaded()

        combined = pd.concat(
            [encoder_df, decoder_df], ignore_index=True
        )
        combined = _prep(combined)

        testing = TimeSeriesDataSet.from_dataset(
            self._training_ds,
            combined,
            predict=True,
            stop_randomization=True,
        )
        loader = testing.to_dataloader(
            train=False, batch_size=1, num_workers=0
        )

        preds_q = self._tft.predict(loader, mode="quantiles")
        # preds_q shape: (1, 48, 3)  -> quantiles [p10, p50, p90]
        arr = preds_q[0].cpu().numpy() if hasattr(preds_q[0], "cpu") else preds_q[0]
        # arr shape: (48, 3)

        result: dict[int, dict[str, float]] = {}
        for i, fxx in enumerate(range(1, 49)):
            result[fxx] = {
                "p10": float(arr[i, 0]),
                "p50": float(arr[i, 1]),
                "p90": float(arr[i, 2]),
            }
        return result
