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

MAX_ENCODER_LENGTH    = 72
MAX_PREDICTION_LENGTH = 48


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

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
            72 rows, covering the 72-hour lookback window ending at init_time.
            Must already contain all model columns (time_idx, sin_hour, etc.)
            added by ModelPipeline._add_model_features().
        decoder_df:
            48 rows, fxx = 1..48 (contains HRRR forecast values + model features).

        Returns
        -------
        {fxx: {"p10": float, "p50": float, "p90": float}}
            Quantile predictions in degC (absolute temperature).
            TFT output (temp_error residuals) is un-residualised by adding
            TMP_2m from decoder_df before returning.
        """
        self._ensure_loaded()

        combined = pd.concat(
            [encoder_df, decoder_df], ignore_index=True
        )

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
        # arr shape: (48, 3) -- temp_error residuals in degC

        # Un-residualise: add TMP_2m (HRRR forecast) back to get absolute temperature.
        # decoder_df is sorted by fxx 1..48; index 0 corresponds to fxx=1.
        decoder_sorted = decoder_df.sort_values("fxx").reset_index(drop=True)
        if "TMP_2m" in decoder_sorted.columns:
            tmp2m_vec = decoder_sorted["TMP_2m"].fillna(0.0).values.astype(float)
        else:
            tmp2m_vec = np.zeros(48, dtype=float)
        arr = arr + tmp2m_vec[:, None]  # broadcast: (48, 3)

        result: dict[int, dict[str, float]] = {}
        for i, fxx in enumerate(range(1, 49)):
            result[fxx] = {
                "p10": float(arr[i, 0]),
                "p50": float(arr[i, 1]),
                "p90": float(arr[i, 2]),
            }
        return result
