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

from model.features import HRRR_KNOWN_REALS, OBS_UNKNOWN_REALS  # noqa: F401
from model.features import TARGET  # noqa: F401  (re-exported for callers)
from model.features import TFT_KNOWN_REALS, TFT_UNKNOWN_REALS

# ---------------------------------------------------------------------------
# Column definitions (must match the training dataset exactly)
# ---------------------------------------------------------------------------

KNOWN_REALS: list[str] = TFT_KNOWN_REALS
UNKNOWN_REALS: list[str] = TFT_UNKNOWN_REALS

MAX_ENCODER_LENGTH    = 72
MAX_PREDICTION_LENGTH = 48


# ---------------------------------------------------------------------------
# NullTftModel -- placeholder when checkpoint is not yet configured
# ---------------------------------------------------------------------------

class NullTftModel:
    """Placeholder TFT model used when a checkpoint has not been loaded.

    Satisfies the ModelPipeline interface (has a predict() method) so
    ModelPipeline.__init__ does not reject it, but raises NotImplementedError
    if predict() is called.  Replace with a real TftModel once a trained
    checkpoint is available.
    """

    def predict(
        self,
        encoder_df: pd.DataFrame,
        decoder_df: pd.DataFrame,
    ) -> dict[int, dict[str, float]]:
        raise NotImplementedError(
            "TFT model has not been loaded. "
            "Provide a trained TftModel checkpoint via Settings.model_store_dir."
        )


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
        # Validate time_idx presence, then sort both frames ascending.
        if "time_idx" not in encoder_df.columns:
            raise ValueError("encoder_df must have a 'time_idx' column")
        if "time_idx" not in decoder_df.columns:
            raise ValueError("decoder_df must have a 'time_idx' column")
        encoder_df = encoder_df.sort_values("time_idx").reset_index(drop=True)
        decoder_df = decoder_df.sort_values("time_idx").reset_index(drop=True)

        # Validate row counts.
        if len(encoder_df) != MAX_ENCODER_LENGTH:
            raise ValueError(
                f"encoder_df must have exactly {MAX_ENCODER_LENGTH} rows, "
                f"got {len(encoder_df)}"
            )
        if len(decoder_df) != MAX_PREDICTION_LENGTH:
            raise ValueError(
                f"decoder_df must have exactly {MAX_PREDICTION_LENGTH} rows, "
                f"got {len(decoder_df)}"
            )

        # Validate decoder fxx completeness.
        if "fxx" not in decoder_df.columns:
            raise ValueError("decoder_df must have a 'fxx' column")
        actual_fxx = set(decoder_df["fxx"].astype(int).tolist())
        expected_fxx = set(range(1, MAX_PREDICTION_LENGTH + 1))
        if actual_fxx != expected_fxx:
            raise ValueError(
                f"decoder_df fxx must be exactly {{1..{MAX_PREDICTION_LENGTH}}}, "
                f"got {sorted(actual_fxx)}"
            )

        # Validate TMP_2m in decoder: must be present and contain no NaN values.
        if "TMP_2m" not in decoder_df.columns:
            raise ValueError("decoder_df must have a 'TMP_2m' column")
        if decoder_df["TMP_2m"].isna().any():
            raise ValueError("decoder_df 'TMP_2m' must not contain NaN values")

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
        # preds_q shape: (1, 48, n_quantiles) -> quantiles [p10, p50, p90]
        arr = preds_q[0].cpu().numpy() if hasattr(preds_q[0], "cpu") else preds_q[0]

        # Validate output shape: must be (MAX_PREDICTION_LENGTH, 3).
        if arr.ndim != 2 or arr.shape[0] != MAX_PREDICTION_LENGTH or arr.shape[1] < 3:
            raise ValueError(
                f"TFT output shape must be ({MAX_PREDICTION_LENGTH}, >=3), "
                f"got {arr.shape}"
            )

        # Un-residualise: add TMP_2m (HRRR forecast) back to get absolute temperature.
        # decoder_df is sorted by time_idx; sort by fxx to align index 0 -> fxx=1.
        tmp2m_vec = (
            decoder_df.sort_values("fxx").reset_index(drop=True)["TMP_2m"]
            .values.astype(float)
        )
        arr = arr + tmp2m_vec[:, None]  # broadcast: (48, n_quantiles)

        result: dict[int, dict[str, float]] = {}
        for i, fxx in enumerate(range(1, MAX_PREDICTION_LENGTH + 1)):
            result[fxx] = {
                "p10": float(arr[i, 0]),
                "p50": float(arr[i, 1]),
                "p90": float(arr[i, 2]),
            }
        return result
