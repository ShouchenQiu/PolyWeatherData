"""TftTrainer: read ParquetStore data, train TFT, and publish to ModelRegistry.

Ported from Herbie-main/tft_step9_train.py (training) and
tft_step7_dataloader.py (dataset construction).

The training uses pytorch-forecasting's TemporalFusionTransformer.  The
class gracefully falls back to logged warnings when pytorch-forecasting
or torch-lightning is not installed, so that the module can always be
imported even in lightweight environments.

Usage (CLI):
    python3 -m scripts.train_tft --parquet-dir data_store/KLGA --output-dir model/store

Usage (API):
    from scripts.train_tft import TftTrainer
    trainer = TftTrainer(store, registry)
    trainer.run(station_id="KLGA", version="v1")
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from config.settings import Settings
from model.features import HRRR_KNOWN_REALS, OBS_UNKNOWN_REALS

if TYPE_CHECKING:
    from data.storage.parquet_store import ParquetStore
    from model.store.registry import ModelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TFT hyper-parameters (matching the reference run_21 configuration)
# ---------------------------------------------------------------------------
_HIDDEN_SIZE            = 32
_ATTENTION_HEAD_SIZE    = 4
_DROPOUT                = 0.20
_HIDDEN_CONTINUOUS_SIZE = 16
_LEARNING_RATE          = 5e-4
_REDUCE_LR_PATIENCE     = 3
_WEIGHT_DECAY           = 1e-3
_QUANTILES              = [0.1, 0.5, 0.9]
_BATCH_SIZE             = 32
_MAX_EPOCHS             = 20
_EARLY_STOP_PATIENCE    = 2
_MIN_EPOCHS             = 5
_TARGET_WINSOR          = 8.0
_MAX_ENCODER_LENGTH     = 72
_MAX_PREDICTION_LENGTH  = 48

_TARGET      = "temp_error"
_GROUP_IDS   = ["init_time_str"]
_STATIC_CATS = ["Node_ID"]

_TEMP_K_COLS = ["TMP_2m", "DPT_2m", "TMP_sfc", "TMP_850mb", "TMP_500mb"]

_KNOWN_REALS: list[str] = [
    "local_idx",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    *HRRR_KNOWN_REALS,
]

_UNKNOWN_REALS: list[str] = OBS_UNKNOWN_REALS


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _wind_components(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (speed, sin_dir, cos_dir) from U/V component arrays."""
    ws   = np.sqrt(u ** 2 + v ** 2)
    wd   = np.arctan2(-u, -v)
    calm = ws == 0.0
    sin_wd = np.where(calm, 0.0, np.sin(wd)).astype("float32")
    cos_wd = np.where(calm, 0.0, np.cos(wd)).astype("float32")
    return ws.astype("float32"), sin_wd, cos_wd


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to a raw 72-step dataset DataFrame."""
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    df["init_time_str"] = df["init_time"].astype(str)
    df["Node_ID"]       = df["Node_ID"].astype(str)
    df["local_idx"]     = (df["fxx"] + 71).astype(int)

    # Kelvin -> Celsius for HRRR temperature columns (values > 200 K are real)
    for col in _TEMP_K_COLS:
        if col in df.columns:
            mask = df[col] > 200.0
            df.loc[mask, col] = (df.loc[mask, col] - 273.15).astype("float32")

    # Wind components
    for u_col, v_col, prefix in [
        ("UGRD_10m",  "VGRD_10m",  "10m"),
        ("UGRD_80m",  "VGRD_80m",  "80m"),
        ("UGRD_500mb","VGRD_500mb","500mb"),
    ]:
        u = df[u_col].fillna(0).values
        v = df[v_col].fillna(0).values
        ws, wds, wdc = _wind_components(u, v)
        df[f"WS_{prefix}"]      = ws
        df[f"WD_sin_{prefix}"]  = wds
        df[f"WD_cos_{prefix}"]  = wdc

    if "obs_U" in df.columns and "obs_V" in df.columns:
        ou = df["obs_U"].fillna(0).values
        ov = df["obs_V"].fillna(0).values
        ws, wds, wdc = _wind_components(ou, ov)
        df["obs_WS"]     = ws
        df["obs_WD_sin"] = wds
        df["obs_WD_cos"] = wdc

    # Log-smoothed heavy-tail variables
    df["log1p_GUST_sfc"] = np.log1p(df["GUST_sfc"].clip(lower=0)).astype("float32") \
        if "GUST_sfc" in df.columns else 0.0
    df["log1p_PRATE"]    = np.log1p(df["PRATE"].clip(lower=0)).astype("float32") \
        if "PRATE" in df.columns else 0.0

    # Residual target: TFT predicts (observed_temp - HRRR_forecast_temp)
    if "raw_tmpc_tenths" in df.columns and "TMP_2m" in df.columns:
        df["temp_error"] = (df["raw_tmpc_tenths"] - df["TMP_2m"]).astype("float32")

    # Ensure all continuous columns are float32
    for col in _KNOWN_REALS + _UNKNOWN_REALS + [_TARGET]:
        if col in df.columns and col != "local_idx":
            df[col] = df[col].astype("float32")

    return df


# ---------------------------------------------------------------------------
# TftTrainer
# ---------------------------------------------------------------------------

class TftTrainer:
    """Train a Temporal Fusion Transformer from ParquetStore data.

    Parameters
    ----------
    store:
        ParquetStore instance holding the merged HRRR + IEM 72-step dataset.
    registry:
        ModelRegistry instance where the trained checkpoint will be published.
    output_dir:
        Directory used for Lightning checkpoints during training.
    """

    def __init__(
        self,
        store:      "ParquetStore",
        registry:   "ModelRegistry",
        output_dir: Path | None = None,
    ) -> None:
        self._store     = store
        self._registry  = registry
        self._out_dir   = Path(output_dir) if output_dir else Path("checkpoints")
        self._out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        station_id:  str = "KLGA",
        version:     str | None = None,
        parquet_key: str = "tft_72h",
        start:       str = "2000-01-01",
        end:         str = "2100-01-01",
    ) -> dict:
        """Load data, train the model, and publish to the registry.

        Parameters
        ----------
        station_id:
            Station key used to locate data in ParquetStore.
        version:
            Version string for the registry (auto-generated from UTC time
            if None).
        parquet_key:
            Sub-directory / key hint for reading the 72-step dataset.
        start / end:
            Date range for data selection (ISO strings).

        Returns
        -------
        Dict with training results: version, best_val_loss, epochs_run,
        checkpoint_path, train_mae, test_mae.
        """
        if version is None:
            version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

        try:
            import torch
            import lightning.pytorch as pl
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.metrics import QuantileLoss
            from pytorch_forecasting.data.encoders import (
                NaNLabelEncoder,
                TorchNormalizer,
            )
            from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        except ImportError as exc:
            logger.error(
                "pytorch-forecasting / lightning not installed: %s. "
                "Install with: pip install pytorch-forecasting lightning",
                exc,
            )
            raise

        pl.seed_everything(42, workers=True)

        # -- Load and prepare data -------------------------------------------
        df_full = self._load_dataset(
            station_id, start=start, end=end
        )
        df_train = _prepare_df(df_full[df_full["split"] == "train"])
        df_val   = _prepare_df(df_full[df_full["split"] == "val"])
        df_test  = _prepare_df(df_full[df_full["split"] == "test"])

        # Winsorise training target
        before_count = int((df_train[_TARGET].abs() > _TARGET_WINSOR).sum())
        df_train[_TARGET] = df_train[_TARGET].clip(-_TARGET_WINSOR, _TARGET_WINSOR)
        logger.info("Winsorised %d train rows with |%s|>%s", before_count, _TARGET, _TARGET_WINSOR)

        # -- Build TimeSeriesDataSet -----------------------------------------
        known  = [c for c in _KNOWN_REALS  if c in df_train.columns]
        unk    = [c for c in _UNKNOWN_REALS if c in df_train.columns]
        training = TimeSeriesDataSet(
            df_train,
            time_idx                          = "local_idx",
            target                            = _TARGET,
            group_ids                         = _GROUP_IDS,
            max_encoder_length                = _MAX_ENCODER_LENGTH,
            max_prediction_length             = _MAX_PREDICTION_LENGTH,
            static_categoricals               = _STATIC_CATS,
            static_reals                      = [],
            time_varying_known_reals          = known,
            time_varying_unknown_reals        = unk,
            time_varying_known_categoricals   = [],
            time_varying_unknown_categoricals = [],
            target_normalizer                 = TorchNormalizer(method="standard"),
            categorical_encoders              = {
                "init_time_str": NaNLabelEncoder(add_nan=True),
                "Node_ID":       NaNLabelEncoder(add_nan=True),
            },
            add_relative_time_idx   = True,
            add_target_scales       = True,
            add_encoder_length      = True,
            allow_missing_timesteps = False,
        )
        validation = TimeSeriesDataSet.from_dataset(
            training, df_val,  predict=False, stop_randomization=True
        )
        testing = TimeSeriesDataSet.from_dataset(
            training, df_test, predict=True,  stop_randomization=True
        )

        train_loader = training.to_dataloader(
            train=True,  batch_size=_BATCH_SIZE, num_workers=0, shuffle=True
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=_BATCH_SIZE, num_workers=0
        )
        test_loader = testing.to_dataloader(
            train=False, batch_size=_BATCH_SIZE, num_workers=0
        )

        # -- Instantiate model -----------------------------------------------
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size                = _HIDDEN_SIZE,
            attention_head_size        = _ATTENTION_HEAD_SIZE,
            dropout                    = _DROPOUT,
            hidden_continuous_size     = _HIDDEN_CONTINUOUS_SIZE,
            loss                       = QuantileLoss(quantiles=_QUANTILES),
            learning_rate              = _LEARNING_RATE,
            log_interval               = 10,
            log_val_interval           = 1,
            reduce_on_plateau_patience = _REDUCE_LR_PATIENCE,
            weight_decay               = _WEIGHT_DECAY,
        )

        # -- Callbacks -------------------------------------------------------
        ckpt_dir  = self._out_dir / version
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(
            monitor   = "val_loss",
            patience  = _EARLY_STOP_PATIENCE,
            min_delta = 5e-4,
            mode      = "min",
            verbose   = True,
        )
        checkpoint = ModelCheckpoint(
            dirpath   = str(ckpt_dir),
            filename  = "tft_best_{epoch:02d}_{val_loss:.4f}",
            monitor   = "val_loss",
            mode      = "min",
            save_top_k= 1,
            verbose   = True,
        )

        # -- Train -----------------------------------------------------------
        trainer = pl.Trainer(
            accelerator         = "auto",
            precision           = 32,
            min_epochs          = _MIN_EPOCHS,
            max_epochs          = _MAX_EPOCHS,
            callbacks           = [early_stop, checkpoint],
            enable_progress_bar = True,
            log_every_n_steps   = 10,
            gradient_clip_val   = 0.1,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_ckpt  = Path(checkpoint.best_model_path)
        best_loss  = float(checkpoint.best_model_score)
        epochs_run = trainer.current_epoch

        # -- Evaluate on test set (MAE on median quantile) -------------------
        test_mae = self._eval_mae(trainer, tft, test_loader)
        train_mae = best_loss  # proxy (quantile loss, not exact MAE)

        # -- Publish to registry ---------------------------------------------
        metrics = {
            "train_mae":  train_mae,
            "test_mae":   test_mae,
            "best_val_loss": best_loss,
            "epochs_run": epochs_run,
        }
        self._registry.publish(
            version      = version,
            ckpt_path    = best_ckpt,
            metrics      = metrics,
        )

        result = {
            "version":          version,
            "best_val_loss":    best_loss,
            "epochs_run":       epochs_run,
            "checkpoint_path":  str(best_ckpt),
            "train_mae":        train_mae,
            "test_mae":         test_mae,
        }
        logger.info("TftTrainer finished: %s", result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(
        self, station_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """Load the pre-built 72-step TFT dataset from the ParquetStore."""
        df = self._store.read(
            station_id,
            start=pd.Timestamp(start, tz="UTC"),
            end=pd.Timestamp(end, tz="UTC"),
        )
        if df.empty:
            raise ValueError(
                f"No data found for station {station_id!r} in range {start}..{end}. "
                "Run batch_download_zarr.py and batch_download_iem.py first."
            )
        return df

    def _eval_mae(self, trainer: Any, model: Any, loader: Any) -> float:
        """Compute MAE on the median quantile over the test loader."""
        try:
            preds = trainer.predict(model, dataloaders=loader)
            # preds is a list of tensors; median = quantile index 1 (0.5)
            import torch
            all_pred = torch.cat([p[0][:, :, 1] for p in preds])  # (N, T)
            all_tgt  = torch.cat([p[1][0] for p in preds])        # (N, T)
            mae = float((all_pred - all_tgt).abs().mean())
            return round(mae, 5)
        except Exception as exc:
            logger.warning("Could not compute test MAE: %s", exc)
            return float("nan")


# Satisfy type checker for _eval_mae Any annotation
from typing import Any  # noqa: E402


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train TFT model from ParquetStore data")
    parser.add_argument("--station",    default="KLGA",   help="Station key in ParquetStore")
    parser.add_argument("--version",    default=None,     help="Model version string")
    parser.add_argument("--start",      default="2000-01-01")
    parser.add_argument("--end",        default="2100-01-01")
    parser.add_argument("--output-dir", default="checkpoints", dest="output_dir")
    args = parser.parse_args()

    from data.storage.parquet_store import ParquetStore
    from model.store.registry import ModelRegistry
    from db.schema import DbClient

    settings  = Settings()
    store     = ParquetStore(settings.data_store_dir)
    db_path   = settings.project_root / "polyweather.db"
    db        = DbClient(str(db_path))
    registry  = ModelRegistry(settings.model_store_dir, db)

    trainer = TftTrainer(store, registry, output_dir=Path(args.output_dir))
    result  = trainer.run(
        station_id = args.station,
        version    = args.version,
        start      = args.start,
        end        = args.end,
    )
    print(json.dumps(result, indent=2))
    db.close()


if __name__ == "__main__":
    _cli()
