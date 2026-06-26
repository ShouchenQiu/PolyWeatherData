"""TftTrainer: read prepared training data, train TFT, and publish candidates.

Ported from Herbie-main/tft_step9_train.py (training) and
tft_step7_dataloader.py (dataset construction).

The training uses pytorch-forecasting's TemporalFusionTransformer.  The
class gracefully falls back to logged warnings when pytorch-forecasting
or torch-lightning is not installed, so that the module can always be
imported even in lightweight environments.

Usage (CLI):
    python3 -m scripts.train_tft --data-store data_store --model-store model_store

Usage (API):
    from model.training.trainer import TftTrainer
    trainer = TftTrainer(store, registry)
    trainer.run(station_id="KLGA", version="v1")
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from model.features import TARGET, TFT_KNOWN_REALS, TFT_UNKNOWN_REALS
from model.training.artifacts import write_training_artifacts
from model.training.config import ModelConfig, TrainerConfig, TrainingExperimentConfig

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

_TARGET      = TARGET
_GROUP_IDS   = ["init_time_str"]
_STATIC_CATS = ["Node_ID"]

_KNOWN_REALS: list[str] = TFT_KNOWN_REALS

_UNKNOWN_REALS: list[str] = TFT_UNKNOWN_REALS


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
        settings: Settings | None = None,
        data_manager: object | None = None,
        decoder_cache: object | None = None,
        dataset_builder: object | None = None,
        station_config: object | None = None,
        random_seed: int = 42,
        torch_deterministic: bool = True,
        precision: str | int = 32,
        num_workers: int = 0,
        activate_model: bool = True,
        experiment_config: TrainingExperimentConfig | None = None,
        gradient_clip_val: float = 0.1,
    ) -> None:
        self._store     = store
        self._registry  = registry
        self._settings  = settings if settings is not None else Settings()
        if experiment_config is not None:
            experiment_config.apply_to_settings(self._settings)
        self._data_mgr  = data_manager
        self._decoder_cache = decoder_cache
        self._dataset_builder = dataset_builder
        self._station_config = station_config
        self._random_seed = int(
            experiment_config.trainer.seed
            if experiment_config is not None else random_seed
        )
        self._torch_deterministic = bool(
            experiment_config.trainer.deterministic
            if experiment_config is not None else torch_deterministic
        )
        self._precision = (
            experiment_config.trainer.precision
            if experiment_config is not None else precision
        )
        self._num_workers = int(
            experiment_config.trainer.num_workers
            if experiment_config is not None else num_workers
        )
        self._activate_model = bool(activate_model)
        self._gradient_clip_val = float(
            experiment_config.trainer.gradient_clip_val
            if experiment_config is not None else gradient_clip_val
        )
        self._experiment_config = experiment_config or self._config_from_current_settings(
            station_id=getattr(self._station_config, "station_id", "KLGA"),
            runtime_profile="legacy_cli",
        )
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

        pl.seed_everything(self._random_seed, workers=True)
        if self._torch_deterministic and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)

        # -- Load and prepare data -------------------------------------------
        df_train, df_val, df_test = self._build_split_frames(station_id)

        # Winsorise training target
        target_winsor = self._settings.tft_target_winsor
        before_count = int((df_train[_TARGET].abs() > target_winsor).sum())
        df_train[_TARGET] = df_train[_TARGET].clip(-target_winsor, target_winsor)
        logger.info("Winsorised %d train rows with |%s|>%s", before_count, _TARGET, target_winsor)

        # -- Build TimeSeriesDataSet -----------------------------------------
        known  = [c for c in _KNOWN_REALS  if c in df_train.columns]
        unk    = [c for c in _UNKNOWN_REALS if c in df_train.columns]
        training = TimeSeriesDataSet(
            df_train,
            time_idx                          = "local_idx",
            target                            = _TARGET,
            group_ids                         = _GROUP_IDS,
            max_encoder_length                = self._settings.tft_max_encoder_length,
            max_prediction_length             = self._settings.tft_max_prediction_length,
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
            train=True,
            batch_size=self._settings.tft_batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=self._settings.tft_batch_size, num_workers=self._num_workers
        )
        test_loader = testing.to_dataloader(
            train=False, batch_size=self._settings.tft_batch_size, num_workers=self._num_workers
        )

        # -- Instantiate model -----------------------------------------------
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size                = self._settings.tft_hidden_size,
            attention_head_size        = self._settings.tft_attention_head_size,
            dropout                    = self._settings.tft_dropout,
            hidden_continuous_size     = self._settings.tft_hidden_continuous_size,
            loss                       = QuantileLoss(quantiles=self._settings.tft_quantiles),
            learning_rate              = self._settings.tft_learning_rate,
            log_interval               = 10,
            log_val_interval           = 1,
            reduce_on_plateau_patience = self._settings.tft_reduce_lr_patience,
            weight_decay               = self._settings.tft_weight_decay,
        )

        # -- Callbacks -------------------------------------------------------
        ckpt_dir  = self._out_dir / version
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(
            monitor   = "val_loss",
            patience  = self._settings.tft_early_stop_patience,
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
            precision           = self._precision,
            min_epochs          = self._settings.tft_min_epochs,
            max_epochs          = self._settings.tft_max_epochs,
            callbacks           = [early_stop, checkpoint],
            enable_progress_bar = True,
            log_every_n_steps   = 10,
            gradient_clip_val   = self._gradient_clip_val,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_ckpt  = Path(checkpoint.best_model_path)
        best_loss  = float(checkpoint.best_model_score)
        epochs_run = trainer.current_epoch
        training_dataset_pkl = ckpt_dir / "training_dataset.pkl"
        with open(training_dataset_pkl, "wb") as fh:
            pickle.dump(training, fh)

        # -- Evaluate on test set (MAE on median quantile) -------------------
        test_mae = self._eval_mae(trainer, tft, test_loader)
        train_mae = float("nan")

        # -- Publish to registry ---------------------------------------------
        metrics = {
            "train_mae":  train_mae,
            "test_mae":   test_mae,
            "best_val_loss": best_loss,
            "epochs_run": epochs_run,
        }
        result = {
            "version":          version,
            "best_val_loss":    best_loss,
            "epochs_run":       epochs_run,
            "checkpoint_path":  str(best_ckpt),
            "train_mae":        train_mae,
            "test_mae":         test_mae,
        }
        extra_artifacts = write_training_artifacts(
            output_dir=ckpt_dir,
            station_id=station_id,
            version=version,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            known=known,
            unknown=unk,
            training=training,
            result=result,
            torch_module=torch,
            lightning_module=pl,
            experiment_config=self._experiment_config,
            random_seed=self._random_seed,
            torch_deterministic=self._torch_deterministic,
            precision=self._precision,
            num_workers=self._num_workers,
            settings=self._settings,
        )
        self._registry.publish(
            version      = version,
            ckpt_path    = best_ckpt,
            metrics      = metrics,
            training_dataset_pkl = training_dataset_pkl,
            activate=self._activate_model,
            extra_artifacts=extra_artifacts,
        )
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
                "Run scripts.run_remote_training download/audit first."
            )
        return df

    def _build_split_frames(self, station_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._dataset_builder is None:
            if self._data_mgr is None or self._decoder_cache is None:
                raise ValueError(
                    "TftTrainer requires a TrainingDatasetBuilder or both "
                    "data_manager and decoder_cache. Legacy split-column "
                    "training data is no longer supported."
                )
            from model.training_data import TrainingDatasetBuilder
            self._dataset_builder = TrainingDatasetBuilder(
                self._data_mgr,
                self._decoder_cache,
                encoder_hours=self._settings.tft_max_encoder_length,
            )

        station = self._station_config or STATIONS[station_id]
        train = self._dataset_builder.build_frame(
            station,
            self._init_times(self._settings.train_start, self._settings.train_end),
            split="train",
        )
        val = self._dataset_builder.build_frame(
            station,
            self._init_times(self._settings.val_start, self._settings.val_end),
            split="val",
        )
        test = self._dataset_builder.build_frame(
            station,
            self._init_times(self._settings.test_start, self._settings.test_end),
            split="test",
        )
        if train.empty or val.empty or test.empty:
            raise ValueError(
                "TrainingDatasetBuilder produced an empty split: "
                f"train={len(train)} val={len(val)} test={len(test)}"
            )
        return train, val, test

    @staticmethod
    def _init_times(start_date: str, end_date: str) -> list[pd.Timestamp]:
        start_ts = pd.Timestamp(start_date + " 00:00:00", tz="UTC")
        end_ts = pd.Timestamp(end_date + " 23:00:00", tz="UTC")
        return list(pd.date_range(start_ts, end_ts, freq="h"))

    def _eval_mae(self, trainer: Any, model: Any, loader: Any) -> float:
        """Compute MAE on the median quantile over the test loader."""
        try:
            import torch

            prediction = model.predict(
                loader,
                return_y=True,
                trainer_kwargs={
                    "accelerator": "auto",
                    "logger": False,
                    "enable_progress_bar": False,
                },
            )
            all_pred = prediction.output
            if all_pred.ndim == 3:
                all_pred = all_pred[:, :, 1]
            target = prediction.y[0] if isinstance(prediction.y, tuple) else prediction.y
            if target is None:
                raise ValueError("prediction target is missing")
            all_tgt = target
            mae = float((all_pred - all_tgt).abs().mean())
            return round(mae, 5)
        except Exception as exc:
            logger.warning("Could not compute test MAE: %s", exc)
            return float("nan")

    def _config_from_current_settings(
        self,
        station_id: str,
        runtime_profile: str,
    ) -> TrainingExperimentConfig:
        return TrainingExperimentConfig(
            station=station_id,
            splits={
                "train": (self._settings.train_start, self._settings.train_end),
                "val": (self._settings.val_start, self._settings.val_end),
                "test": (self._settings.test_start, self._settings.test_end),
            },
            model=ModelConfig(
                hidden_size=self._settings.tft_hidden_size,
                attention_head_size=self._settings.tft_attention_head_size,
                dropout=self._settings.tft_dropout,
                hidden_continuous_size=self._settings.tft_hidden_continuous_size,
                learning_rate=self._settings.tft_learning_rate,
                weight_decay=self._settings.tft_weight_decay,
                reduce_lr_patience=self._settings.tft_reduce_lr_patience,
                quantiles=list(self._settings.tft_quantiles),
                target_winsor=self._settings.tft_target_winsor,
                max_encoder_length=self._settings.tft_max_encoder_length,
                max_prediction_length=self._settings.tft_max_prediction_length,
            ),
            trainer=TrainerConfig(
                min_epochs=self._settings.tft_min_epochs,
                max_epochs=self._settings.tft_max_epochs,
                batch_size=self._settings.tft_batch_size,
                seed=self._random_seed,
                deterministic=self._torch_deterministic,
                precision=str(self._precision),
                num_workers=self._num_workers,
                early_stop_patience=self._settings.tft_early_stop_patience,
                gradient_clip_val=self._gradient_clip_val,
            ),
            runtime_profile=runtime_profile,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _settings_from_cli_args(args: Any) -> Settings:
    """Build Settings from CLI overrides while preserving production defaults."""
    smoke_summary_path = getattr(args, "smoke_summary", None)
    smoke_summary: dict[str, Any] = {}
    if smoke_summary_path:
        with open(smoke_summary_path, encoding="utf-8") as fh:
            smoke_summary = json.load(fh)

    data_store = getattr(args, "data_store", None)
    if data_store is None and smoke_summary.get("data_store_dir"):
        data_store = smoke_summary["data_store_dir"]

    settings_kwargs: dict[str, Any] = {}
    if data_store is not None:
        settings_kwargs["data_store_dir"] = Path(data_store)

    model_store = getattr(args, "model_store", None)
    if model_store is not None:
        settings_kwargs["model_store_dir"] = Path(model_store)

    settings = Settings(**settings_kwargs)
    experiment_path = getattr(args, "experiment_config", None)
    experiment = (
        TrainingExperimentConfig.from_json(experiment_path)
        if experiment_path else None
    )

    smoke_split = smoke_summary.get("smoke_split")
    if smoke_split and experiment is None:
        for split_name in ("train", "val", "test"):
            bounds = smoke_split[split_name]
            if isinstance(bounds, dict):
                start = bounds["start"]
                end = bounds["end"]
            else:
                start, end = bounds
            setattr(settings, f"{split_name}_start", start)
            setattr(settings, f"{split_name}_end", end)

    if experiment is not None:
        experiment.apply_to_settings(settings)
    else:
        min_epochs = getattr(args, "min_epochs", None)
        if min_epochs is not None:
            settings.tft_min_epochs = int(min_epochs)
        max_epochs = getattr(args, "max_epochs", None)
        if max_epochs is not None:
            settings.tft_max_epochs = int(max_epochs)
        batch_size = getattr(args, "batch_size", None)
        if batch_size is not None:
            settings.tft_batch_size = int(batch_size)

    return settings


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train TFT model from ParquetStore data")
    parser.add_argument("--station",    default="KLGA",   help="Station key in ParquetStore")
    parser.add_argument("--version",    default=None,     help="Model version string")
    parser.add_argument("--start",      default="2000-01-01")
    parser.add_argument("--end",        default="2100-01-01")
    parser.add_argument("--output-dir", default="checkpoints", dest="output_dir")
    parser.add_argument("--data-store", default=None, dest="data_store")
    parser.add_argument("--db", default=None, dest="db")
    parser.add_argument("--model-store", default=None, dest="model_store")
    parser.add_argument("--smoke-summary", default=None, dest="smoke_summary")
    parser.add_argument("--min-epochs", type=int, default=None, dest="min_epochs")
    parser.add_argument("--max-epochs", type=int, default=None, dest="max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    parser.add_argument("--experiment-config", default=None, dest="experiment_config")
    parser.add_argument("--candidate-only", action="store_true", dest="candidate_only")
    args = parser.parse_args()

    from data.storage.parquet_store import ParquetStore
    from data.manager import DataManager
    from model.training_data import TrainingDecoderCache
    from model.store.registry import ModelRegistry
    from db.schema import DbClient

    settings  = _settings_from_cli_args(args)
    experiment = (
        TrainingExperimentConfig.from_json(args.experiment_config)
        if args.experiment_config else None
    )
    forecast_store = ParquetStore(settings.forecast_store_dir)
    obs_store = ParquetStore(settings.obs_store_dir)
    data_mgr = DataManager([], [], forecast_store, obs_store, settings=settings)
    decoder_cache = TrainingDecoderCache(settings.training_decoder_cache_dir)
    db_path   = Path(args.db) if args.db else settings.project_root / "polyweather.db"
    db        = DbClient(str(db_path))
    registry  = ModelRegistry(settings.model_store_dir, db)

    trainer = TftTrainer(
        forecast_store,
        registry,
        output_dir=Path(args.output_dir),
        settings=settings,
        data_manager=data_mgr,
        decoder_cache=decoder_cache,
        station_config=STATIONS[args.station],
        random_seed=args.seed,
        torch_deterministic=args.deterministic,
        precision=args.precision,
        num_workers=args.num_workers,
        activate_model=not args.candidate_only,
        experiment_config=experiment,
    )
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
