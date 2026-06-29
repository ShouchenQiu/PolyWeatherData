"""TftTrainer: read prepared training data, train TFT, and publish candidates.

Ported from Herbie-main/tft_step9_train.py (training) and
tft_step7_dataloader.py (dataset construction).

The training uses pytorch-forecasting's TemporalFusionTransformer.  The
class gracefully falls back to logged warnings when pytorch-forecasting
or torch-lightning is not installed, so that the module can always be
imported even in lightweight environments.

Usage (CLI):
    python3 -m scripts.train_tft --run-dir runs/example --version candidate-v1

Usage (API):
    from training.trainer import TftTrainer
    trainer = TftTrainer(store, registry)
    trainer.run_validation(station_id="KLGA", execution_id="trial-1")
"""
from __future__ import annotations

import json
import logging
import pickle
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from model.features import TARGET, TFT_KNOWN_REALS, TFT_UNKNOWN_REALS
from training.artifacts import write_training_artifacts
from training.config import (
    ExperimentTemplate,
    ResolvedTrialSnapshot,
)
from training.dataset import DECODER_WEIGHT_COLUMN
from training.experiment import FinalEvaluationApproval, ValidationExecutionLimits
from training.workspace import TrainingWorkspace

if TYPE_CHECKING:
    from data.storage.parquet_store import ParquetStore
    from model.store.registry import ModelRegistry
    from training.audit_frames import AuditedFrameCache

logger = logging.getLogger(__name__)

_TARGET      = TARGET
_GROUP_IDS   = ["init_time_str"]
_STATIC_CATS = ["Node_ID"]

_KNOWN_REALS: list[str] = TFT_KNOWN_REALS

_UNKNOWN_REALS: list[str] = TFT_UNKNOWN_REALS


def train_workspace(
    run_dir: Path | str,
    station_id: str,
    version: str,
    experiment_config: Path | str | None = None,
    train_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one audited train/validation execution without touching test."""
    workspace = TrainingWorkspace(run_dir)
    workspace.require_ready_audit()
    workspace.ensure()
    started_at = datetime.now(timezone.utc).isoformat()
    train = train_fn or _train_validation
    train_result = train(
        workspace=workspace,
        station_id=station_id,
        version=version,
        experiment_config=experiment_config,
    )
    workspace.write_json(
        workspace.run_dir / "validation_result.json",
        train_result,
    )
    result = {
        "status": "validated",
        "station": station_id,
        "version": version,
        "train_result": train_result,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    workspace.record_stage("validation", result)
    return result


def _train_validation(**kwargs: Any) -> dict[str, Any]:
    from data.manager import DataManager
    from data.storage.parquet_store import ParquetStore
    from db.schema import DbClient
    from model.store.registry import ModelRegistry
    from training.audit_frames import AuditedFrameCache
    from training.dataset import TrainingDecoderCache

    workspace: TrainingWorkspace = kwargs["workspace"]
    settings = Settings(
        data_store_dir=workspace.data_store_dir,
        model_store_dir=workspace.model_store_dir,
    )
    if not kwargs.get("experiment_config"):
        raise ValueError("validation execution requires --experiment-config")
    experiment = ExperimentTemplate.from_json(kwargs["experiment_config"])
    audit_summary = workspace.require_ready_audit()
    experiment.validate_audit(audit_summary)
    if kwargs["station_id"] != experiment.station:
        raise ValueError(
            f"requested station {kwargs['station_id']!r} does not match "
            f"template station {experiment.station!r}"
        )
    experiment.apply_to_settings(settings)

    forecast_store = ParquetStore(settings.forecast_store_dir)
    obs_store = ParquetStore(settings.obs_store_dir)
    data_manager = DataManager([], [], forecast_store, obs_store, settings=settings)
    decoder_cache = TrainingDecoderCache(settings.training_decoder_cache_dir)
    db = DbClient(str(workspace.db_path))
    try:
        trainer = TftTrainer(
            forecast_store,
            ModelRegistry(settings.model_store_dir, db),
            output_dir=kwargs.get("output_dir", workspace.checkpoints_dir),
            settings=settings,
            data_manager=data_manager,
            decoder_cache=decoder_cache,
            station_config=STATIONS[kwargs["station_id"]],
            audited_frame_cache=AuditedFrameCache(
                workspace.audit_frames_dir,
                workspace.data_store_dir,
            ),
            random_seed=experiment.trainer.seed,
            torch_deterministic=experiment.trainer.deterministic,
            precision=experiment.trainer.precision,
            num_workers=experiment.trainer.num_workers,
            experiment_config=experiment,
        )
        return trainer.run_validation(
            station_id=kwargs["station_id"],
            execution_id=kwargs["version"],
            limits=kwargs.get("limits"),
            progress_reporter=kwargs.get("progress_reporter"),
            trial_parameters=kwargs.get("trial_parameters"),
        )
    finally:
        db.close()


def _train_final(**kwargs: Any) -> dict[str, Any]:
    """Run the catalog-approved, test-reading final execution."""
    from data.manager import DataManager
    from data.storage.parquet_store import ParquetStore
    from db.schema import DbClient
    from model.store.registry import ModelRegistry
    from training.audit_frames import AuditedFrameCache
    from training.dataset import TrainingDecoderCache

    workspace: TrainingWorkspace = kwargs["workspace"]
    experiment = ExperimentTemplate.from_json(kwargs["experiment_config"])
    audit_summary = workspace.require_ready_audit()
    experiment.validate_audit(audit_summary)
    if kwargs["station_id"] != experiment.station:
        raise ValueError("final station does not match experiment template")
    settings = Settings(
        data_store_dir=workspace.data_store_dir,
        model_store_dir=workspace.model_store_dir,
    )
    experiment.apply_to_settings(settings)
    forecast_store = ParquetStore(settings.forecast_store_dir)
    obs_store = ParquetStore(settings.obs_store_dir)
    db = DbClient(str(workspace.db_path))
    try:
        trainer = TftTrainer(
            forecast_store,
            ModelRegistry(settings.model_store_dir, db),
            output_dir=kwargs.get("output_dir", workspace.checkpoints_dir),
            settings=settings,
            data_manager=DataManager(
                [], [], forecast_store, obs_store, settings=settings
            ),
            decoder_cache=TrainingDecoderCache(
                settings.training_decoder_cache_dir
            ),
            station_config=STATIONS[kwargs["station_id"]],
            audited_frame_cache=AuditedFrameCache(
                workspace.audit_frames_dir,
                workspace.data_store_dir,
            ),
            random_seed=experiment.trainer.seed,
            torch_deterministic=experiment.trainer.deterministic,
            precision=experiment.trainer.precision,
            num_workers=experiment.trainer.num_workers,
            experiment_config=experiment,
        )
        return trainer.run_final(
            station_id=kwargs["station_id"],
            version=kwargs["version"],
            approval=kwargs["approval"],
            trial_snapshot=kwargs["trial_snapshot"],
        )
    finally:
        db.close()


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
        audited_frame_cache: "AuditedFrameCache | None" = None,
        random_seed: int = 42,
        torch_deterministic: bool = True,
        precision: str | int = 32,
        num_workers: int = 0,
        experiment_config: ExperimentTemplate | None = None,
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
        self._audited_frame_cache = audited_frame_cache
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
        self._gradient_clip_val = float(
            experiment_config.trainer.gradient_clip_val
            if experiment_config is not None else gradient_clip_val
        )
        self._experiment_config = experiment_config
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
        """Reject the retired unapproved train/test/publish entry point."""
        raise PermissionError(
            "TftTrainer.run is locked; use run_validation or approved run_final"
        )

    def run_validation(
        self,
        station_id: str,
        execution_id: str,
        limits: ValidationExecutionLimits | None = None,
        progress_reporter: Callable[[dict[str, object]], None] | None = None,
        trial_parameters: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Fit against train/validation only and return validation evidence."""
        return self._run(
            station_id=station_id,
            version=execution_id,
            include_test=False,
            publish_candidate=False,
            limits=limits,
            progress_reporter=progress_reporter,
            trial_parameters=trial_parameters,
            final_approval=None,
            final_snapshot=None,
        )

    def run_final(
        self,
        station_id: str,
        version: str,
        approval: FinalEvaluationApproval | None,
        trial_snapshot: ResolvedTrialSnapshot | None = None,
    ) -> dict[str, Any]:
        """Run the only test-reading path after explicit final approval."""
        if approval is None:
            raise PermissionError("final approval is required")
        if not isinstance(approval, FinalEvaluationApproval):
            raise PermissionError(
                "final execution requires a verified final approval"
            )
        if not isinstance(trial_snapshot, ResolvedTrialSnapshot):
            raise PermissionError(
                "final execution requires the selected trial snapshot"
            )
        if (
            trial_snapshot.execution_type != "trial"
            or approval.study_id != trial_snapshot.study_id
            or approval.selected_trial_number != trial_snapshot.trial_number
            or approval.trial_snapshot_hash != trial_snapshot.snapshot_hash()
        ):
            raise PermissionError(
                "final approval does not match selected trial snapshot"
            )
        if self._experiment_config is None:
            raise PermissionError(
                "final candidate requires a versioned experiment template"
            )
        if trial_snapshot.template_hash != self._experiment_config.config_hash():
            raise PermissionError(
                "selected trial snapshot does not match experiment template"
            )
        return self._run(
            station_id=station_id,
            version=version,
            include_test=True,
            publish_candidate=True,
            limits=ValidationExecutionLimits.for_final(
                self._experiment_config,
                dict(trial_snapshot.parameters),
            ),
            progress_reporter=None,
            trial_parameters=dict(trial_snapshot.parameters),
            final_approval=approval,
            final_snapshot=trial_snapshot,
        )

    def _run(
        self,
        *,
        station_id: str,
        version: str | None,
        include_test: bool,
        publish_candidate: bool,
        limits: ValidationExecutionLimits | None,
        progress_reporter: Callable[[dict[str, object]], None] | None,
        trial_parameters: dict[str, object] | None,
        final_approval: FinalEvaluationApproval | None,
        final_snapshot: ResolvedTrialSnapshot | None,
    ) -> dict[str, Any]:
        if version is None:
            version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        if publish_candidate and not include_test:
            raise ValueError("candidate publishing requires final test evaluation")
        if trial_parameters is not None:
            if (
                limits is None
                or limits.execution_type not in {"trial", "final"}
            ):
                raise PermissionError(
                    "resolved parameters require a trial or final execution"
                )
            if self._experiment_config is None:
                raise PermissionError(
                    "trial parameters require an experiment template"
                )
            expected = set(self._experiment_config.study.search_space)
            if set(trial_parameters) != expected:
                raise ValueError(
                    "trial parameters must exactly match template search space"
                )
            setting_names = {
                "hidden_size": "tft_hidden_size",
                "attention_head_size": "tft_attention_head_size",
                "hidden_continuous_size": "tft_hidden_continuous_size",
                "dropout": "tft_dropout",
                "learning_rate": "tft_learning_rate",
                "weight_decay": "tft_weight_decay",
            }
            for name, setting_name in setting_names.items():
                setattr(
                    self._settings,
                    setting_name,
                    trial_parameters[name],
                )
        batch_size = (
            limits.batch_size
            if limits is not None
            else self._settings.tft_batch_size
        )
        num_workers = (
            limits.num_workers if limits is not None else self._num_workers
        )
        precision = limits.precision if limits is not None else self._precision
        min_epochs = (
            limits.min_epochs
            if limits is not None
            else self._settings.tft_min_epochs
        )
        max_epochs = (
            limits.max_epochs
            if limits is not None
            else self._settings.tft_max_epochs
        )

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
            torch.use_deterministic_algorithms(True, warn_only=True)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

        def report(**payload: object) -> None:
            if progress_reporter is not None:
                progress_reporter(payload)

        class ExecutionProgressCallback(pl.Callback):
            @staticmethod
            def _should_report(completed: int, total: int) -> bool:
                return (
                    completed == 1
                    or completed % 10 == 0
                    or completed == total
                )

            def on_train_batch_end(
                self,
                trainer: object,
                _module: object,
                _outputs: object,
                _batch: object,
                batch_idx: int,
            ) -> None:
                completed = batch_idx + 1
                total = int(getattr(trainer, "num_training_batches", 0))
                if not self._should_report(completed, total):
                    return
                report(
                    phase="fit",
                    epoch=int(getattr(trainer, "current_epoch", 0)),
                    train_batches_completed=completed,
                    train_batches_total=total,
                    gpu_memory_allocated_bytes=(
                        int(torch.cuda.memory_allocated())
                        if cuda_available
                        else 0
                    ),
                )

            def on_validation_batch_end(
                self,
                trainer: object,
                _module: object,
                _outputs: object,
                _batch: object,
                batch_idx: int,
                dataloader_idx: int = 0,
            ) -> None:
                if bool(getattr(trainer, "sanity_checking", False)):
                    return
                totals = getattr(trainer, "num_val_batches", [0])
                total = int(totals[dataloader_idx] if totals else 0)
                completed = batch_idx + 1
                if not self._should_report(completed, total):
                    return
                report(
                    phase="validate",
                    epoch=int(getattr(trainer, "current_epoch", 0)),
                    val_batches_completed=completed,
                    val_batches_total=total,
                    gpu_memory_allocated_bytes=(
                        int(torch.cuda.memory_allocated())
                        if cuda_available
                        else 0
                    ),
                )

            def on_validation_epoch_end(
                self,
                trainer: object,
                _module: object,
            ) -> None:
                if bool(getattr(trainer, "sanity_checking", False)):
                    return
                metrics = getattr(trainer, "callback_metrics", {})
                value = metrics.get("val_loss")
                if value is None:
                    return
                report(
                    phase="validation_epoch_end",
                    epoch=int(getattr(trainer, "current_epoch", 0)),
                    validation_quantile_loss=float(value),
                    gpu_memory_allocated_bytes=(
                        int(torch.cuda.memory_allocated())
                        if cuda_available
                        else 0
                    ),
                )

        # -- Load and prepare data -------------------------------------------
        report(phase="load_data")
        split_frames = self._build_split_frames(
            station_id,
            include_test=include_test,
        )
        df_train, df_val = split_frames[0], split_frames[1]
        df_test = split_frames[2] if include_test else None

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
            weight                            = DECODER_WEIGHT_COLUMN,
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
        testing = (
            TimeSeriesDataSet.from_dataset(
                training, df_test, predict=True, stop_randomization=True
            )
            if df_test is not None
            else None
        )

        train_loader = training.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = (
            testing.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            if testing is not None
            else None
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
            # pytorch-forecasting's -1e9 default overflows under FP16 autocast.
            mask_bias                  = -float("inf"),
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
        callbacks = [early_stop, checkpoint]
        if progress_reporter is not None:
            callbacks.append(ExecutionProgressCallback())
        gradient_clip_val = (
            float(trial_parameters["gradient_clip_val"])
            if trial_parameters is not None
            else self._gradient_clip_val
        )
        trainer_kwargs: dict[str, Any] = dict(
            accelerator         = "auto",
            precision           = precision,
            min_epochs          = min_epochs,
            max_epochs          = max_epochs,
            callbacks           = callbacks,
            logger              = False,
            enable_progress_bar = True,
            log_every_n_steps   = 10,
            gradient_clip_val   = gradient_clip_val,
        )
        if limits is not None:
            trainer_kwargs.update(
                limit_train_batches=limits.limit_train_batches,
                limit_val_batches=limits.limit_val_batches,
            )
            if limits.timeout_minutes is not None:
                trainer_kwargs["max_time"] = timedelta(
                    minutes=limits.timeout_minutes
                )
        trainer = pl.Trainer(**trainer_kwargs)
        report(phase="fit", epoch=0)
        fit_started = time.monotonic()
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        fit_seconds = time.monotonic() - fit_started

        best_ckpt  = Path(checkpoint.best_model_path)
        best_loss  = float(checkpoint.best_model_score)
        final_loss = float(
            trainer.callback_metrics.get("val_loss", best_loss)
        )
        epochs_run = trainer.current_epoch
        report(phase="checkpoint_reload", checkpoint_path=str(best_ckpt))
        reloaded = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
        del reloaded
        training_dataset_pkl = ckpt_dir / "training_dataset.pkl"
        with open(training_dataset_pkl, "wb") as fh:
            pickle.dump(training, fh)

        # -- Evaluate on test set (MAE on median quantile) -------------------
        train_mae = float("nan")

        metrics = {
            "train_mae":  train_mae,
            "best_val_loss": best_loss,
            "epochs_run": epochs_run,
        }
        result = {
            "version":          version,
            "execution_type": (
                limits.execution_type if limits is not None else "final"
            ),
            "best_val_loss":    best_loss,
            "final_val_loss":   final_loss,
            "epochs_run":       epochs_run,
            "checkpoint_path":  str(best_ckpt),
            "checkpoint_reload_verified": True,
            "train_mae":        train_mae,
            "environment": {
                "python": platform.python_version(),
                "torch": str(getattr(torch, "__version__", "unknown")),
                "torch_cuda_build": str(
                    getattr(getattr(torch, "version", None), "cuda", None)
                ),
                "cuda_available": cuda_available,
                "deterministic_mode": (
                    "warn_only" if self._torch_deterministic else "disabled"
                ),
                "gpu_name": (
                    str(torch.cuda.get_device_name(0))
                    if cuda_available
                    else None
                ),
            },
            "gpu_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if cuda_available
                and hasattr(torch.cuda, "max_memory_allocated")
                else 0
            ),
            "gpu_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if cuda_available
                and hasattr(torch.cuda, "max_memory_reserved")
                else 0
            ),
            "gpu_seconds": fit_seconds if cuda_available else 0.0,
        }
        if include_test:
            test_mae = self._eval_mae(trainer, tft, test_loader)
            metrics["test_mae"] = test_mae
            result["test_mae"] = test_mae
        if publish_candidate:
            if df_test is None:
                raise RuntimeError("final candidate requires a test frame")
            if self._experiment_config is None:
                raise PermissionError(
                    "final candidate requires a versioned experiment template"
                )
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
            if final_approval is None or final_snapshot is None:
                raise PermissionError(
                    "candidate lineage requires final approval and snapshot"
                )
            lineage_path = ckpt_dir / "candidate_lineage.json"
            lineage_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "inactive/review_required",
                        "version": version,
                        "study_id": final_snapshot.study_id,
                        "selected_trial_number": final_snapshot.trial_number,
                        "trial_snapshot_hash": final_snapshot.snapshot_hash(),
                        "template_hash": final_snapshot.template_hash,
                        "data_order_hash": final_snapshot.data_order_hash,
                        "parameters": dict(final_snapshot.parameters),
                        "approval": {
                            "approval_id": final_approval.approval_id,
                            "approved_by": final_approval.approved_by,
                            "approved_at": final_approval.approved_at,
                        },
                        "test_metrics": {
                            "test_mae": result.get("test_mae"),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                encoding="ascii",
            )
            extra_artifacts.append(lineage_path)
            self._registry.publish(
                version=version,
                ckpt_path=best_ckpt,
                metrics=metrics,
                training_dataset_pkl=training_dataset_pkl,
                activate=False,
                extra_artifacts=extra_artifacts,
            )
        else:
            validation_result_path = ckpt_dir / "validation_result.json"
            validation_result_path.write_text(
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                encoding="ascii",
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
                "Run the prepare and audit stages first."
            )
        return df

    def _build_split_frames(
        self,
        station_id: str,
        *,
        include_test: bool = True,
    ) -> tuple[pd.DataFrame, ...]:
        ranges = [
            ("train", self._settings.train_start, self._settings.train_end),
            ("val", self._settings.val_start, self._settings.val_end),
        ]
        if include_test:
            ranges.append(
                ("test", self._settings.test_start, self._settings.test_end)
            )
        if self._audited_frame_cache is not None:
            cached_frames: list[pd.DataFrame] = []
            for split, start, end in ranges:
                print(
                    f"[train data] validate audited {split} frame "
                    f"range={start}:{end}",
                    flush=True,
                )
                cached = self._audited_frame_cache.load(
                    split,
                    station_id,
                    start,
                    end,
                )
                if cached is None:
                    raise RuntimeError(
                        f"audited {split} frame is missing, stale, or does not "
                        f"match station={station_id} range={start}:{end}; "
                        "rerun scripts.audit_training_data"
                    )
                frame, _fingerprint = cached
                print(
                    f"[train data] reuse audited {split} frame rows={len(frame)}",
                    flush=True,
                )
                cached_frames.append(frame)
            return tuple(cached_frames)

        if self._dataset_builder is None:
            if self._data_mgr is None or self._decoder_cache is None:
                raise ValueError(
                    "TftTrainer requires a TrainingDatasetBuilder or both "
                    "data_manager and decoder_cache. Legacy split-column "
                    "training data is no longer supported."
                )
            from training.dataset import TrainingDatasetBuilder
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
        frames = [train, val]
        if include_test:
            frames.append(
                self._dataset_builder.build_frame(
                    station,
                    self._init_times(
                        self._settings.test_start,
                        self._settings.test_end,
                    ),
                    split="test",
                )
            )
        if any(frame.empty for frame in frames):
            raise ValueError(
                "TrainingDatasetBuilder produced an empty split: "
                + " ".join(
                    f"{name}={len(frame)}"
                    for (name, _start, _end), frame in zip(ranges, frames)
                )
            )
        return tuple(frames)

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
            if isinstance(prediction.y, tuple):
                target = prediction.y[0]
                weight = prediction.y[1] if len(prediction.y) > 1 else None
            else:
                target = prediction.y
                weight = None
            if target is None:
                raise ValueError("prediction target is missing")
            absolute_error = (all_pred - target).abs()
            if weight is None:
                mae = float(absolute_error.mean())
            else:
                trusted_weight = weight.to(absolute_error)
                weight_sum = trusted_weight.sum()
                if float(weight_sum) <= 0:
                    raise ValueError("prediction target weight has no trusted cells")
                mae = float((absolute_error * trusted_weight).sum() / weight_sum)
            return round(mae, 5)
        except Exception as exc:
            logger.warning("Could not compute test MAE: %s", exc)
            return float("nan")
