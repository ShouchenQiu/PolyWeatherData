"""Optuna adapter for one catalog-managed training study."""
from __future__ import annotations

import contextlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import optuna

from training.catalog import TrainingCatalog
from training.config import ExperimentTemplate, ResolvedTrialSnapshot
from training.experiment import ValidationExecutionLimits
from training.trainer import _train_validation
from training.workspace import TrainingWorkspace


@dataclass(frozen=True)
class TrialEvaluation:
    best_val_loss: float
    final_val_loss: float
    epochs_run: int
    gpu_seconds: float
    gpu_peak_memory_allocated_bytes: int
    checkpoint_path: str | None
    environment: dict[str, object] | None = None


@dataclass
class TrialContext:
    trial: optuna.Trial
    parameters: dict[str, object]
    snapshot: ResolvedTrialSnapshot
    execution_id: str
    workspace_path: Path
    template_path: Path
    output_dir: Path
    epoch_reporter: Callable[[int, float], None]

    def report_epoch(self, epoch: int, validation_loss: float) -> None:
        if epoch < 0 or not math.isfinite(validation_loss):
            raise ValueError("epoch report must contain a finite validation loss")
        self.epoch_reporter(epoch, float(validation_loss))
        self.trial.report(float(validation_loss), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"median pruner stopped trial at epoch {epoch}"
            )


@dataclass(frozen=True)
class SearchSummary:
    study_id: str
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    gpu_seconds: float
    status: str


TrialEvaluator = Callable[[TrialContext], TrialEvaluation]
TrainingBackend = Callable[..., dict[str, Any]]


class _NonFiniteTrialMetric(ValueError):
    pass


class _SearchCancellation(RuntimeError):
    pass


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class OptunaEngine:
    """Prepare and resume a versioned Optuna study after training gates."""

    def __init__(
        self,
        catalog: TrainingCatalog,
        *,
        storage_url: str,
        training_backend: TrainingBackend = _train_validation,
    ) -> None:
        self._catalog = catalog
        self.storage_url = storage_url
        self._training_backend = training_backend

    def prepare(self, study_id: str) -> optuna.Study:
        study = self._catalog.validate_study(study_id)
        if study.status not in {
            "queued",
            "running",
            "completed",
            "budget_exhausted",
        }:
            raise PermissionError(
                "Optuna preparation requires approved queued/running study"
            )
        self._catalog.get_study_approval(study_id, "search")
        template_record = self._catalog.get_template(study.template_hash)
        template = ExperimentTemplate.from_dict(
            json.loads(template_record.config_json)
        )
        calibration = self._successful_calibration(study_id)
        calibration_gpu_seconds = float(
            calibration.metrics.get("gpu_seconds", 0.0)
        )
        if calibration_gpu_seconds <= 0:
            raise ValueError(
                "successful calibration must record positive gpu_seconds"
            )
        estimated_trial_gpu_seconds = (
            calibration_gpu_seconds * template.trainer.max_epochs
        )
        snapshot_path = (
            Path(study.workspace_path)
            / "studies"
            / study_id
            / "optuna_study_snapshot.json"
        )
        snapshot = {
            "schema_version": 1,
            "study_id": study_id,
            "template_hash": study.template_hash,
            "data_order_hash": self._catalog.get_workspace(
                study.workspace_path
            ).data_order_hash,
            "study_spec": template.study.to_dict(),
            "max_epochs": template.trainer.max_epochs,
            "calibration_execution_id": calibration.execution_id,
            "calibration_best_val_loss": calibration.metrics.get(
                "best_val_loss"
            ),
            "calibration_gpu_seconds": calibration_gpu_seconds,
            "estimated_trial_gpu_seconds": estimated_trial_gpu_seconds,
            "estimated_trials_within_budget": (
                template.study.budget_gpu_hours
                * 3600.0
                / estimated_trial_gpu_seconds
            ),
        }
        created_snapshot = False
        if snapshot_path.exists():
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if existing != snapshot:
                raise RuntimeError(
                    "frozen Optuna study snapshot does not match catalog"
                )
        else:
            snapshot_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            created_snapshot = True

        optuna_study = optuna.create_study(
            study_name=study.optuna_study_name,
            storage=self.storage_url,
            direction=template.study.direction,
            sampler=optuna.samplers.TPESampler(
                seed=template.study.sampler.seed
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=template.study.pruner.startup_trials,
                n_warmup_steps=template.study.pruner.warmup_epochs,
            ),
            load_if_exists=True,
        )
        if created_snapshot:
            self._catalog.register_artifact(
                study_id,
                artifact_type="optuna_study_snapshot",
                path=snapshot_path,
            )
        return optuna_study

    def suggest_parameters(
        self,
        study_id: str,
        trial: optuna.Trial,
    ) -> dict[str, object]:
        """Resolve exactly the versioned template's search-space parameters."""
        study = self._catalog.validate_study(study_id)
        template = ExperimentTemplate.from_dict(
            json.loads(
                self._catalog.get_template(study.template_hash).config_json
            )
        )
        parameters: dict[str, object] = {}
        for name, spec in sorted(template.study.search_space.items()):
            if spec.kind == "categorical":
                parameters[name] = trial.suggest_categorical(
                    name,
                    list(spec.choices),
                )
            elif spec.kind == "float":
                parameters[name] = trial.suggest_float(
                    name,
                    float(spec.minimum),
                    float(spec.maximum),
                    log=spec.log,
                )
            elif spec.kind == "int":
                parameters[name] = trial.suggest_int(
                    name,
                    int(spec.minimum),
                    int(spec.maximum),
                    log=spec.log,
                )
            else:
                raise ValueError(
                    f"unsupported search parameter kind: {spec.kind}"
                )
        return parameters

    def run(
        self,
        study_id: str,
        *,
        evaluator: TrialEvaluator | None = None,
        max_trials: int | None = None,
        budget_gpu_seconds: float | None = None,
        finalize_at_trial_cap: bool = True,
    ) -> SearchSummary:
        """Run serial trials until a local trial cap or cumulative GPU budget."""
        if max_trials is not None and max_trials <= 0:
            raise ValueError("max_trials must be positive")
        optuna_study = self.prepare(study_id)
        self._catalog.start_search(study_id)
        study_record = self._catalog.get_study(study_id)
        budget_limit = (
            float(budget_gpu_seconds)
            if budget_gpu_seconds is not None
            else study_record.budget_gpu_hours * 3600.0
        )
        if not math.isfinite(budget_limit) or budget_limit <= 0:
            raise ValueError("GPU budget must be positive and finite")
        workspace = self._catalog.get_workspace(study_record.workspace_path)
        selected_evaluator = evaluator or self._evaluate_tft_trial
        self._reconcile_interrupted_trials(study_id, optuna_study)

        def objective(trial: optuna.Trial) -> float:
            parameters = self.suggest_parameters(study_id, trial)
            snapshot = ResolvedTrialSnapshot.from_dict(
                {
                    "schema_version": 1,
                    "study_id": study_id,
                    "trial_number": trial.number,
                    "execution_type": "trial",
                    "template_hash": study_record.template_hash,
                    "data_order_hash": workspace.data_order_hash,
                    "parameters": parameters,
                }
            )
            trial_dir = (
                Path(study_record.workspace_path)
                / "studies"
                / study_id
                / "trials"
                / f"trial-{trial.number:05d}"
                / "attempt-1"
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            execution = self._catalog.start_trial_execution(
                study_id,
                trial_number=trial.number,
                limits=parameters,
                log_path=trial_dir / "execution.log",
            )
            snapshot_path = trial_dir / "trial_snapshot.json"
            snapshot_path.write_text(
                json.dumps(
                    snapshot.to_dict(),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._catalog.register_artifact(
                study_id,
                execution_id=execution.execution_id,
                artifact_type="trial_snapshot",
                path=snapshot_path,
            )
            started = time.monotonic()

            def report_epoch(epoch: int, loss: float) -> None:
                current = self._catalog.get_execution(
                    execution.execution_id
                )
                if current.status == "cancel_requested":
                    raise _SearchCancellation(
                        "operator requested Optuna cancellation"
                    )
                self._catalog.record_execution_progress(
                    execution.execution_id,
                    {
                        "phase": "validation_epoch_end",
                        "epoch": epoch,
                        "validation_quantile_loss": loss,
                    },
                )

            context = TrialContext(
                trial=trial,
                parameters=parameters,
                snapshot=snapshot,
                execution_id=execution.execution_id,
                workspace_path=Path(study_record.workspace_path),
                template_path=Path(
                    self._catalog.get_template(
                        study_record.template_hash
                    ).path
                ),
                output_dir=trial_dir,
                epoch_reporter=report_epoch,
            )
            try:
                result = selected_evaluator(context)
                self._validate_evaluation(result)
            except _SearchCancellation as exc:
                gpu_seconds = time.monotonic() - started
                self._record_trial_error(
                    study_id,
                    execution.execution_id,
                    trial_dir,
                    exc,
                )
                trial.set_user_attr("gpu_seconds", gpu_seconds)
                trial.set_user_attr("failure_class", "cancelled")
                self._catalog.complete_cancel(execution.execution_id)
                raise
            except optuna.TrialPruned as exc:
                gpu_seconds = time.monotonic() - started
                self._record_trial_error(
                    study_id,
                    execution.execution_id,
                    trial_dir,
                    exc,
                )
                trial.set_user_attr("gpu_seconds", gpu_seconds)
                self._catalog.finish_trial_execution(
                    execution.execution_id,
                    status="pruned",
                    metrics={"gpu_seconds": gpu_seconds},
                    error=str(exc),
                )
                raise
            except Exception as exc:
                gpu_seconds = time.monotonic() - started
                self._record_trial_error(
                    study_id,
                    execution.execution_id,
                    trial_dir,
                    exc,
                )
                failure_class = self._classify_failure(exc)
                trial.set_user_attr("gpu_seconds", gpu_seconds)
                trial.set_user_attr("failure_class", failure_class)
                self._catalog.finish_trial_execution(
                    execution.execution_id,
                    status="failed",
                    metrics={
                        "gpu_seconds": gpu_seconds,
                        "failure_class": failure_class,
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            metrics = {
                "best_val_loss": result.best_val_loss,
                "final_val_loss": result.final_val_loss,
                "epochs_run": result.epochs_run,
                "gpu_seconds": result.gpu_seconds,
                "gpu_peak_memory_allocated_bytes": (
                    result.gpu_peak_memory_allocated_bytes
                ),
                "checkpoint_path": result.checkpoint_path,
                "environment": result.environment,
                "trial_snapshot_hash": snapshot.snapshot_hash(),
            }
            trial.set_user_attr("gpu_seconds", result.gpu_seconds)
            trial.set_user_attr(
                "trial_snapshot_hash",
                snapshot.snapshot_hash(),
            )
            self._catalog.finish_trial_execution(
                execution.execution_id,
                status="succeeded",
                metrics=metrics,
            )
            return result.best_val_loss

        def stop_at_budget(
            current_study: optuna.Study,
            _trial: optuna.trial.FrozenTrial,
        ) -> None:
            if self._catalog.get_study(study_id).status == "cancelled":
                current_study.stop()
                return
            if self._gpu_seconds(study_id) >= budget_limit:
                current_study.stop()

        if self._gpu_seconds(study_id) < budget_limit:
            optuna_study.optimize(
                objective,
                n_trials=max_trials,
                n_jobs=1,
                catch=(Exception,),
                callbacks=[stop_at_budget],
            )
        current_status = self._catalog.get_study(study_id).status
        if current_status == "cancelled":
            status = "cancelled"
        elif self._gpu_seconds(study_id) >= budget_limit:
            status = "budget_exhausted"
        elif max_trials is not None and finalize_at_trial_cap:
            status = "completed"
        else:
            status = "running"
        if status not in {"running", "cancelled"}:
            self._catalog.finish_search(study_id, status=status)
        return self._summary(study_id, optuna_study, status=status)

    def _reconcile_interrupted_trials(
        self,
        study_id: str,
        optuna_study: optuna.Study,
    ) -> None:
        interrupted_numbers = {
            execution.trial_number
            for execution in self._catalog.list_executions(study_id)
            if execution.execution_type == "trial"
            and execution.status == "interrupted"
        }
        for trial in optuna_study.trials:
            if (
                trial.state == optuna.trial.TrialState.RUNNING
                and trial.number in interrupted_numbers
            ):
                optuna_study.tell(
                    trial.number,
                    state=optuna.trial.TrialState.FAIL,
                )

    def _evaluate_tft_trial(
        self,
        context: TrialContext,
    ) -> TrialEvaluation:
        template = ExperimentTemplate.from_json(context.template_path)
        limits = ValidationExecutionLimits.for_trial(
            template,
            context.parameters,
        )
        log_path = context.output_dir / "execution.log"
        result_path = context.output_dir / "validation_result.json"

        def report(progress: dict[str, object]) -> None:
            if "validation_quantile_loss" in progress:
                context.report_epoch(
                    int(progress["epoch"]),
                    float(progress["validation_quantile_loss"]),
                )
            else:
                self._catalog.record_execution_progress(
                    context.execution_id,
                    progress,
                )

        with log_path.open("a", encoding="utf-8") as log:
            with (
                contextlib.redirect_stdout(_Tee(sys.stdout, log)),
                contextlib.redirect_stderr(_Tee(sys.stderr, log)),
            ):
                result = self._training_backend(
                    workspace=TrainingWorkspace(context.workspace_path),
                    station_id=template.station,
                    version=context.execution_id,
                    experiment_config=context.template_path,
                    limits=limits,
                    progress_reporter=report,
                    output_dir=context.output_dir / "checkpoints",
                    trial_parameters=context.parameters,
                )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self._catalog.register_artifact(
            context.snapshot.study_id,
            execution_id=context.execution_id,
            artifact_type="execution_log",
            path=log_path,
        )
        self._catalog.register_artifact(
            context.snapshot.study_id,
            execution_id=context.execution_id,
            artifact_type="validation_result",
            path=result_path,
        )
        checkpoint_path = Path(str(result["checkpoint_path"]))
        if checkpoint_path.is_file():
            self._catalog.register_artifact(
                context.snapshot.study_id,
                execution_id=context.execution_id,
                artifact_type="checkpoint",
                path=checkpoint_path,
            )
        dataset_path = checkpoint_path.parent / "training_dataset.pkl"
        if dataset_path.is_file():
            self._catalog.register_artifact(
                context.snapshot.study_id,
                execution_id=context.execution_id,
                artifact_type="training_dataset",
                path=dataset_path,
            )
        return TrialEvaluation(
            best_val_loss=float(result["best_val_loss"]),
            final_val_loss=float(
                result.get("final_val_loss", result["best_val_loss"])
            ),
            epochs_run=int(result["epochs_run"]),
            gpu_seconds=float(result["gpu_seconds"]),
            gpu_peak_memory_allocated_bytes=int(
                result["gpu_peak_memory_allocated_bytes"]
            ),
            checkpoint_path=str(result["checkpoint_path"]),
            environment=dict(result.get("environment", {})),
        )

    def _record_trial_error(
        self,
        study_id: str,
        execution_id: str,
        trial_dir: Path,
        exc: Exception,
    ) -> None:
        log_path = trial_dir / "execution.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n{type(exc).__name__}: {exc}\n")
            traceback.print_exception(exc, file=log)
        if not any(
            artifact.path == str(log_path.resolve())
            for artifact in self._catalog.list_artifacts(study_id)
        ):
            self._catalog.register_artifact(
                study_id,
                execution_id=execution_id,
                artifact_type="execution_log",
                path=log_path,
            )

    @staticmethod
    def _validate_evaluation(result: TrialEvaluation) -> None:
        if not math.isfinite(result.best_val_loss) or not math.isfinite(
            result.final_val_loss
        ):
            raise _NonFiniteTrialMetric(
                "trial evaluation contains NaN/non-finite validation loss"
            )
        if (
            result.epochs_run <= 0
            or not math.isfinite(result.gpu_seconds)
            or result.gpu_seconds < 0
            or result.gpu_peak_memory_allocated_bytes < 0
        ):
            raise ValueError("trial evaluation contains invalid metrics")

    @staticmethod
    def _classify_failure(exc: Exception) -> str:
        if isinstance(exc, _NonFiniteTrialMetric):
            return "nan"
        if "out of memory" in str(exc).lower():
            return "oom"
        return "exception"

    def _summary(
        self,
        study_id: str,
        study: optuna.Study,
        *,
        status: str,
    ) -> SearchSummary:
        states = [trial.state for trial in study.trials]
        return SearchSummary(
            study_id=study_id,
            completed_trials=states.count(optuna.trial.TrialState.COMPLETE),
            pruned_trials=states.count(optuna.trial.TrialState.PRUNED),
            failed_trials=states.count(optuna.trial.TrialState.FAIL),
            gpu_seconds=self._gpu_seconds(study_id),
            status=status,
        )

    def _gpu_seconds(self, study_id: str) -> float:
        return sum(
            float(execution.metrics.get("gpu_seconds", 0.0))
            for execution in self._catalog.list_executions(study_id)
            if execution.execution_type == "trial"
        )

    def _successful_calibration(self, study_id: str):
        matches = [
            execution
            for execution in self._catalog.list_executions(study_id)
            if execution.execution_type == "calibration"
            and execution.status == "succeeded"
        ]
        if len(matches) != 1:
            raise PermissionError(
                "Optuna study requires exactly one successful calibration"
            )
        return matches[0]
