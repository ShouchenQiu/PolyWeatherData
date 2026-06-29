"""Managed train/validation execution runner with durable evidence."""
from __future__ import annotations

import contextlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from training.catalog import ExecutionRecord, TrainingCatalog
from training.config import ExperimentTemplate
from training.experiment import ValidationExecutionLimits
from training.package import package_candidate_model
from training.trainer import _train_final, _train_validation
from training.workspace import TrainingWorkspace


ValidationBackend = Callable[..., dict[str, Any]]
FinalBackend = Callable[..., dict[str, Any]]


class _ExecutionCancellation(RuntimeError):
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


def run_training_execution(
    catalog: TrainingCatalog,
    execution_id: str,
    *,
    validation_backend: ValidationBackend = _train_validation,
) -> ExecutionRecord:
    """Run one queued smoke/calibration and persist its observable evidence."""
    queued = catalog.get_execution(execution_id)
    study = catalog.validate_study(queued.study_id)
    workspace = TrainingWorkspace(study.workspace_path)
    template_record = catalog.get_template(study.template_hash)
    template = ExperimentTemplate.from_dict(
        json.loads(template_record.config_json)
    )
    limits = ValidationExecutionLimits.from_catalog_limits(
        queued.execution_type,
        queued.limits,
        template,
    )
    execution_dir = (
        workspace.run_dir
        / "studies"
        / study.study_id
        / queued.execution_type
        / f"attempt-{queued.attempt}"
    )
    execution_dir.mkdir(parents=True, exist_ok=True)
    log_path = execution_dir / "execution.log"
    result_path = execution_dir / "validation_result.json"
    current = catalog.get_execution(execution_id)
    if current.status == "queued":
        catalog.start_execution(execution_id, log_path=log_path)
    elif current.status != "running":
        raise PermissionError(
            "training execution must be queued or transactionally claimed"
        )

    def report(progress: dict[str, object]) -> None:
        if catalog.get_execution(execution_id).status == "cancel_requested":
            raise _ExecutionCancellation("operator requested cancellation")
        catalog.record_execution_progress(execution_id, progress)
        print(
            "[execution progress] "
            + json.dumps(progress, sort_keys=True, default=str),
            flush=True,
        )

    try:
        with log_path.open("a", encoding="utf-8") as log:
            with (
                contextlib.redirect_stdout(_Tee(sys.stdout, log)),
                contextlib.redirect_stderr(_Tee(sys.stderr, log)),
            ):
                result = validation_backend(
                    workspace=workspace,
                    station_id=study.station_id,
                    version=execution_id,
                    experiment_config=template_record.path,
                    limits=limits,
                    progress_reporter=report,
                    output_dir=execution_dir / "checkpoints",
                )
                if (
                    catalog.get_execution(execution_id).status
                    == "cancel_requested"
                ):
                    raise _ExecutionCancellation(
                        "operator requested cancellation"
                    )
                result_path.write_text(
                    json.dumps(
                        result,
                        indent=2,
                        sort_keys=True,
                        default=str,
                    ),
                    encoding="utf-8",
                )
        completed = catalog.finish_execution(execution_id, metrics=result)
    except _ExecutionCancellation as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nexecution cancelled: {exc}\n")
        cancelled = catalog.complete_cancel(execution_id)
        catalog.register_artifact(
            study.study_id,
            execution_id=execution_id,
            artifact_type="execution_log",
            path=log_path,
        )
        return cancelled
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        catalog.fail_execution(
            execution_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        catalog.register_artifact(
            study.study_id,
            execution_id=execution_id,
            artifact_type="execution_log",
            path=log_path,
        )
        raise

    catalog.register_artifact(
        study.study_id,
        execution_id=execution_id,
        artifact_type="execution_log",
        path=log_path,
    )
    catalog.register_artifact(
        study.study_id,
        execution_id=execution_id,
        artifact_type="validation_result",
        path=result_path,
    )
    return completed


def run_final_execution(
    catalog: TrainingCatalog,
    study_id: str,
    *,
    final_backend: FinalBackend = _train_final,
) -> ExecutionRecord:
    """Consume the one-shot approval and publish an inactive candidate."""
    study = catalog.validate_study(study_id)
    workspace = TrainingWorkspace(study.workspace_path)
    template = catalog.get_template(study.template_hash)
    approval = catalog.consume_final_approval(study_id)
    snapshot = catalog.get_final_trial_snapshot(study_id)
    execution_id = f"{study_id}-final-a1"
    execution = catalog.get_execution(execution_id)
    execution_dir = Path(execution.log_path).parent
    execution_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(execution.log_path)
    result_path = execution_dir / "train_result.json"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            with (
                contextlib.redirect_stdout(_Tee(sys.stdout, log)),
                contextlib.redirect_stderr(_Tee(sys.stderr, log)),
            ):
                result = final_backend(
                    workspace=workspace,
                    station_id=study.station_id,
                    version=study.candidate_version,
                    experiment_config=template.path,
                    approval=approval,
                    trial_snapshot=snapshot,
                    output_dir=execution_dir / "checkpoints",
                )
                package = package_candidate_model(
                    workspace.run_dir,
                    study.candidate_version,
                )
                package_path = execution_dir / Path(
                    str(package["package_path"])
                ).name
                shutil.copy2(package["package_path"], package_path)
                lineage_source = (
                    workspace.model_store_dir
                    / study.candidate_version
                    / "candidate_lineage.json"
                )
                lineage_path = execution_dir / "candidate_lineage.json"
                shutil.copy2(lineage_source, lineage_path)
                result["candidate_package_path"] = str(package_path)
                result["candidate_package_sha256"] = package["sha256"]
                result_path.write_text(
                    json.dumps(
                        result,
                        indent=2,
                        sort_keys=True,
                        default=str,
                    ),
                    encoding="utf-8",
                )
        completed = catalog.finish_execution(
            execution_id,
            metrics=result,
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        catalog.fail_execution(
            execution_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        catalog.register_artifact(
            study_id,
            execution_id=execution_id,
            artifact_type="execution_log",
            path=log_path,
        )
        raise
    for artifact_type, path in (
        ("execution_log", log_path),
        ("final_result", result_path),
        ("candidate_lineage", lineage_path),
        ("candidate_package", package_path),
    ):
        catalog.register_artifact(
            study_id,
            execution_id=execution_id,
            artifact_type=artifact_type,
            path=path,
        )
    return completed
