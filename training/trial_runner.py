"""Managed train/validation execution runner with durable evidence."""
from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from training.catalog import ExecutionRecord, TrainingCatalog
from training.config import ExperimentTemplate
from training.experiment import ValidationExecutionLimits
from training.trainer import _train_validation
from training.workspace import TrainingWorkspace


ValidationBackend = Callable[..., dict[str, Any]]


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
