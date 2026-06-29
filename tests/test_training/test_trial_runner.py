from __future__ import annotations

from pathlib import Path

from training.catalog import TrainingCatalog
from training.experiment import ValidationExecutionLimits
from training.trial_runner import run_training_execution
from training.workspace import TrainingWorkspace


def _ready_workspace(tmp_path: Path) -> TrainingWorkspace:
    workspace = TrainingWorkspace(tmp_path / "runs" / "klga-ready")
    workspace.ensure()
    workspace.write_json(
        workspace.audit_summary_path,
        {
            "ready": True,
            "station_id": "KLGA",
            "data_order_hash": "data-order-hash",
            "failures": [],
        },
    )
    for split, (start, end) in {
        "train": ("2024-10-07", "2026-03-31"),
        "val": ("2026-04-01", "2026-04-30"),
        "test": ("2026-05-01", "2026-05-31"),
    }.items():
        workspace.write_json(
            workspace.audit_frames_dir / f"{split}.manifest.json",
            {
                "schema_version": 1,
                "split": split,
                "station_id": "KLGA",
                "start_date": start,
                "end_date": end,
                "source_fingerprint": f"{split}-fingerprint",
                "row_count": 120,
            },
        )
    return workspace


def test_runner_persists_live_smoke_evidence_and_artifacts(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    execution = catalog.enqueue_smoke(study.study_id)

    def validation_backend(
        *,
        limits: ValidationExecutionLimits,
        progress_reporter,
        output_dir: Path,
        **_kwargs,
    ):
        assert limits.execution_type == "smoke"
        assert limits.limit_train_batches == 20
        progress_reporter(
            {
                "phase": "fit",
                "epoch": 0,
                "train_batches_completed": 20,
                "gpu_memory_allocated_bytes": 1024,
            }
        )
        print("backend smoke complete")
        return {
            "best_val_loss": 0.25,
            "checkpoint_path": str(output_dir / "best.ckpt"),
            "checkpoint_reload_verified": True,
            "gpu_peak_memory_allocated_bytes": 2048,
            "gpu_seconds": 10.0,
            "environment": {"cuda_available": True},
        }

    completed = run_training_execution(
        catalog,
        execution.execution_id,
        validation_backend=validation_backend,
    )

    assert completed.status == "succeeded"
    assert completed.progress["train_batches_completed"] == 20
    assert catalog.get_study(study.study_id).status == "smoke_passed"
    artifact_types = {
        artifact.artifact_type
        for artifact in catalog.list_artifacts(study.study_id)
    }
    assert artifact_types == {"execution_log", "validation_result"}
    assert "backend smoke complete" in Path(completed.log_path).read_text()


def test_running_execution_honours_catalog_cancel_request(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog_path = tmp_path / "training_catalog.db"
    catalog = TrainingCatalog(catalog_path)
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    execution = catalog.enqueue_smoke(study.study_id)

    def validation_backend(*, progress_reporter, **_kwargs):
        with TrainingCatalog(catalog_path) as operator:
            operator.request_cancel(execution.execution_id)
        progress_reporter(
            {
                "phase": "fit",
                "train_batches_completed": 10,
                "train_batches_total": 20,
            }
        )
        raise AssertionError("cancelled execution must stop the backend")

    cancelled = run_training_execution(
        catalog,
        execution.execution_id,
        validation_backend=validation_backend,
    )

    assert cancelled.status == "cancelled"
    assert catalog.get_study(study.study_id).status == "cancelled"
