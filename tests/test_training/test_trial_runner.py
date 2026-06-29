from __future__ import annotations

from pathlib import Path
import json

from training.catalog import TrainingCatalog
from training.config import ExperimentTemplate, ResolvedTrialSnapshot
from training.experiment import ValidationExecutionLimits
from training.trial_runner import run_final_execution, run_training_execution
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


def test_final_runner_consumes_once_and_packages_inactive_candidate(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    for status in (
        "smoke_queued", "smoke_running", "smoke_passed",
        "calibration_queued", "calibration_running", "gated",
        "queued", "running", "completed", "awaiting_selection",
    ):
        catalog.transition_study(study.study_id, status)
    template = ExperimentTemplate.from_json(
        "experiments/klga/tft_optuna_v1.json"
    )
    parameters = {
        name: spec.choices[0]
        if spec.kind == "categorical"
        else spec.minimum
        for name, spec in template.study.search_space.items()
    }
    snapshot = ResolvedTrialSnapshot.from_dict(
        {
            "schema_version": 1,
            "study_id": study.study_id,
            "trial_number": 3,
            "execution_type": "trial",
            "template_hash": study.template_hash,
            "data_order_hash": "data-order-hash",
            "parameters": parameters,
        }
    )
    evidence_path = (
        workspace.run_dir
        / "studies"
        / study.study_id
        / "top_k_validation_evidence.json"
    )
    workspace.write_json(
        evidence_path,
        {
            "test_accessed": False,
            "trials": [
                {
                    "trial_number": snapshot.trial_number,
                    "trial_snapshot_hash": snapshot.snapshot_hash(),
                }
            ],
        },
    )
    catalog.register_artifact(
        study.study_id,
        artifact_type="top_k_validation_evidence",
        path=evidence_path,
    )
    catalog.approve_final(study.study_id, snapshot, approved_by="pytest")

    def final_backend(*, workspace, version, trial_snapshot, **_kwargs):
        version_dir = workspace.model_store_dir / version
        version_dir.mkdir(parents=True)
        for name in (
            "model.ckpt",
            "training_dataset.pkl",
            "training_metadata.json",
            "experiment_config.json",
            "model_contract.json",
        ):
            (version_dir / name).write_text("{}", encoding="ascii")
        (version_dir / "meta.json").write_text(
            json.dumps({"metrics": {"test_mae": 1.2}}),
            encoding="ascii",
        )
        (version_dir / "candidate_lineage.json").write_text(
            json.dumps(
                {
                    "state": "inactive/review_required",
                    "study_id": trial_snapshot.study_id,
                }
            ),
            encoding="ascii",
        )
        return {"test_mae": 1.2}

    completed = run_final_execution(
        catalog,
        study.study_id,
        final_backend=final_backend,
    )

    assert completed.status == "succeeded"
    assert catalog.get_study(study.study_id).status == "candidate_ready"
    assert {
        item.artifact_type
        for item in catalog.list_artifacts(study.study_id)
    } == {
        "candidate_lineage",
        "candidate_package",
        "execution_log",
        "final_result",
        "top_k_validation_evidence",
    }
    assert not (workspace.model_store_dir / "active_version.txt").exists()
