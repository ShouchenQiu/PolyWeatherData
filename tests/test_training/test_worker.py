from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from training.catalog import TrainingCatalog
from training.worker import GpuLock, TrainingWorker
from training.workspace import TrainingWorkspace


def _queued_smoke(tmp_path: Path) -> tuple[Path, str]:
    workspace = TrainingWorkspace(tmp_path / "runs" / "ready")
    workspace.ensure()
    workspace.write_json(
        workspace.audit_summary_path,
        {
            "ready": True,
            "station_id": "KLGA",
            "data_order_hash": "data-order-hash",
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
    catalog_path = tmp_path / "training_catalog.db"
    with TrainingCatalog(catalog_path) as catalog:
        study = catalog.create_study(
            workspace.run_dir,
            "experiments/klga/tft_optuna_v1.json",
        )
        execution = catalog.enqueue_smoke(study.study_id)
    return catalog_path, execution.execution_id


def test_only_one_worker_can_transactionally_claim_queued_execution(
    tmp_path: Path,
) -> None:
    catalog_path, execution_id = _queued_smoke(tmp_path)

    def claim(worker_id: str):
        with TrainingCatalog(catalog_path) as catalog:
            return catalog.claim_next_execution(worker_id=worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(claim, ("worker-a", "worker-b")))

    winners = [record for record in claimed if record is not None]
    assert len(winners) == 1
    assert winners[0].execution_id == execution_id
    assert winners[0].status == "running"
    assert winners[0].worker_id in {"worker-a", "worker-b"}
    with TrainingCatalog(catalog_path) as catalog:
        assert catalog.get_study(winners[0].study_id).status == "smoke_running"


def test_worker_restart_marks_abandoned_running_execution_interrupted(
    tmp_path: Path,
) -> None:
    catalog_path, execution_id = _queued_smoke(tmp_path)
    with TrainingCatalog(catalog_path) as catalog:
        catalog.claim_next_execution(worker_id="dead-worker")

    with TrainingCatalog(catalog_path) as restarted:
        recovered = restarted.recover_interrupted_executions(
            worker_id="replacement-worker"
        )
        execution = restarted.get_execution(execution_id)
        study_status = restarted.get_study(execution.study_id).status

    assert [item.execution_id for item in recovered] == [execution_id]
    assert execution.status == "interrupted"
    assert "dead-worker" in execution.error
    assert study_status == "interrupted"


def test_retry_preserves_interrupted_attempt_and_queues_new_attempt(
    tmp_path: Path,
) -> None:
    catalog_path, execution_id = _queued_smoke(tmp_path)
    with TrainingCatalog(catalog_path) as catalog:
        catalog.claim_next_execution(worker_id="dead-worker")
        catalog.recover_interrupted_executions(worker_id="replacement")

        retried = catalog.retry_execution(execution_id)
        attempts = catalog.list_executions(retried.study_id)

    assert retried.attempt == 2
    assert retried.status == "queued"
    assert retried.execution_id.endswith("-smoke-a2")
    assert [(item.attempt, item.status) for item in attempts] == [
        (1, "interrupted"),
        (2, "queued"),
    ]


def test_queued_cancel_is_terminal_and_idempotent(
    tmp_path: Path,
) -> None:
    catalog_path, execution_id = _queued_smoke(tmp_path)
    with TrainingCatalog(catalog_path) as catalog:
        first = catalog.request_cancel(execution_id)
        second = catalog.request_cancel(execution_id)
        study_status = catalog.get_study(first.study_id).status

    assert first.status == "cancelled"
    assert second == first
    assert first.finished_at is not None
    assert study_status == "cancelled"


def test_gpu_lock_rejects_second_worker(tmp_path: Path) -> None:
    lock_path = tmp_path / "training.gpu.lock"

    with GpuLock(lock_path):
        with pytest.raises(RuntimeError, match="GPU is already owned"):
            with GpuLock(lock_path):
                pass


def test_restarted_worker_runs_previously_queued_execution(
    tmp_path: Path,
) -> None:
    catalog_path, execution_id = _queued_smoke(tmp_path)

    def execute(catalog: TrainingCatalog, claimed_id: str):
        assert claimed_id == execution_id
        assert catalog.get_execution(claimed_id).status == "running"
        return catalog.finish_execution(
            claimed_id,
            metrics={"best_val_loss": 0.5},
        )

    worker = TrainingWorker(
        catalog_path,
        lock_path=tmp_path / "gpu.lock",
        worker_id="worker-restarted",
        execution_runner=execute,
    )

    completed = worker.run_once()

    assert completed is not None
    assert completed.status == "succeeded"
    with TrainingCatalog(catalog_path) as catalog:
        assert catalog.get_study(completed.study_id).status == "smoke_passed"


def test_worker_restart_finishes_abandoned_cancel_request(
    tmp_path: Path,
) -> None:
    catalog_path, execution_id = _queued_smoke(tmp_path)
    with TrainingCatalog(catalog_path) as catalog:
        catalog.claim_next_execution(worker_id="dead-worker")
        catalog.request_cancel(execution_id)

    with TrainingCatalog(catalog_path) as restarted:
        restarted.recover_interrupted_executions(worker_id="replacement")
        execution = restarted.get_execution(execution_id)
        study_status = restarted.get_study(execution.study_id).status

    assert execution.status == "cancelled"
    assert study_status == "cancelled"


def test_worker_claims_approved_search_when_execution_queue_is_empty(
    tmp_path: Path,
) -> None:
    catalog_path, smoke_id = _queued_smoke(tmp_path)
    with TrainingCatalog(catalog_path) as catalog:
        smoke = catalog.get_execution(smoke_id)
        catalog.start_execution(
            smoke_id,
            log_path=tmp_path / "smoke.log",
        )
        catalog.finish_execution(smoke_id, metrics={"gpu_seconds": 1.0})
        calibration = catalog.enqueue_calibration(smoke.study_id)
        catalog.start_execution(
            calibration.execution_id,
            log_path=tmp_path / "calibration.log",
        )
        catalog.finish_execution(
            calibration.execution_id,
            metrics={"gpu_seconds": 2.0},
        )
        catalog.approve_search(smoke.study_id, approved_by="pytest")

    claimed_studies = []

    def run_search(catalog: TrainingCatalog, study_id: str):
        claimed_studies.append(study_id)
        return catalog.finish_search(study_id, status="completed")

    worker = TrainingWorker(
        catalog_path,
        lock_path=tmp_path / "gpu.lock",
        worker_id="search-worker",
        search_runner=run_search,
    )

    completed = worker.run_once()

    assert completed is not None
    assert completed.status == "completed"
    assert claimed_studies == [smoke.study_id]
