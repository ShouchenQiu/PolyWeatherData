from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from training.catalog import TrainingCatalog
from training.config import ResolvedTrialSnapshot
from training.workspace import TrainingWorkspace


def _ready_workspace(tmp_path: Path) -> TrainingWorkspace:
    workspace = TrainingWorkspace(tmp_path / "runs" / "klga-ready")
    workspace.ensure()
    splits = {
        "train": ("2024-10-07", "2026-03-31"),
        "val": ("2026-04-01", "2026-04-30"),
        "test": ("2026-05-01", "2026-05-31"),
    }
    workspace.write_json(
        workspace.audit_summary_path,
        {
            "ready": True,
            "station_id": "KLGA",
            "data_order_hash": "data-order-hash",
            "failures": [],
        },
    )
    workspace.audit_frames_dir.mkdir(parents=True, exist_ok=True)
    for split, (start, end) in splits.items():
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


def _advance_to_awaiting_selection(
    catalog: TrainingCatalog,
    study_id: str,
) -> None:
    for status in (
        "smoke_queued",
        "smoke_running",
        "smoke_passed",
        "calibration_queued",
        "calibration_running",
        "gated",
        "queued",
        "running",
        "completed",
        "awaiting_selection",
    ):
        catalog.transition_study(study_id, status)


def test_create_study_registers_ready_workspace_and_template(tmp_path: Path) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")

    created = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    loaded = catalog.get_study(created.study_id)

    assert created.study_id.startswith("klga-tft_optuna_v1-")
    assert created.candidate_version == f"tft_klga_{created.study_id}_final"
    assert created.status == "draft"
    assert loaded == created
    assert catalog.get_workspace(workspace.run_dir).data_order_hash == (
        "data-order-hash"
    )
    assert catalog.get_template(created.template_hash).template_id == (
        "tft_optuna_v1"
    )


def test_workspace_and_template_can_sync_without_creating_study(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")

    workspace_record = catalog.sync_workspace(workspace.run_dir)
    template_record = catalog.sync_template(
        "experiments/klga/tft_optuna_v1.json"
    )

    assert catalog.get_workspace(workspace.run_dir) == workspace_record
    assert catalog.get_template(template_record.template_hash) == template_record
    assert catalog.list_studies() == []


def test_existing_study_cannot_continue_after_audit_becomes_stale(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    workspace.write_json(
        workspace.audit_summary_path,
        {
            "ready": False,
            "station_id": "KLGA",
            "data_order_hash": "data-order-hash",
            "failures": ["source changed"],
        },
    )

    try:
        catalog.validate_study(study.study_id)
    except RuntimeError as exc:
        assert "ready=true" in str(exc)
    else:
        raise AssertionError("stale audit must prevent study continuation")


def test_existing_study_rejects_changed_data_hash_even_if_audit_is_ready(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    workspace.write_json(
        workspace.audit_summary_path,
        {
            "ready": True,
            "station_id": "KLGA",
            "data_order_hash": "changed-data-order-hash",
            "failures": [],
        },
    )

    with pytest.raises(RuntimeError, match="workspace audit changed"):
        catalog.validate_study(study.study_id)


def test_create_study_rejects_audited_split_range_mismatch(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    manifest_path = workspace.audit_frames_dir / "val.manifest.json"
    manifest = workspace.read_json(manifest_path)
    manifest["start_date"] = "2026-04-02"
    workspace.write_json(manifest_path, manifest)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")

    with pytest.raises(ValueError, match=r"val range.*does not match"):
        catalog.create_study(
            workspace.run_dir,
            "experiments/klga/tft_optuna_v1.json",
        )

    assert catalog.list_studies() == []


def test_concurrent_study_creation_allocates_unique_ids(tmp_path: Path) -> None:
    workspace = _ready_workspace(tmp_path)
    db_path = tmp_path / "training_catalog.db"

    def create_one(_index: int) -> str:
        with TrainingCatalog(db_path) as catalog:
            return catalog.create_study(
                workspace.run_dir,
                "experiments/klga/tft_optuna_v1.json",
            ).study_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        study_ids = list(pool.map(create_one, range(8)))

    with TrainingCatalog(db_path) as catalog:
        studies = catalog.list_studies()

    assert len(set(study_ids)) == 8
    assert len({study.candidate_version for study in studies}) == 8
    assert len(studies) == 8


def test_snapshot_write_failure_does_not_commit_study(tmp_path: Path) -> None:
    workspace = _ready_workspace(tmp_path)
    study_id = "klga-tft_optuna_v1-20260629T000000Z-deadbeef"
    (workspace.run_dir / "studies" / study_id).mkdir(parents=True)
    catalog = TrainingCatalog(
        tmp_path / "training_catalog.db",
        clock=lambda: datetime(2026, 6, 29, tzinfo=timezone.utc),
        nonce_factory=lambda: "deadbeef",
    )

    with pytest.raises(FileExistsError):
        catalog.create_study(
            workspace.run_dir,
            "experiments/klga/tft_optuna_v1.json",
        )

    assert catalog.list_studies() == []


def test_study_transition_updates_status_and_appends_event(tmp_path: Path) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )

    updated = catalog.transition_study(
        study.study_id,
        "smoke_queued",
        message="operator queued smoke",
    )
    events = catalog.list_events(study.study_id)

    assert updated.status == "smoke_queued"
    assert [(event.from_status, event.to_status) for event in events] == [
        (None, "draft"),
        ("draft", "smoke_queued"),
    ]
    assert events[-1].message == "operator queued smoke"


def test_catalog_indexes_artifact_with_content_hash(tmp_path: Path) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    artifact_path = workspace.run_dir / "studies" / study.study_id / "evidence.json"
    artifact_path.write_text('{"ready": true}', encoding="ascii")

    indexed = catalog.register_artifact(
        study.study_id,
        artifact_type="validation_evidence",
        path=artifact_path,
    )

    assert indexed.sha256
    assert indexed.size_bytes == artifact_path.stat().st_size
    assert catalog.list_artifacts(study.study_id) == [indexed]


def test_final_approval_can_be_issued_and_consumed_only_once(
    tmp_path: Path,
) -> None:
    workspace = _ready_workspace(tmp_path)
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    _advance_to_awaiting_selection(catalog, study.study_id)
    snapshot = ResolvedTrialSnapshot.from_dict(
        {
            "schema_version": 1,
            "study_id": study.study_id,
            "trial_number": 7,
            "execution_type": "trial",
            "template_hash": study.template_hash,
            "data_order_hash": "data-order-hash",
            "parameters": {"hidden_size": 32, "batch_size": 4},
        }
    )

    approved = catalog.approve_final(
        study.study_id,
        snapshot,
        approved_by="john",
    )
    consumed = catalog.consume_final_approval(study.study_id)

    assert consumed == approved
    assert catalog.get_final_trial_snapshot(study.study_id) == snapshot
    assert catalog.get_study(study.study_id).status == "final_running"
    with pytest.raises(PermissionError, match="already consumed"):
        catalog.consume_final_approval(study.study_id)
    with pytest.raises(PermissionError, match="already approved"):
        catalog.approve_final(study.study_id, snapshot, approved_by="john")
