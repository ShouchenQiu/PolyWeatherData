import json

import pytest

from training.trainer import train_workspace
from training.workspace import TrainingWorkspace


def test_workspace_records_stage_and_requires_ready_audit(tmp_path) -> None:
    workspace = TrainingWorkspace(tmp_path / "runs" / "experiment-01")
    workspace.ensure()
    workspace.record_stage(
        "prepare",
        {
            "status": "prepared",
            "station": "KLGA",
            "started_at": "2026-06-27T00:00:00+00:00",
            "finished_at": "2026-06-27T01:00:00+00:00",
        },
    )

    state = json.loads(workspace.stage_state_path.read_text(encoding="ascii"))
    manifest = json.loads(workspace.manifest_path.read_text(encoding="ascii"))
    assert state["stages"]["prepare"]["status"] == "prepared"
    assert manifest["run_id"] == "experiment-01"
    assert manifest["station"] == "KLGA"

    with pytest.raises(RuntimeError, match="audit_summary"):
        workspace.require_ready_audit()

    workspace.write_json(workspace.audit_summary_path, {"ready": True})
    assert workspace.require_ready_audit()["ready"] is True


def test_train_workspace_requires_audit_and_records_result(tmp_path) -> None:
    run_dir = tmp_path / "run"
    workspace = TrainingWorkspace(run_dir)
    workspace.ensure()

    with pytest.raises(RuntimeError, match="audit_summary"):
        train_workspace(
            run_dir,
            station_id="KLGA",
            version="candidate-v1",
            train_fn=lambda **_kwargs: {"version": "candidate-v1"},
        )

    workspace.write_json(workspace.audit_summary_path, {"ready": True})
    result = train_workspace(
        run_dir,
        station_id="KLGA",
        version="candidate-v1",
        train_fn=lambda **_kwargs: {"version": "candidate-v1"},
    )

    assert result["status"] == "trained"
    assert json.loads((run_dir / "train_result.json").read_text()) == {
        "version": "candidate-v1"
    }
