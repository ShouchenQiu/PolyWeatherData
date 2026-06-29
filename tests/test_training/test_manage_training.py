from __future__ import annotations

import json
from pathlib import Path

from scripts.manage_training import main
from training.workspace import TrainingWorkspace


def _workspace(tmp_path: Path) -> TrainingWorkspace:
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
    ranges = {
        "train": ("2024-10-07", "2026-03-31"),
        "val": ("2026-04-01", "2026-04-30"),
        "test": ("2026-05-01", "2026-05-31"),
    }
    for split, (start, end) in ranges.items():
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


def test_manage_training_create_study_prints_managed_identity(
    tmp_path: Path,
    capsys,
) -> None:
    workspace = _workspace(tmp_path)

    exit_code = main(
        [
            "--catalog",
            str(tmp_path / "training_catalog.db"),
            "create-study",
            "--workspace",
            str(workspace.run_dir),
            "--template",
            "experiments/klga/tft_optuna_v1.json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"].startswith("klga-tft_optuna_v1-")
    assert payload["candidate_version"].endswith("_final")
    assert payload["status"] == "draft"
