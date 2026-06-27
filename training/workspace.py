"""Isolated run workspace and stage evidence for the training factory."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingWorkspace:
    """Own paths, manifests, and stage evidence below one run directory."""

    run_dir: Path

    def __init__(self, run_dir: Path | str) -> None:
        object.__setattr__(self, "run_dir", Path(run_dir))

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def data_store_dir(self) -> Path:
        return self.run_dir / "data_store"

    @property
    def model_store_dir(self) -> Path:
        return self.run_dir / "model_store"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.run_dir / "polyweather.db"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def stage_state_path(self) -> Path:
        return self.run_dir / "stage_state.json"

    @property
    def audit_summary_path(self) -> Path:
        return self.run_dir / "audit_summary.json"

    def ensure(self) -> None:
        for path in (
            self.data_store_dir,
            self.model_store_dir,
            self.checkpoints_dir,
            self.artifacts_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="ascii"))

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="ascii",
        )

    def require_ready_audit(self) -> dict[str, Any]:
        if not self.audit_summary_path.exists():
            raise RuntimeError("training requires audit_summary.json with ready=true")
        summary = self.read_json(self.audit_summary_path)
        if not summary.get("ready", False):
            raise RuntimeError("training requires audit_summary.json with ready=true")
        return summary

    def record_stage(self, stage: str, result: dict[str, Any]) -> None:
        state = (
            self.read_json(self.stage_state_path)
            if self.stage_state_path.exists()
            else {"stages": {}}
        )
        state.setdefault("stages", {})[stage] = {
            "status": result.get("status"),
            "version": result.get("version"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
        self.write_json(self.stage_state_path, state)

        manifest = (
            self.read_json(self.manifest_path)
            if self.manifest_path.exists()
            else {
                "run_id": self.run_id,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
            }
        )
        manifest["status"] = result.get("status", manifest.get("status"))
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, value in result.items():
            if key not in {"started_at", "finished_at", "status"}:
                manifest[key] = value
        self.write_json(self.manifest_path, manifest)
