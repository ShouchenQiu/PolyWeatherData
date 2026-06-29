"""Persistent experiment catalog for training workspaces and studies."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from training.config import ExperimentTemplate
from training.config import ResolvedTrialSnapshot
from training.experiment import FinalEvaluationApproval
from training.workspace import TrainingWorkspace


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class WorkspaceRecord:
    path: str
    run_id: str
    station_id: str
    data_order_hash: str
    splits: dict[str, tuple[str, str]]
    registered_at: str
    updated_at: str


@dataclass(frozen=True)
class TemplateRecord:
    template_hash: str
    template_id: str
    path: str
    station_id: str
    config_json: str
    registered_at: str
    updated_at: str


@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    workspace_path: str
    template_hash: str
    station_id: str
    status: str
    budget_gpu_hours: float
    optuna_study_name: str
    candidate_version: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StudyEvent:
    event_id: int
    study_id: str
    from_status: str | None
    to_status: str
    message: str | None
    occurred_at: str


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: int
    study_id: str
    execution_id: str | None
    artifact_type: str
    path: str
    sha256: str
    size_bytes: int
    created_at: str


_STUDY_TRANSITIONS = {
    "draft": {"smoke_queued"},
    "smoke_queued": {"smoke_running", "cancel_requested"},
    "smoke_running": {
        "smoke_passed",
        "failed",
        "cancel_requested",
        "interrupted",
    },
    "smoke_passed": {"calibration_queued"},
    "calibration_queued": {"calibration_running", "cancel_requested"},
    "calibration_running": {
        "gated",
        "failed",
        "cancel_requested",
        "interrupted",
    },
    "gated": {"queued"},
    "queued": {"running", "cancel_requested"},
    "running": {
        "budget_exhausted",
        "completed",
        "failed",
        "cancel_requested",
        "interrupted",
    },
    "budget_exhausted": {"awaiting_selection"},
    "completed": {"awaiting_selection"},
    "awaiting_selection": {"final_approved"},
    "final_approved": {"final_running"},
    "final_running": {"candidate_ready", "failed", "interrupted"},
    "cancel_requested": {"cancelled"},
}


class TrainingCatalog:
    """Own catalog migration, registration, validation, and study identity."""

    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], datetime] = _utcnow,
        nonce_factory: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._nonce_factory = nonce_factory or (lambda: uuid.uuid4().hex[:8])
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._enable_wal()
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrainingCatalog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS catalog_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workspaces (
                path             TEXT PRIMARY KEY,
                run_id           TEXT NOT NULL,
                station_id       TEXT NOT NULL,
                data_order_hash  TEXT NOT NULL,
                splits_json      TEXT NOT NULL,
                registered_at    TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS templates (
                template_hash  TEXT PRIMARY KEY,
                template_id    TEXT NOT NULL,
                path           TEXT NOT NULL,
                station_id     TEXT NOT NULL,
                config_json    TEXT NOT NULL,
                registered_at  TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS studies (
                study_id           TEXT PRIMARY KEY,
                workspace_path     TEXT NOT NULL REFERENCES workspaces(path),
                template_hash      TEXT NOT NULL REFERENCES templates(template_hash),
                station_id         TEXT NOT NULL,
                status             TEXT NOT NULL,
                budget_gpu_hours   REAL NOT NULL,
                optuna_study_name  TEXT NOT NULL UNIQUE,
                candidate_version  TEXT NOT NULL UNIQUE,
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS study_events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id     TEXT NOT NULL REFERENCES studies(study_id),
                from_status  TEXT,
                to_status    TEXT NOT NULL,
                message      TEXT,
                occurred_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id       TEXT NOT NULL REFERENCES studies(study_id),
                execution_id   TEXT,
                artifact_type  TEXT NOT NULL,
                path           TEXT NOT NULL,
                sha256         TEXT NOT NULL,
                size_bytes     INTEGER NOT NULL,
                created_at     TEXT NOT NULL,
                UNIQUE(study_id, artifact_type, path)
            );
            CREATE TABLE IF NOT EXISTS final_approvals (
                approval_id           TEXT PRIMARY KEY,
                study_id              TEXT NOT NULL UNIQUE
                                      REFERENCES studies(study_id),
                selected_trial_number INTEGER NOT NULL,
                trial_snapshot_hash   TEXT NOT NULL,
                trial_snapshot_json   TEXT NOT NULL,
                approved_by           TEXT NOT NULL,
                approved_at           TEXT NOT NULL,
                consumed_at           TEXT
            );
            COMMIT;
            """
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO catalog_migrations(version, applied_at)
            VALUES (1, ?)
            """,
            (self._now_iso(),),
        )

    def _enable_wal(self) -> None:
        for attempt in range(20):
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc) or attempt == 19:
                    raise
                time.sleep(min(0.01 * (attempt + 1), 0.1))

    def create_study(
        self,
        workspace_path: Path | str,
        template_path: Path | str,
    ) -> StudyRecord:
        workspace_record, template, template_record = self._validated_records(
            workspace_path,
            template_path,
        )
        now = self._now_iso()
        snapshot_path: Path | None = None
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._upsert_workspace(workspace_record, now)
            self._upsert_template(template_record, now)
            record = self._insert_unique_study(
                workspace_record,
                template,
                template_record,
                now,
            )
            self._insert_event(
                record.study_id,
                from_status=None,
                to_status="draft",
                message="study created",
                occurred_at=now,
            )
            snapshot_path = self._write_study_snapshot(
                record,
                template,
                workspace_record,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            if snapshot_path is not None:
                _remove_created_snapshot(snapshot_path)
            raise
        return record

    def sync_workspace(self, workspace_path: Path | str) -> WorkspaceRecord:
        workspace = TrainingWorkspace(Path(workspace_path).resolve())
        audit = workspace.require_ready_audit()
        station_id = str(audit.get("station_id", "")).strip()
        data_order_hash = str(audit.get("data_order_hash", "")).strip()
        if not station_id or not data_order_hash:
            raise ValueError(
                "ready audit must contain station_id and data_order_hash"
            )
        splits = self._read_split_manifests(workspace, station_id)
        now = self._now_iso()
        record = WorkspaceRecord(
            path=str(workspace.run_dir),
            run_id=workspace.run_id,
            station_id=station_id,
            data_order_hash=data_order_hash,
            splits=splits,
            registered_at=now,
            updated_at=now,
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._upsert_workspace(record, now)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_workspace(workspace.run_dir)

    def sync_template(self, template_path: Path | str) -> TemplateRecord:
        path = Path(template_path).resolve()
        template = ExperimentTemplate.from_json(path)
        now = self._now_iso()
        record = TemplateRecord(
            template_hash=template.config_hash(),
            template_id=template.template_id,
            path=str(path),
            station_id=template.station,
            config_json=json.dumps(
                template.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            registered_at=now,
            updated_at=now,
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._upsert_template(record, now)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_template(record.template_hash)

    def get_workspace(self, path: Path | str) -> WorkspaceRecord:
        row = self._conn.execute(
            "SELECT * FROM workspaces WHERE path = ?",
            (str(Path(path).resolve()),),
        ).fetchone()
        if row is None:
            raise KeyError(f"workspace is not registered: {path}")
        return _workspace_record(row)

    def get_template(self, template_hash: str) -> TemplateRecord:
        row = self._conn.execute(
            "SELECT * FROM templates WHERE template_hash = ?",
            (template_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(f"template is not registered: {template_hash}")
        return _template_record(row)

    def get_study(self, study_id: str) -> StudyRecord:
        row = self._conn.execute(
            "SELECT * FROM studies WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"study does not exist: {study_id}")
        return _study_record(row)

    def list_studies(self) -> list[StudyRecord]:
        rows = self._conn.execute(
            "SELECT * FROM studies ORDER BY created_at, study_id"
        ).fetchall()
        return [_study_record(row) for row in rows]

    def transition_study(
        self,
        study_id: str,
        to_status: str,
        *,
        message: str | None = None,
    ) -> StudyRecord:
        self.validate_study(study_id)
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.get_study(study_id)
            allowed = _STUDY_TRANSITIONS.get(current.status, set())
            if to_status not in allowed:
                raise ValueError(
                    f"invalid study transition {current.status!r} -> "
                    f"{to_status!r}"
                )
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = ?, updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (to_status, now, study_id, current.status),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                study_id,
                from_status=current.status,
                to_status=to_status,
                message=message,
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_study(study_id)

    def list_events(self, study_id: str) -> list[StudyEvent]:
        rows = self._conn.execute(
            """
            SELECT * FROM study_events
            WHERE study_id = ?
            ORDER BY event_id
            """,
            (study_id,),
        ).fetchall()
        return [StudyEvent(**dict(row)) for row in rows]

    def register_artifact(
        self,
        study_id: str,
        *,
        artifact_type: str,
        path: Path | str,
        execution_id: str | None = None,
    ) -> ArtifactRecord:
        study = self.get_study(study_id)
        artifact_path = Path(path).resolve()
        study_root = (Path(study.workspace_path) / "studies" / study_id).resolve()
        if study_root not in (artifact_path, *artifact_path.parents):
            raise ValueError("artifact path must live inside its study directory")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact file does not exist: {artifact_path}")
        artifact_type = artifact_type.strip()
        if not artifact_type:
            raise ValueError("artifact_type must be non-empty")
        now = self._now_iso()
        digest = _sha256_file(artifact_path)
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO artifacts(
                    study_id, execution_id, artifact_type, path, sha256,
                    size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id,
                    execution_id,
                    artifact_type,
                    str(artifact_path),
                    digest,
                    artifact_path.stat().st_size,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("artifact is already registered") from exc
        return self._artifact_by_id(int(cursor.lastrowid))

    def list_artifacts(self, study_id: str) -> list[ArtifactRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM artifacts
            WHERE study_id = ?
            ORDER BY artifact_id
            """,
            (study_id,),
        ).fetchall()
        return [ArtifactRecord(**dict(row)) for row in rows]

    def _artifact_by_id(self, artifact_id: int) -> ArtifactRecord:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"artifact does not exist: {artifact_id}")
        return ArtifactRecord(**dict(row))

    def approve_final(
        self,
        study_id: str,
        trial_snapshot: ResolvedTrialSnapshot,
        *,
        approved_by: str,
    ) -> FinalEvaluationApproval:
        study = self.validate_study(study_id)
        workspace = self.get_workspace(study.workspace_path)
        approved_by = approved_by.strip()
        if not approved_by:
            raise ValueError("approved_by must be non-empty")
        if (
            trial_snapshot.execution_type != "trial"
            or trial_snapshot.study_id != study_id
            or trial_snapshot.template_hash != study.template_hash
            or trial_snapshot.data_order_hash != workspace.data_order_hash
        ):
            raise PermissionError(
                "selected trial snapshot does not match study evidence"
            )
        now = self._now_iso()
        approval_id = f"final-{uuid.uuid4().hex}"
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self._conn.execute(
                "SELECT approval_id FROM final_approvals WHERE study_id = ?",
                (study_id,),
            ).fetchone()
            if existing is not None:
                raise PermissionError("final is already approved for this study")
            current = self.get_study(study_id)
            if current.status != "awaiting_selection":
                raise PermissionError(
                    "study must be awaiting_selection before final approval"
                )
            self._conn.execute(
                """
                INSERT INTO final_approvals(
                    approval_id, study_id, selected_trial_number,
                    trial_snapshot_hash, trial_snapshot_json, approved_by,
                    approved_at, consumed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    approval_id,
                    study_id,
                    trial_snapshot.trial_number,
                    trial_snapshot.snapshot_hash(),
                    json.dumps(
                        trial_snapshot.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    approved_by,
                    now,
                ),
            )
            self._conn.execute(
                """
                UPDATE studies SET status = 'final_approved', updated_at = ?
                WHERE study_id = ? AND status = 'awaiting_selection'
                """,
                (now, study_id),
            )
            self._insert_event(
                study_id,
                from_status="awaiting_selection",
                to_status="final_approved",
                message=f"final trial {trial_snapshot.trial_number} approved",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self._final_approval(study_id)

    def consume_final_approval(
        self,
        study_id: str,
    ) -> FinalEvaluationApproval:
        self.validate_study(study_id)
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT * FROM final_approvals WHERE study_id = ?",
                (study_id,),
            ).fetchone()
            if row is None:
                raise PermissionError("final is not approved for this study")
            if row["consumed_at"] is not None:
                raise PermissionError(
                    "final approval was already consumed for this study"
                )
            current = self.get_study(study_id)
            if current.status != "final_approved":
                raise PermissionError("study is not ready to start final")
            cursor = self._conn.execute(
                """
                UPDATE final_approvals SET consumed_at = ?
                WHERE study_id = ? AND consumed_at IS NULL
                """,
                (now, study_id),
            )
            if cursor.rowcount != 1:
                raise PermissionError(
                    "final approval was already consumed for this study"
                )
            self._conn.execute(
                """
                UPDATE studies SET status = 'final_running', updated_at = ?
                WHERE study_id = ? AND status = 'final_approved'
                """,
                (now, study_id),
            )
            self._insert_event(
                study_id,
                from_status="final_approved",
                to_status="final_running",
                message="final approval consumed",
                occurred_at=now,
            )
            approval = _approval_from_row(row)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return approval

    def _final_approval(self, study_id: str) -> FinalEvaluationApproval:
        row = self._conn.execute(
            "SELECT * FROM final_approvals WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"final approval does not exist: {study_id}")
        return _approval_from_row(row)

    def get_final_trial_snapshot(
        self,
        study_id: str,
    ) -> ResolvedTrialSnapshot:
        row = self._conn.execute(
            """
            SELECT trial_snapshot_json FROM final_approvals
            WHERE study_id = ?
            """,
            (study_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"final approval does not exist: {study_id}")
        return ResolvedTrialSnapshot.from_dict(
            json.loads(row["trial_snapshot_json"])
        )

    def validate_study(self, study_id: str) -> StudyRecord:
        study = self.get_study(study_id)
        registered_template = self.get_template(study.template_hash)
        workspace, _template, template = self._validated_records(
            study.workspace_path,
            registered_template.path,
        )
        registered_workspace = self.get_workspace(study.workspace_path)
        if template.template_hash != study.template_hash:
            raise RuntimeError(
                "study template changed after creation; create a new study"
            )
        if (
            workspace.data_order_hash != registered_workspace.data_order_hash
            or workspace.splits != registered_workspace.splits
        ):
            raise RuntimeError(
                "study workspace audit changed after creation; create a new study"
            )
        return study

    def _validated_records(
        self,
        workspace_path: Path | str,
        template_path: Path | str,
    ) -> tuple[WorkspaceRecord, ExperimentTemplate, TemplateRecord]:
        workspace = TrainingWorkspace(Path(workspace_path).resolve())
        audit = workspace.require_ready_audit()
        template_file = Path(template_path).resolve()
        template = ExperimentTemplate.from_json(template_file)
        data_order_hash = template.validate_audit(audit)
        splits = self._validate_split_manifests(workspace, template)
        now = self._now_iso()
        workspace_record = WorkspaceRecord(
            path=str(workspace.run_dir),
            run_id=workspace.run_id,
            station_id=template.station,
            data_order_hash=data_order_hash,
            splits=splits,
            registered_at=now,
            updated_at=now,
        )
        template_record = TemplateRecord(
            template_hash=template.config_hash(),
            template_id=template.template_id,
            path=str(template_file),
            station_id=template.station,
            config_json=json.dumps(
                template.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            registered_at=now,
            updated_at=now,
        )
        return workspace_record, template, template_record

    @staticmethod
    def _validate_split_manifests(
        workspace: TrainingWorkspace,
        template: ExperimentTemplate,
    ) -> dict[str, tuple[str, str]]:
        validated: dict[str, tuple[str, str]] = {}
        for split in ("train", "val", "test"):
            path = workspace.audit_frames_dir / f"{split}.manifest.json"
            if not path.exists():
                raise ValueError(f"workspace missing audited {split} manifest")
            manifest = workspace.read_json(path)
            expected_start, expected_end = template.splits[split]
            actual = (
                str(manifest.get("start_date", "")),
                str(manifest.get("end_date", "")),
            )
            if manifest.get("split") != split:
                raise ValueError(f"audited {split} manifest has wrong split")
            if manifest.get("station_id") != template.station:
                raise ValueError(
                    f"audited {split} station does not match template station"
                )
            if actual != (expected_start, expected_end):
                raise ValueError(
                    f"audited {split} range {actual} does not match "
                    f"template range {(expected_start, expected_end)}"
                )
            if not manifest.get("source_fingerprint"):
                raise ValueError(
                    f"audited {split} manifest lacks source_fingerprint"
                )
            if int(manifest.get("row_count", 0)) <= 0:
                raise ValueError(f"audited {split} manifest has no rows")
            validated[split] = actual
        return validated

    @staticmethod
    def _read_split_manifests(
        workspace: TrainingWorkspace,
        station_id: str,
    ) -> dict[str, tuple[str, str]]:
        splits: dict[str, tuple[str, str]] = {}
        for split in ("train", "val", "test"):
            path = workspace.audit_frames_dir / f"{split}.manifest.json"
            if not path.exists():
                raise ValueError(f"workspace missing audited {split} manifest")
            manifest = workspace.read_json(path)
            if manifest.get("split") != split:
                raise ValueError(f"audited {split} manifest has wrong split")
            if manifest.get("station_id") != station_id:
                raise ValueError(
                    f"audited {split} station does not match ready audit"
                )
            start = str(manifest.get("start_date", ""))
            end = str(manifest.get("end_date", ""))
            if not start or not end:
                raise ValueError(f"audited {split} manifest lacks range")
            if not manifest.get("source_fingerprint"):
                raise ValueError(
                    f"audited {split} manifest lacks source_fingerprint"
                )
            if int(manifest.get("row_count", 0)) <= 0:
                raise ValueError(f"audited {split} manifest has no rows")
            splits[split] = (start, end)
        return splits

    def _upsert_workspace(self, record: WorkspaceRecord, now: str) -> None:
        self._conn.execute(
            """
            INSERT INTO workspaces(
                path, run_id, station_id, data_order_hash, splits_json,
                registered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                run_id=excluded.run_id,
                station_id=excluded.station_id,
                data_order_hash=excluded.data_order_hash,
                splits_json=excluded.splits_json,
                updated_at=excluded.updated_at
            """,
            (
                record.path,
                record.run_id,
                record.station_id,
                record.data_order_hash,
                json.dumps(record.splits, sort_keys=True),
                record.registered_at,
                now,
            ),
        )

    def _upsert_template(self, record: TemplateRecord, now: str) -> None:
        self._conn.execute(
            """
            INSERT INTO templates(
                template_hash, template_id, path, station_id, config_json,
                registered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(template_hash) DO UPDATE SET
                path=excluded.path,
                config_json=excluded.config_json,
                updated_at=excluded.updated_at
            """,
            (
                record.template_hash,
                record.template_id,
                record.path,
                record.station_id,
                record.config_json,
                record.registered_at,
                now,
            ),
        )

    def _insert_unique_study(
        self,
        workspace: WorkspaceRecord,
        template: ExperimentTemplate,
        template_record: TemplateRecord,
        now: str,
    ) -> StudyRecord:
        timestamp = self._clock().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for _attempt in range(10):
            nonce = self._nonce_factory().lower().replace("-", "")[:8]
            study_id = (
                f"{template.station.lower()}-{template.template_id}-"
                f"{timestamp}-{nonce}"
            )
            candidate_version = (
                f"tft_{template.station.lower()}_{study_id}_final"
            )
            record = StudyRecord(
                study_id=study_id,
                workspace_path=workspace.path,
                template_hash=template_record.template_hash,
                station_id=template.station,
                status="draft",
                budget_gpu_hours=template.study.budget_gpu_hours,
                optuna_study_name=study_id,
                candidate_version=candidate_version,
                created_at=now,
                updated_at=now,
            )
            try:
                self._conn.execute(
                    """
                    INSERT INTO studies(
                        study_id, workspace_path, template_hash, station_id,
                        status, budget_gpu_hours, optuna_study_name,
                        candidate_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.study_id,
                        record.workspace_path,
                        record.template_hash,
                        record.station_id,
                        record.status,
                        record.budget_gpu_hours,
                        record.optuna_study_name,
                        record.candidate_version,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            except sqlite3.IntegrityError:
                continue
            return record
        raise RuntimeError("could not allocate a unique study_id")

    def _insert_event(
        self,
        study_id: str,
        *,
        from_status: str | None,
        to_status: str,
        message: str | None,
        occurred_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO study_events(
                study_id, from_status, to_status, message, occurred_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (study_id, from_status, to_status, message, occurred_at),
        )

    @staticmethod
    def _write_study_snapshot(
        record: StudyRecord,
        template: ExperimentTemplate,
        workspace: WorkspaceRecord,
    ) -> Path:
        path = (
            Path(record.workspace_path)
            / "studies"
            / record.study_id
            / "study_snapshot.json"
        )
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text(
            json.dumps(
                {
                    "study": asdict(record),
                    "template": template.to_dict(),
                    "data_order_hash": workspace.data_order_hash,
                    "splits": {
                        key: list(value)
                        for key, value in workspace.splits.items()
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="ascii",
        )
        return path

    def _now_iso(self) -> str:
        return self._clock().astimezone(timezone.utc).isoformat()


def _workspace_record(row: sqlite3.Row) -> WorkspaceRecord:
    splits = json.loads(row["splits_json"])
    return WorkspaceRecord(
        path=row["path"],
        run_id=row["run_id"],
        station_id=row["station_id"],
        data_order_hash=row["data_order_hash"],
        splits={key: tuple(value) for key, value in splits.items()},
        registered_at=row["registered_at"],
        updated_at=row["updated_at"],
    )


def _template_record(row: sqlite3.Row) -> TemplateRecord:
    return TemplateRecord(**dict(row))


def _study_record(row: sqlite3.Row) -> StudyRecord:
    return StudyRecord(**dict(row))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _approval_from_row(row: sqlite3.Row) -> FinalEvaluationApproval:
    return FinalEvaluationApproval.from_catalog_record(
        {
            "approval_id": row["approval_id"],
            "study_id": row["study_id"],
            "selected_trial_number": row["selected_trial_number"],
            "trial_snapshot_hash": row["trial_snapshot_hash"],
            "approved_by": row["approved_by"],
            "approved_at": row["approved_at"],
        }
    )


def _remove_created_snapshot(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError:
        pass
