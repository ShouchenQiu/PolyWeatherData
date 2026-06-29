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


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    study_id: str
    execution_type: str
    trial_number: int | None
    attempt: int
    status: str
    limits: dict[str, object]
    progress: dict[str, object]
    metrics: dict[str, object]
    error: str | None
    log_path: str | None
    worker_id: str | None
    created_at: str
    started_at: str | None
    heartbeat_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class StudyApprovalRecord:
    approval_id: str
    study_id: str
    approval_type: str
    approved_by: str
    approved_at: str


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
            CREATE TABLE IF NOT EXISTS executions (
                execution_id    TEXT PRIMARY KEY,
                study_id        TEXT NOT NULL REFERENCES studies(study_id),
                execution_type  TEXT NOT NULL,
                trial_number    INTEGER NOT NULL DEFAULT -1,
                attempt         INTEGER NOT NULL,
                status          TEXT NOT NULL,
                limits_json     TEXT NOT NULL,
                progress_json   TEXT NOT NULL,
                metrics_json    TEXT NOT NULL,
                error           TEXT,
                log_path        TEXT,
                worker_id       TEXT,
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                heartbeat_at    TEXT,
                finished_at     TEXT,
                UNIQUE(study_id, execution_type, trial_number, attempt)
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
            CREATE TABLE IF NOT EXISTS study_approvals (
                approval_id    TEXT PRIMARY KEY,
                study_id       TEXT NOT NULL REFERENCES studies(study_id),
                approval_type  TEXT NOT NULL,
                approved_by    TEXT NOT NULL,
                approved_at    TEXT NOT NULL,
                UNIQUE(study_id, approval_type)
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
        execution_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(executions)"
            ).fetchall()
        }
        if "worker_id" not in execution_columns:
            self._conn.execute(
                "ALTER TABLE executions ADD COLUMN worker_id TEXT"
            )
        execution_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(executions)"
            ).fetchall()
        }
        if "trial_number" not in execution_columns:
            self._conn.executescript(
                """
                BEGIN IMMEDIATE;
                ALTER TABLE executions RENAME TO executions_legacy;
                CREATE TABLE executions (
                    execution_id    TEXT PRIMARY KEY,
                    study_id        TEXT NOT NULL REFERENCES studies(study_id),
                    execution_type  TEXT NOT NULL,
                    trial_number    INTEGER NOT NULL DEFAULT -1,
                    attempt         INTEGER NOT NULL,
                    status          TEXT NOT NULL,
                    limits_json     TEXT NOT NULL,
                    progress_json   TEXT NOT NULL,
                    metrics_json    TEXT NOT NULL,
                    error           TEXT,
                    log_path        TEXT,
                    worker_id       TEXT,
                    created_at      TEXT NOT NULL,
                    started_at      TEXT,
                    heartbeat_at    TEXT,
                    finished_at     TEXT,
                    UNIQUE(
                        study_id, execution_type, trial_number, attempt
                    )
                );
                INSERT INTO executions(
                    execution_id, study_id, execution_type, trial_number,
                    attempt, status, limits_json, progress_json, metrics_json,
                    error, log_path, worker_id, created_at, started_at,
                    heartbeat_at, finished_at
                )
                SELECT
                    execution_id, study_id, execution_type, -1,
                    attempt, status, limits_json, progress_json, metrics_json,
                    error, log_path, worker_id, created_at, started_at,
                    heartbeat_at, finished_at
                FROM executions_legacy;
                DROP TABLE executions_legacy;
                COMMIT;
                """
            )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO catalog_migrations(version, applied_at)
            VALUES (2, ?)
            """,
            (self._now_iso(),),
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO catalog_migrations(version, applied_at)
            VALUES (3, ?)
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

    def list_workspaces(self) -> list[WorkspaceRecord]:
        rows = self._conn.execute(
            "SELECT * FROM workspaces ORDER BY registered_at, path"
        ).fetchall()
        return [_workspace_record(row) for row in rows]

    def get_template(self, template_hash: str) -> TemplateRecord:
        row = self._conn.execute(
            "SELECT * FROM templates WHERE template_hash = ?",
            (template_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(f"template is not registered: {template_hash}")
        return _template_record(row)

    def list_templates(self) -> list[TemplateRecord]:
        rows = self._conn.execute(
            "SELECT * FROM templates ORDER BY registered_at, template_hash"
        ).fetchall()
        return [_template_record(row) for row in rows]

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

    def enqueue_smoke(self, study_id: str) -> ExecutionRecord:
        """Create the first bounded smoke execution and advance its gate."""
        study = self.validate_study(study_id)
        if study.status != "draft":
            raise PermissionError("smoke can only be enqueued from draft")
        template_record = self.get_template(study.template_hash)
        template = ExperimentTemplate.from_dict(
            json.loads(template_record.config_json)
        )
        limits = asdict(template.execution.smoke)
        now = self._now_iso()
        execution_id = f"{study_id}-smoke-a1"
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.get_study(study_id)
            if current.status != "draft":
                raise PermissionError("smoke can only be enqueued from draft")
            self._conn.execute(
                """
                INSERT INTO executions(
                    execution_id, study_id, execution_type, attempt, status,
                    limits_json, progress_json, metrics_json, error, log_path,
                    worker_id, created_at, started_at, heartbeat_at, finished_at
                ) VALUES (?, ?, 'smoke', 1, 'queued', ?, '{}', '{}', NULL,
                          NULL, NULL, ?, NULL, NULL, NULL)
                """,
                (
                    execution_id,
                    study_id,
                    json.dumps(limits, sort_keys=True, separators=(",", ":")),
                    now,
                ),
            )
            self._conn.execute(
                """
                UPDATE studies SET status = 'smoke_queued', updated_at = ?
                WHERE study_id = ? AND status = 'draft'
                """,
                (now, study_id),
            )
            self._insert_event(
                study_id,
                from_status="draft",
                to_status="smoke_queued",
                message=f"smoke execution {execution_id} queued",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

    def enqueue_calibration(self, study_id: str) -> ExecutionRecord:
        """Queue baseline calibration only after successful smoke evidence."""
        study = self.validate_study(study_id)
        if study.status != "smoke_passed":
            raise PermissionError(
                "calibration requires study status smoke_passed"
            )
        template_record = self.get_template(study.template_hash)
        template = ExperimentTemplate.from_dict(
            json.loads(template_record.config_json)
        )
        limits = {
            **asdict(template.execution.calibration),
            "batch_size": template.trainer.batch_size,
            "precision": template.trainer.precision,
            "num_workers": template.trainer.num_workers,
        }
        now = self._now_iso()
        execution_id = f"{study_id}-calibration-a1"
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.get_study(study_id)
            if current.status != "smoke_passed":
                raise PermissionError(
                    "calibration requires study status smoke_passed"
                )
            self._conn.execute(
                """
                INSERT INTO executions(
                    execution_id, study_id, execution_type, attempt, status,
                    limits_json, progress_json, metrics_json, error, log_path,
                    worker_id, created_at, started_at, heartbeat_at, finished_at
                ) VALUES (?, ?, 'calibration', 1, 'queued', ?, '{}', '{}',
                          NULL, NULL, NULL, ?, NULL, NULL, NULL)
                """,
                (
                    execution_id,
                    study_id,
                    json.dumps(limits, sort_keys=True, separators=(",", ":")),
                    now,
                ),
            )
            self._conn.execute(
                """
                UPDATE studies
                SET status = 'calibration_queued', updated_at = ?
                WHERE study_id = ? AND status = 'smoke_passed'
                """,
                (now, study_id),
            )
            self._insert_event(
                study_id,
                from_status="smoke_passed",
                to_status="calibration_queued",
                message=f"calibration execution {execution_id} queued",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

    def get_execution(self, execution_id: str) -> ExecutionRecord:
        row = self._conn.execute(
            "SELECT * FROM executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"execution does not exist: {execution_id}")
        return _execution_record(row)

    def list_executions(self, study_id: str) -> list[ExecutionRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM executions
            WHERE study_id = ?
            ORDER BY created_at, execution_id
            """,
            (study_id,),
        ).fetchall()
        return [_execution_record(row) for row in rows]

    def start_search(self, study_id: str) -> StudyRecord:
        """Transition one approved Optuna study from queued to running."""
        study = self.validate_study(study_id)
        if study.status == "running":
            return study
        if study.status != "queued":
            raise PermissionError("search must be approved and queued")
        return self.transition_study(
            study_id,
            "running",
            message="Optuna search started",
        )

    def claim_next_search(self, *, worker_id: str) -> StudyRecord | None:
        """Claim an approved or interrupted-in-process Optuna study."""
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT * FROM studies
                WHERE status IN ('running', 'queued')
                ORDER BY
                    CASE status WHEN 'running' THEN 0 ELSE 1 END,
                    updated_at,
                    study_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            if row["status"] == "queued":
                cursor = self._conn.execute(
                    """
                    UPDATE studies SET status = 'running', updated_at = ?
                    WHERE study_id = ? AND status = 'queued'
                    """,
                    (now, row["study_id"]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("study status changed concurrently")
                self._insert_event(
                    row["study_id"],
                    from_status="queued",
                    to_status="running",
                    message=f"Optuna search claimed by {worker_id}",
                    occurred_at=now,
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_study(row["study_id"])

    def start_trial_execution(
        self,
        study_id: str,
        *,
        trial_number: int,
        limits: dict[str, object],
        log_path: Path | str,
    ) -> ExecutionRecord:
        """Record one running Optuna trial under an already-running study."""
        study = self.validate_study(study_id)
        if study.status != "running":
            raise PermissionError("trial requires a running Optuna study")
        if trial_number < 0:
            raise ValueError("trial_number must be non-negative")
        execution_id = f"{study_id}-trial-{trial_number:05d}-a1"
        now = self._now_iso()
        try:
            self._conn.execute(
                """
                INSERT INTO executions(
                    execution_id, study_id, execution_type, trial_number,
                    attempt, status,
                    limits_json, progress_json, metrics_json, error, log_path,
                    worker_id, created_at, started_at, heartbeat_at, finished_at
                ) VALUES (?, ?, 'trial', ?, 1, 'running', ?, '{}', '{}', NULL,
                          ?, 'optuna', ?, ?, ?, NULL)
                """,
                (
                    execution_id,
                    study_id,
                    trial_number,
                    json.dumps(limits, sort_keys=True, separators=(",", ":")),
                    str(Path(log_path).resolve()),
                    now,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"trial execution already exists: {execution_id}"
            ) from exc
        return self.get_execution(execution_id)

    def finish_trial_execution(
        self,
        execution_id: str,
        *,
        status: str,
        metrics: dict[str, object],
        error: str | None = None,
    ) -> ExecutionRecord:
        """Finish one trial without changing the parent study state."""
        if status not in {"succeeded", "pruned", "failed", "interrupted"}:
            raise ValueError("invalid terminal trial status")
        execution = self.get_execution(execution_id)
        if execution.execution_type != "trial" or execution.status != "running":
            raise PermissionError("only a running trial can finish")
        if status in {"failed", "interrupted"} and not (error or "").strip():
            raise ValueError("failed/interrupted trial requires an error")
        now = self._now_iso()
        cursor = self._conn.execute(
            """
            UPDATE executions
            SET status = ?, metrics_json = ?, error = ?, heartbeat_at = ?,
                finished_at = ?
            WHERE execution_id = ? AND status = 'running'
            """,
            (
                status,
                json.dumps(metrics, sort_keys=True, separators=(",", ":")),
                error,
                now,
                now,
                execution_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("trial execution status changed concurrently")
        return self.get_execution(execution_id)

    def finish_search(self, study_id: str, *, status: str) -> StudyRecord:
        if status not in {"completed", "budget_exhausted"}:
            raise ValueError("invalid terminal search status")
        return self.transition_study(
            study_id,
            status,
            message=f"Optuna search {status}",
        )

    def claim_next_execution(
        self,
        *,
        worker_id: str,
    ) -> ExecutionRecord | None:
        """Atomically claim the oldest queued execution for one worker."""
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT e.*, s.workspace_path, s.status AS study_status
                FROM executions e
                JOIN studies s ON s.study_id = e.study_id
                WHERE e.status = 'queued'
                ORDER BY e.created_at, e.execution_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            expected_study_status = f"{row['execution_type']}_queued"
            if row["study_status"] != expected_study_status:
                raise RuntimeError(
                    "queued execution does not match its study status"
                )
            running_study_status = f"{row['execution_type']}_running"
            log_path = (
                Path(row["workspace_path"])
                / "studies"
                / row["study_id"]
                / row["execution_type"]
                / f"attempt-{row['attempt']}"
                / "execution.log"
            ).resolve()
            cursor = self._conn.execute(
                """
                UPDATE executions
                SET status = 'running', log_path = ?, worker_id = ?,
                    started_at = ?, heartbeat_at = ?
                WHERE execution_id = ? AND status = 'queued'
                """,
                (
                    str(log_path),
                    worker_id,
                    now,
                    now,
                    row["execution_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution status changed concurrently")
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = ?, updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (
                    running_study_status,
                    now,
                    row["study_id"],
                    expected_study_status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                row["study_id"],
                from_status=expected_study_status,
                to_status=running_study_status,
                message=(
                    f"execution {row['execution_id']} claimed by {worker_id}"
                ),
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(row["execution_id"])

    def recover_interrupted_executions(
        self,
        *,
        worker_id: str,
    ) -> list[ExecutionRecord]:
        """Mark abandoned running work interrupted after GPU-lock takeover."""
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        now = self._now_iso()
        recovered_ids: list[str] = []
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self._conn.execute(
                """
                SELECT * FROM executions
                WHERE status IN ('running', 'cancel_requested')
                ORDER BY started_at, execution_id
                """
            ).fetchall()
            for row in rows:
                if row["status"] == "cancel_requested":
                    cursor = self._conn.execute(
                        """
                        UPDATE executions
                        SET status = 'cancelled', heartbeat_at = ?,
                            finished_at = ?
                        WHERE execution_id = ?
                          AND status = 'cancel_requested'
                        """,
                        (now, now, row["execution_id"]),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "execution status changed concurrently"
                        )
                    cursor = self._conn.execute(
                        """
                        UPDATE studies
                        SET status = 'cancelled', updated_at = ?
                        WHERE study_id = ?
                          AND status = 'cancel_requested'
                        """,
                        (now, row["study_id"]),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "cancel request does not match study status"
                        )
                    self._insert_event(
                        row["study_id"],
                        from_status="cancel_requested",
                        to_status="cancelled",
                        message=(
                            f"abandoned cancel completed by {worker_id}"
                        ),
                        occurred_at=now,
                    )
                    recovered_ids.append(row["execution_id"])
                    continue
                previous_worker = row["worker_id"] or "unknown worker"
                reason = (
                    f"worker {previous_worker} stopped before completion; "
                    f"recovered by {worker_id}"
                )
                metrics = json.loads(row["metrics_json"])
                started_at = row["started_at"]
                if started_at:
                    elapsed = max(
                        0.0,
                        (
                            datetime.fromisoformat(now)
                            - datetime.fromisoformat(started_at)
                        ).total_seconds(),
                    )
                    metrics["gpu_seconds"] = max(
                        float(metrics.get("gpu_seconds", 0.0)),
                        elapsed,
                    )
                cursor = self._conn.execute(
                    """
                    UPDATE executions
                    SET status = 'interrupted', error = ?, metrics_json = ?,
                        heartbeat_at = ?, finished_at = ?
                    WHERE execution_id = ? AND status = 'running'
                    """,
                    (
                        reason,
                        json.dumps(
                            metrics,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        now,
                        now,
                        row["execution_id"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("execution status changed concurrently")
                if row["execution_type"] == "trial":
                    study = self.get_study(row["study_id"])
                    if study.status != "running":
                        raise RuntimeError(
                            "running trial requires a running study"
                        )
                    self._insert_event(
                        row["study_id"],
                        from_status="running",
                        to_status="running",
                        message=(
                            f"trial execution {row['execution_id']} "
                            f"interrupted: {reason}"
                        ),
                        occurred_at=now,
                    )
                    recovered_ids.append(row["execution_id"])
                    continue
                from_study_status = f"{row['execution_type']}_running"
                cursor = self._conn.execute(
                    """
                    UPDATE studies SET status = 'interrupted', updated_at = ?
                    WHERE study_id = ? AND status = ?
                    """,
                    (now, row["study_id"], from_study_status),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "running execution does not match study status"
                    )
                self._insert_event(
                    row["study_id"],
                    from_status=from_study_status,
                    to_status="interrupted",
                    message=reason,
                    occurred_at=now,
                )
                recovered_ids.append(row["execution_id"])
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return [
            self.get_execution(execution_id)
            for execution_id in recovered_ids
        ]

    def retry_execution(self, execution_id: str) -> ExecutionRecord:
        """Queue a new attempt while preserving failed/interrupted evidence."""
        previous = self.get_execution(execution_id)
        if previous.execution_type == "trial":
            raise PermissionError(
                "Optuna trials are resumed by the study, not retried manually"
            )
        if previous.status not in {"failed", "interrupted", "cancelled"}:
            raise PermissionError(
                "retry requires failed, interrupted, or cancelled execution"
            )
        study = self.validate_study(previous.study_id)
        if study.status not in {"failed", "interrupted", "cancelled"}:
            raise PermissionError("study is not retryable")
        now = self._now_iso()
        queued_study_status = f"{previous.execution_type}_queued"
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS max_attempt
                FROM executions
                WHERE study_id = ? AND execution_type = ?
                """,
                (previous.study_id, previous.execution_type),
            ).fetchone()
            attempt = int(row["max_attempt"]) + 1
            retry_id = (
                f"{previous.study_id}-{previous.execution_type}-a{attempt}"
            )
            self._conn.execute(
                """
                INSERT INTO executions(
                    execution_id, study_id, execution_type, attempt, status,
                    limits_json, progress_json, metrics_json, error, log_path,
                    worker_id, created_at, started_at, heartbeat_at, finished_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, '{}', '{}', NULL, NULL,
                          NULL, ?, NULL, NULL, NULL)
                """,
                (
                    retry_id,
                    previous.study_id,
                    previous.execution_type,
                    attempt,
                    json.dumps(
                        previous.limits,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now,
                ),
            )
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = ?, updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (
                    queued_study_status,
                    now,
                    previous.study_id,
                    study.status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                previous.study_id,
                from_status=study.status,
                to_status=queued_study_status,
                message=(
                    f"execution {previous.execution_id} retried as {retry_id}"
                ),
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(retry_id)

    def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Request cooperative cancellation, or cancel queued work immediately."""
        execution = self.get_execution(execution_id)
        if execution.status in {"cancel_requested", "cancelled"}:
            return execution
        if execution.status not in {"queued", "running"}:
            raise PermissionError("only queued or running execution can cancel")
        study = self.get_study(execution.study_id)
        expected_study_status = (
            "running"
            if execution.execution_type == "trial"
            and execution.status == "running"
            else f"{execution.execution_type}_{execution.status}"
        )
        if study.status != expected_study_status:
            raise RuntimeError("execution does not match study status")
        now = self._now_iso()
        execution_status = (
            "cancelled" if execution.status == "queued" else "cancel_requested"
        )
        study_status = execution_status
        finished_at = now if execution_status == "cancelled" else None
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                """
                UPDATE executions
                SET status = ?, heartbeat_at = ?, finished_at = ?
                WHERE execution_id = ? AND status = ?
                """,
                (
                    execution_status,
                    now,
                    finished_at,
                    execution_id,
                    execution.status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution status changed concurrently")
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = ?, updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (
                    study_status,
                    now,
                    execution.study_id,
                    study.status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                execution.study_id,
                from_status=study.status,
                to_status=study_status,
                message=f"cancel requested for execution {execution_id}",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

    def complete_cancel(self, execution_id: str) -> ExecutionRecord:
        """Acknowledge cooperative cancellation from the running adapter."""
        execution = self.get_execution(execution_id)
        if execution.status == "cancelled":
            return execution
        if execution.status != "cancel_requested":
            raise PermissionError("execution has no pending cancel request")
        study = self.get_study(execution.study_id)
        if study.status != "cancel_requested":
            raise RuntimeError("cancel request does not match study status")
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                """
                UPDATE executions
                SET status = 'cancelled', heartbeat_at = ?, finished_at = ?
                WHERE execution_id = ? AND status = 'cancel_requested'
                """,
                (now, now, execution_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution status changed concurrently")
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = 'cancelled', updated_at = ?
                WHERE study_id = ? AND status = 'cancel_requested'
                """,
                (now, execution.study_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                execution.study_id,
                from_status="cancel_requested",
                to_status="cancelled",
                message=f"execution {execution_id} cancelled",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

    def start_execution(
        self,
        execution_id: str,
        *,
        log_path: Path | str,
    ) -> ExecutionRecord:
        execution = self.get_execution(execution_id)
        study = self.validate_study(execution.study_id)
        expected_study_status = f"{execution.execution_type}_queued"
        running_study_status = f"{execution.execution_type}_running"
        if execution.status != "queued" or study.status != expected_study_status:
            raise PermissionError("execution is not queued for its study")
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                """
                UPDATE executions
                SET status = 'running', log_path = ?, started_at = ?,
                    heartbeat_at = ?, worker_id = 'direct'
                WHERE execution_id = ? AND status = 'queued'
                """,
                (str(Path(log_path).resolve()), now, now, execution_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution status changed concurrently")
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = ?, updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (
                    running_study_status,
                    now,
                    study.study_id,
                    expected_study_status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                study.study_id,
                from_status=expected_study_status,
                to_status=running_study_status,
                message=f"execution {execution_id} started",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

    def record_execution_progress(
        self,
        execution_id: str,
        progress: dict[str, object],
    ) -> ExecutionRecord:
        execution = self.get_execution(execution_id)
        if execution.status != "running":
            raise PermissionError("progress requires a running execution")
        merged = {**execution.progress, **progress}
        now = self._now_iso()
        cursor = self._conn.execute(
            """
            UPDATE executions SET progress_json = ?, heartbeat_at = ?
            WHERE execution_id = ? AND status = 'running'
            """,
            (
                json.dumps(merged, sort_keys=True, separators=(",", ":")),
                now,
                execution_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("execution status changed concurrently")
        return self.get_execution(execution_id)

    def finish_execution(
        self,
        execution_id: str,
        *,
        metrics: dict[str, object],
    ) -> ExecutionRecord:
        execution = self.get_execution(execution_id)
        if execution.status != "running":
            raise PermissionError("only a running execution can succeed")
        if execution.execution_type == "smoke":
            from_study_status = "smoke_running"
            to_study_status = "smoke_passed"
        elif execution.execution_type == "calibration":
            from_study_status = "calibration_running"
            to_study_status = "gated"
        elif execution.execution_type == "final":
            from_study_status = "final_running"
            to_study_status = "candidate_ready"
        else:
            raise ValueError(
                f"unsupported execution type: {execution.execution_type}"
            )
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                """
                UPDATE executions
                SET status = 'succeeded', metrics_json = ?, heartbeat_at = ?,
                    finished_at = ?
                WHERE execution_id = ? AND status = 'running'
                """,
                (
                    json.dumps(metrics, sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                    execution_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution status changed concurrently")
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = ?, updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (
                    to_study_status,
                    now,
                    execution.study_id,
                    from_study_status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study is not running this execution")
            self._insert_event(
                execution.study_id,
                from_status=from_study_status,
                to_status=to_study_status,
                message=f"execution {execution_id} succeeded",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

    def fail_execution(
        self,
        execution_id: str,
        *,
        error: str,
    ) -> ExecutionRecord:
        execution = self.get_execution(execution_id)
        if execution.status not in {"queued", "running"}:
            raise PermissionError("only queued or running execution can fail")
        error = error.strip()
        if not error:
            raise ValueError("execution failure reason must be non-empty")
        study = self.get_study(execution.study_id)
        expected_statuses = {
            f"{execution.execution_type}_queued",
            f"{execution.execution_type}_running",
        }
        if study.status not in expected_statuses:
            raise PermissionError("study is not executing this task")
        now = self._now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                """
                UPDATE executions
                SET status = 'failed', error = ?, heartbeat_at = ?,
                    finished_at = ?
                WHERE execution_id = ? AND status IN ('queued', 'running')
                """,
                (error, now, now, execution_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution status changed concurrently")
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = 'failed', updated_at = ?
                WHERE study_id = ? AND status = ?
                """,
                (now, execution.study_id, study.status),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                execution.study_id,
                from_status=study.status,
                to_status="failed",
                message=f"execution {execution_id} failed: {error}",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_execution(execution_id)

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
        top_k_artifacts = [
            artifact
            for artifact in self.list_artifacts(study_id)
            if artifact.artifact_type == "top_k_validation_evidence"
        ]
        if len(top_k_artifacts) != 1:
            raise PermissionError(
                "final approval requires frozen Top-K validation evidence"
            )
        top_k = json.loads(
            Path(top_k_artifacts[0].path).read_text(encoding="utf-8")
        )
        selected = next(
            (
                item
                for item in top_k.get("trials", [])
                if int(item.get("trial_number", -1))
                == trial_snapshot.trial_number
            ),
            None,
        )
        if (
            selected is None
            or selected.get("trial_snapshot_hash")
            != trial_snapshot.snapshot_hash()
            or top_k.get("test_accessed") is not False
        ):
            raise PermissionError(
                "selected trial is not in frozen validation-only Top-K evidence"
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

    def approve_search(
        self,
        study_id: str,
        *,
        approved_by: str,
    ) -> StudyApprovalRecord:
        """Record explicit human approval before Optuna may be queued."""
        self.validate_study(study_id)
        approved_by = approved_by.strip()
        if not approved_by:
            raise ValueError("approved_by must be non-empty")
        now = self._now_iso()
        approval_id = f"search-{uuid.uuid4().hex}"
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self._conn.execute(
                """
                SELECT approval_id FROM study_approvals
                WHERE study_id = ? AND approval_type = 'search'
                """,
                (study_id,),
            ).fetchone()
            if existing is not None:
                raise PermissionError(
                    "search is already approved for this study"
                )
            study = self.get_study(study_id)
            if study.status != "gated":
                raise PermissionError(
                    "study must be gated before search approval"
                )
            self._conn.execute(
                """
                INSERT INTO study_approvals(
                    approval_id, study_id, approval_type, approved_by,
                    approved_at
                ) VALUES (?, ?, 'search', ?, ?)
                """,
                (approval_id, study_id, approved_by, now),
            )
            cursor = self._conn.execute(
                """
                UPDATE studies SET status = 'queued', updated_at = ?
                WHERE study_id = ? AND status = 'gated'
                """,
                (now, study_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("study status changed concurrently")
            self._insert_event(
                study_id,
                from_status="gated",
                to_status="queued",
                message=f"search approved by {approved_by}",
                occurred_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_study_approval(study_id, "search")

    def get_study_approval(
        self,
        study_id: str,
        approval_type: str,
    ) -> StudyApprovalRecord:
        row = self._conn.execute(
            """
            SELECT * FROM study_approvals
            WHERE study_id = ? AND approval_type = ?
            """,
            (study_id, approval_type),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"study approval does not exist: {study_id}/{approval_type}"
            )
        return StudyApprovalRecord(**dict(row))

    def list_study_approvals(
        self,
        study_id: str,
    ) -> list[StudyApprovalRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM study_approvals
            WHERE study_id = ?
            ORDER BY approved_at, approval_id
            """,
            (study_id,),
        ).fetchall()
        return [StudyApprovalRecord(**dict(row)) for row in rows]

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
            execution_id = f"{study_id}-final-a1"
            log_path = (
                Path(current.workspace_path)
                / "studies"
                / study_id
                / "final"
                / "attempt-1"
                / "execution.log"
            ).resolve()
            self._conn.execute(
                """
                INSERT INTO executions(
                    execution_id, study_id, execution_type, trial_number,
                    attempt, status, limits_json, progress_json, metrics_json,
                    error, log_path, worker_id, created_at, started_at,
                    heartbeat_at, finished_at
                ) VALUES (?, ?, 'final', ?, 1, 'running', ?, '{}', '{}',
                          NULL, ?, 'final-cli', ?, ?, ?, NULL)
                """,
                (
                    execution_id,
                    study_id,
                    int(row["selected_trial_number"]),
                    json.dumps(
                        {
                            "selected_trial_number": int(
                                row["selected_trial_number"]
                            ),
                            "trial_snapshot_hash": row[
                                "trial_snapshot_hash"
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    str(log_path),
                    now,
                    now,
                    now,
                ),
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


def _execution_record(row: sqlite3.Row) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=row["execution_id"],
        study_id=row["study_id"],
        execution_type=row["execution_type"],
        trial_number=(
            None if int(row["trial_number"]) < 0 else int(row["trial_number"])
        ),
        attempt=int(row["attempt"]),
        status=row["status"],
        limits=json.loads(row["limits_json"]),
        progress=json.loads(row["progress_json"]),
        metrics=json.loads(row["metrics_json"]),
        error=row["error"],
        log_path=row["log_path"],
        worker_id=row["worker_id"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        heartbeat_at=row["heartbeat_at"],
        finished_at=row["finished_at"],
    )


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
