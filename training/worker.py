"""Single-GPU background worker primitives for managed training."""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import Callable, TextIO

from training.catalog import ExecutionRecord, StudyRecord, TrainingCatalog
from training.trial_runner import run_training_execution


class GpuLock:
    """Cross-process exclusive lock for the one local training GPU."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._handle: TextIO | None = None

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="ascii")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"GPU is already owned; lock is held: {self.path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, *_args: object) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


ExecutionRunner = Callable[
    [TrainingCatalog, str],
    ExecutionRecord,
]
SearchRunner = Callable[[TrainingCatalog, str], StudyRecord]


class TrainingWorker:
    """Recover, claim, and execute at most one catalog task under GPU lock."""

    def __init__(
        self,
        catalog_path: Path | str,
        *,
        lock_path: Path | str,
        worker_id: str,
        execution_runner: ExecutionRunner | None = None,
        search_runner: SearchRunner | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.lock_path = Path(lock_path)
        self.worker_id = worker_id.strip()
        if not self.worker_id:
            raise ValueError("worker_id must be non-empty")
        self._execution_runner = (
            execution_runner or _run_claimed_training_execution
        )
        self._search_runner = search_runner

    def run_once(self) -> ExecutionRecord | StudyRecord | None:
        with GpuLock(self.lock_path):
            with TrainingCatalog(self.catalog_path) as catalog:
                catalog.recover_interrupted_executions(
                    worker_id=self.worker_id
                )
                claimed = catalog.claim_next_execution(
                    worker_id=self.worker_id
                )
                if claimed is not None:
                    try:
                        return self._execution_runner(
                            catalog,
                            claimed.execution_id,
                        )
                    except Exception as exc:
                        current = catalog.get_execution(claimed.execution_id)
                        if current.status == "running":
                            catalog.fail_execution(
                                claimed.execution_id,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        raise
                if self._search_runner is None:
                    return None
                search = catalog.claim_next_search(worker_id=self.worker_id)
                if search is None:
                    return None
                try:
                    return self._search_runner(catalog, search.study_id)
                except Exception as exc:
                    current = catalog.get_study(search.study_id)
                    if current.status == "running":
                        catalog.transition_study(
                            search.study_id,
                            "failed",
                            message=(
                                f"Optuna engine failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    raise

    def run_forever(
        self,
        *,
        poll_seconds: float = 5.0,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        should_stop = stop_requested or (lambda: False)
        while not should_stop():
            completed = self.run_once()
            if completed is None:
                time.sleep(poll_seconds)


def _run_claimed_training_execution(
    catalog: TrainingCatalog,
    execution_id: str,
) -> ExecutionRecord:
    return run_training_execution(catalog, execution_id)
