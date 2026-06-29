"""Long-running single-GPU training worker CLI."""
from __future__ import annotations

import argparse
import os
import signal
import socket
from pathlib import Path
from typing import Sequence

from training.optuna_engine import OptunaEngine
from training.worker import TrainingWorker


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the managed training worker")
    parser.add_argument(
        "--catalog",
        default="runs/_management/training_catalog.db",
    )
    parser.add_argument(
        "--gpu-lock",
        default="runs/_management/training.gpu.lock",
    )
    parser.add_argument(
        "--optuna-storage",
        default="sqlite:///runs/_management/optuna.db",
    )
    parser.add_argument("--worker-id")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"

    stopping = False

    def run_search(catalog, study_id):
        OptunaEngine(
            catalog,
            storage_url=args.optuna_storage,
        ).run(study_id, pause_requested=lambda: stopping)
        return catalog.get_study(study_id)

    worker = TrainingWorker(
        Path(args.catalog),
        lock_path=Path(args.gpu_lock),
        worker_id=worker_id,
        search_runner=run_search,
    )
    if args.once:
        worker.run_once()
        return 0

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker.run_forever(
        poll_seconds=args.poll_seconds,
        stop_requested=lambda: stopping,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
