"""Run one catalog-managed smoke or calibration execution."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from training.catalog import TrainingCatalog
from training.trial_runner import run_training_execution
from training.worker import GpuLock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one queued managed training execution"
    )
    parser.add_argument(
        "--catalog",
        default="runs/_management/training_catalog.db",
    )
    parser.add_argument("--execution", required=True)
    parser.add_argument(
        "--gpu-lock",
        default="runs/_management/training.gpu.lock",
    )
    args = parser.parse_args(argv)
    with GpuLock(args.gpu_lock):
        with TrainingCatalog(args.catalog) as catalog:
            result = run_training_execution(catalog, args.execution)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
