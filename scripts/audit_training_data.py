"""CLI adapter for training-data audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.stations import STATIONS
from training.audit import (
    audit_training_readiness,
    audit_weather_backfill,
    audit_workspace,
)


def _split(value: str) -> tuple[str, str]:
    start, end = value.split(":", 1)
    return start, end


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Audit training weather data")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--data-store", default=None)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--train", help="start:end")
    parser.add_argument("--val", help="start:end")
    parser.add_argument("--test", help="start:end")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="report readiness progress every N init windows",
    )
    args = parser.parse_args()

    if not args.run_dir and not args.data_store:
        parser.error("provide --run-dir or --data-store")
    run_dir = Path(args.run_dir) if args.run_dir else None
    data_store = (
        Path(args.data_store)
        if args.data_store
        else run_dir / "data_store"
    )
    split_values = (args.train, args.val, args.test)
    if all(split_values):
        split_ranges = {
            "train": _split(args.train),
            "val": _split(args.val),
            "test": _split(args.test),
        }
        if run_dir:
            result = audit_workspace(
                run_dir,
                args.station,
                split_ranges,
                progress=True,
                progress_every=args.progress_every,
            )
            summary = result["audit"]
        else:
            summary = audit_training_readiness(
                data_store,
                args.station,
                split_ranges,
                progress=True,
                progress_every=args.progress_every,
            )
    elif args.start and args.end:
        summary = audit_weather_backfill(
            data_store,
            args.station,
            args.start,
            args.end,
        )
    else:
        parser.error("provide --start/--end or all of --train/--val/--test")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
