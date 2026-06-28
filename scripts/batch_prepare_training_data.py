"""CLI adapter for training-factory weather preparation."""
from __future__ import annotations

import argparse
from pathlib import Path

from config.stations import STATIONS
from training.prepare_data import DEFAULT_END, DEFAULT_START, prepare_training_data


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Prepare batch training weather data")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-hrrr", action="store_true")
    parser.add_argument("--skip-iem", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    args = parser.parse_args()

    prepare_training_data(
        run_dir=Path(args.run_dir),
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        workers=args.workers,
        dry_run=args.dry_run,
        skip_hrrr=args.skip_hrrr,
        skip_iem=args.skip_iem,
        skip_decoder=args.skip_decoder,
    )


if __name__ == "__main__":
    _cli()
