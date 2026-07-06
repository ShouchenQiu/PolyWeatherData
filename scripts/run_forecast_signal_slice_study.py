"""Discover development-only forecast advantage slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.signal_slice_study import (
    run_forecast_signal_slice_study,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-root", required=True, type=Path)
    parser.add_argument("--forecast-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-training-events", type=int, default=12)
    parser.add_argument("--minimum-shortlist-events", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_forecast_signal_slice_study(
        signal_root=args.signal_root,
        forecast_root=args.forecast_root,
        output=args.output,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_training_events=args.min_training_events,
        minimum_shortlist_events=args.minimum_shortlist_events,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
