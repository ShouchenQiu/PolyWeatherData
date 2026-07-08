"""Validate frozen forecast distribution advantage slices on a holdout split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.signal_slice_study import (
    run_frozen_forecast_distribution_validation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-slice-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validation-signal-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validation-forecast-root",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_frozen_forecast_distribution_validation(
        development_slice_root=args.development_slice_root,
        validation_signal_root=args.validation_signal_root,
        validation_forecast_root=args.validation_forecast_root,
        output=args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
