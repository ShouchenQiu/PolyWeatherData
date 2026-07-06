"""Evaluate raw forecast probabilities against normalized synthetic midpoints."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from backtest.signal_study import (
    run_forecast_signal_study,
    signal_study_id,
)
from backtest.study import StudyInputs, validate_study_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--start-target")
    parser.add_argument("--end-target")
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--probability-floor", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.start_target) != bool(args.end_target):
        raise ValueError(
            "--start-target and --end-target must be provided together"
        )
    clob_manifest = json.loads(
        (args.clob_root / "manifest.json").read_text(encoding="utf-8")
    )
    start_target, end_target = (
        (str(args.start_target), str(args.end_target))
        if args.start_target
        else tuple(str(value) for value in clob_manifest["requested_range"])
    )
    inputs = StudyInputs(
        forecast_root=args.forecast_root,
        quotes_root=args.quotes_root,
        clob_root=args.clob_root,
        actuals_path=args.actual_highs,
    )
    validation = validate_study_inputs(
        study_id="signal-pending",
        station_id=args.station,
        start_target=start_target,
        end_target=end_target,
        inputs=inputs,
        expected_split_role="oof_development",
    )
    study_id = signal_study_id(
        station_id=args.station,
        start_target=start_target,
        end_target=end_target,
        input_sha256=validation.input_sha256,
        ece_bins=args.ece_bins,
        probability_floor=args.probability_floor,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    manifest = run_forecast_signal_study(
        replace(validation, study_id=study_id),
        output=args.output,
        ece_bins=args.ece_bins,
        probability_floor=args.probability_floor,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
