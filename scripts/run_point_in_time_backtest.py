"""Run one deterministic strategy against immutable replay inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.forecast_cache import ForecastCache
from backtest.study import BacktestStudy, StudyInputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=["baseline", "near_term", "liquid"],
        default="baseline",
    )
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--start-target")
    parser.add_argument("--end-target")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--study-id")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.strategy != "baseline":
        raise ValueError(
            "governed diagnostic replay supports only the frozen baseline "
            "strategy"
        )
    if bool(args.start_target) != bool(args.end_target):
        raise ValueError(
            "--start-target and --end-target must be provided together"
        )
    clob_manifest = json.loads(
        (args.clob_root / "manifest.json").read_text(encoding="utf-8")
    )
    start_target, end_target = (
        (args.start_target, args.end_target)
        if args.start_target
        else tuple(clob_manifest["requested_range"])
    )
    inputs = StudyInputs(
        forecast_root=args.forecast_root,
        quotes_root=args.quotes_root,
        clob_root=args.clob_root,
        actuals_path=args.actual_highs,
    )
    catalog_path = args.catalog or args.output.parent / "studies.sqlite"
    study = BacktestStudy(catalog_path)
    forecast = ForecastCache.load(args.forecast_root)
    split_roles = set(forecast.lineages["split_role"].astype(str))
    if len(split_roles) != 1:
        raise ValueError("forecast cache must contain exactly one split_role")
    split_role = next(iter(split_roles))
    if split_role == "oof_development":
        if args.study_id:
            study_id = args.study_id
        else:
            record = study.create_development(
                station_id=args.station,
                start_target=str(start_target),
                end_target=str(end_target),
                inputs=inputs,
                strategy={"min_edge": 0.04, "risk_usd": 50.0},
            )
            study_id = record.study_id
        study.run_development(
            study_id,
            output=args.output,
            folds=args.folds,
            bootstrap_samples=args.bootstrap_samples,
        )
    elif split_role == "post_test_holdout":
        if not args.study_id:
            raise ValueError(
                "--study-id is required for post_test_holdout evaluation"
            )
        study.run_holdout(
            args.study_id,
            inputs=inputs,
            output=args.output,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        raise ValueError(f"{split_role} is forbidden for trading studies")
    summary = json.loads(
        (args.output / "summary.json").read_text(encoding="utf-8")
    )
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
