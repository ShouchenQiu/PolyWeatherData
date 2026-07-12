"""Build strict historical TFT trajectories for offline daily-high research."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.historical_crossfit import (
    DefaultHistoricalFoldExecutor,
    HistoricalCrossfitRunner,
    historical_six_fold_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument(
        "--plan", default="historical_6fold_20250401_20260331"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-template", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = historical_six_fold_plan()
    if args.plan != plan.plan_id:
        raise ValueError(f"unsupported cross-fit plan: {args.plan}")
    if args.station != plan.station_id:
        raise ValueError(f"plan {plan.plan_id} is fixed to {plan.station_id}")
    template = args.reference_template or (
        args.source_run
        / "artifacts"
        / "ab_experiments"
        / "hist-error-ab-a-seed42.json"
    )
    result = HistoricalCrossfitRunner(
        execute_fold=DefaultHistoricalFoldExecutor(reference_template=template),
        plan=plan,
    ).run(source_run=args.source_run, output=args.output, resume=args.resume)
    print(
        "completed=" + ",".join(result.completed)
        + " skipped=" + ",".join(result.skipped),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
