"""Build April development forecasts with leakage-safe OOF calibration."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd

from training.backtest_forecast_generation import (
    build_forecast_cache,
    fit_oof_calibrations,
)
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.mc_calibration import McCalibration
from training.mc_calibration import fit_mc_calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable OOF forecast cache for strategy development."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start-init")
    parser.add_argument("--end-init")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=2_000)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    fit_calibration: Callable[..., McCalibration] = fit_mc_calibration,
) -> int:
    args = _parser().parse_args(argv)
    station = STATIONS[args.station]
    predictions = pd.read_parquet(args.predictions)
    predictions["init_time"] = pd.to_datetime(
        predictions["init_time"], utc=True
    )
    if args.start_init:
        predictions = predictions[
            predictions["init_time"] >= _utc(args.start_init)
        ]
    if args.end_init:
        end = _utc(args.end_init) + pd.Timedelta(days=1)
        predictions = predictions[predictions["init_time"] < end]
    if predictions.empty:
        raise ValueError("no hourly predictions remain in the requested range")

    availability = HrrrAvailabilityDataset.load(args.availability_root)
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date"]],
        on="event_id",
        how="inner",
        validate="many_to_one",
    )
    lineages, assignments, calibrations = fit_oof_calibrations(
        predictions,
        checkpoint_path=args.checkpoint,
        station_timezone=station.timezone,
        folds=args.folds,
        mc_samples=args.mc_samples,
        fit_calibration=fit_calibration,
    )
    cache = build_forecast_cache(
        hourly_predictions=predictions,
        availability=availability.cycles,
        markets=markets,
        lineages=lineages,
        cycle_lineages=assignments,
        calibrations=calibrations,
        station_id=station.station_id,
        station_timezone=station.timezone,
    )
    manifest = cache.save(
        args.output,
        source_sha256={
            "hourly_predictions": _sha256(args.predictions),
            "availability_manifest": _sha256(
                args.availability_root / "manifest.json"
            ),
            "clob_manifest": _sha256(args.clob_root / "manifest.json"),
        },
    )
    print(
        f"Saved {manifest['row_counts']['snapshots']} OOF snapshots and "
        f"{manifest['row_counts']['probabilities']} market probabilities "
        f"to {args.output}",
        flush=True,
    )
    return 0


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


if __name__ == "__main__":
    raise SystemExit(main())
