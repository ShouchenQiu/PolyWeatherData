"""Build model-test or holdout forecast cache from one fixed checkpoint."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.mc_calibration import McCalibration
from model.daily_high import fit_partial_day_bias
from training.backtest_forecast_generation import partial_bias_sha256
from training.backtest_forecast_generation import build_forecast_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument(
        "--bias-predictions",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split-role",
        required=True,
        choices=["model_test", "post_test_holdout"],
    )
    parser.add_argument("--start-init", required=True)
    parser.add_argument("--end-init", required=True)
    parser.add_argument("--target-start", required=True)
    parser.add_argument("--target-end", required=True)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--mc-samples", type=int, default=2_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mc_samples <= 0:
        raise ValueError("--mc-samples must be positive")
    station = STATIONS[args.station]
    predictions = pd.concat(
        [pd.read_parquet(path) for path in args.predictions],
        ignore_index=True,
    )
    predictions["init_time"] = pd.to_datetime(
        predictions["init_time"],
        utc=True,
    )
    start = _utc(args.start_init)
    end = _utc(args.end_init) + pd.Timedelta(days=1)
    predictions = predictions[
        (predictions["init_time"] >= start)
        & (predictions["init_time"] < end)
    ].copy()
    if predictions.empty:
        raise ValueError("no predictions remain in requested init range")
    calibration = McCalibration.load(
        args.calibration_root,
        args.checkpoint,
    )
    observations = PointInTimeObservationDataset.load(
        args.observations_root
    )
    bias_predictions = pd.concat(
        [pd.read_parquet(path) for path in args.bias_predictions],
        ignore_index=True,
    )
    partial_bias = fit_partial_day_bias(
        hourly_predictions=bias_predictions,
        observations=observations.observations,
        calibration=calibration,
        station_timezone=station.timezone,
        mc_samples=args.mc_samples,
        seed=42,
    )
    lineage_id = f"fixed-{args.split_role.replace('_', '-')}"
    lineages = pd.DataFrame(
        [
            {
                "lineage_id": lineage_id,
                "checkpoint_sha256": calibration.model_sha256,
                "calibration_sha256": _calibration_sha256(
                    args.calibration_root,
                    partial_bias,
                ),
                "split_role": args.split_role,
                "mc_seed": 42,
                "mc_samples": args.mc_samples,
            }
        ]
    )
    cycle_lineages = pd.DataFrame(
        {
            "init_time": sorted(predictions["init_time"].unique()),
            "lineage_id": lineage_id,
        }
    )
    availability = HrrrAvailabilityDataset.load(args.availability_root)
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date", "event_end_at"]],
        on="event_id",
        how="inner",
        validate="many_to_one",
    )
    cache = build_forecast_cache(
        hourly_predictions=predictions,
        observations=observations.observations,
        availability=availability.cycles,
        markets=markets,
        lineages=lineages,
        cycle_lineages=cycle_lineages,
        calibrations={lineage_id: calibration},
        partial_biases={lineage_id: partial_bias},
        station_id=station.station_id,
        station_timezone=station.timezone,
        target_start=args.target_start,
        target_end=args.target_end,
    )
    source_hashes = {
        f"predictions_{index}": _sha256(path)
        for index, path in enumerate(args.predictions)
    }
    source_hashes.update(
        {
            f"bias_predictions_{index}": _sha256(path)
            for index, path in enumerate(args.bias_predictions)
        }
    )
    source_hashes.update(
        {
            "availability_manifest": _sha256(
                args.availability_root / "manifest.json"
            ),
            "clob_manifest": _sha256(args.clob_root / "manifest.json"),
            "checkpoint": _sha256(args.checkpoint),
            "mc_calibration_bundle": _calibration_files_sha256(
                args.calibration_root
            ),
            "partial_bias_bundle": partial_bias_sha256(partial_bias),
            "observations_manifest": _sha256(
                args.observations_root / "manifest.json"
            ),
        }
    )
    manifest = cache.save(args.output, source_sha256=source_hashes)
    print(
        f"Saved {manifest['row_counts']['snapshots']} "
        f"{args.split_role} snapshots and "
        f"{manifest['row_counts']['probabilities']} probabilities "
        f"to {args.output}",
        flush=True,
    )
    return 0


def _calibration_sha256(root: Path, partial_bias: object) -> str:
    hasher = hashlib.sha256()
    hasher.update(_calibration_files_sha256(root).encode("ascii"))
    hasher.update(partial_bias_sha256(partial_bias).encode("ascii"))
    return hasher.hexdigest()


def _calibration_files_sha256(root: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("mc_calibration.json", "mc_calibration.npz"):
        hasher.update(name.encode("ascii"))
        hasher.update((root / name).read_bytes())
    return hasher.hexdigest()


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
