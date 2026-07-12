"""Build one bounded historical future-max example fold per process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS
from scripts.run_crossfit_gaussian_revalidation import _combined_observations
from training.daily_high_gaussian import (
    build_future_max_examples,
    build_historical_snapshot_cases,
)
from training.historical_crossfit import (
    historical_six_fold_plan,
    load_historical_crossfit_predictions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--historical-data-store", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", required=True, choices=[fold.fold_id for fold in historical_six_fold_plan().folds])
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = historical_six_fold_plan()
    fold = next(item for item in plan.folds if item.fold_id == args.fold)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    payload = output / f"{fold.fold_id}.parquet"
    if payload.exists():
        raise FileExistsError(payload)
    station = STATIONS[args.station]
    predictions = load_historical_crossfit_predictions(args.crossfit_root)
    current = PointInTimeObservationDataset.load(args.observations_root).observations
    observations = _combined_observations(
        historical_data_store=args.historical_data_store,
        station_id=station.station_id,
        current=current,
    )
    fold_predictions = predictions[predictions["fold_id"] == fold.fold_id]
    cases = build_historical_snapshot_cases(
        fold_predictions, station_timezone=station.timezone,
        target_start=fold.predict[0], target_end=fold.predict[1],
    )
    examples = build_future_max_examples(
        hourly_predictions=fold_predictions, observations=observations,
        snapshots=cases, station_timezone=station.timezone,
    )
    if examples.frame.empty:
        raise RuntimeError(f"no complete historical examples for {fold.fold_id}")
    examples.frame.to_parquet(payload, index=False)
    _write_manifest(
        output,
        plan,
        source_sha256={
            "crossfit_root_manifest": _sha256(args.crossfit_root / "manifest.json"),
            "current_observations_manifest": _sha256(
                args.observations_root / "manifest.json"
            ),
            **{
                f"{item.fold_id}_crossfit_manifest": _sha256(
                    args.crossfit_root / item.fold_id / "crossfit_manifest.json"
                )
                for item in plan.folds
            },
        },
    )
    print(f"{fold.fold_id}: {len(examples.frame)} examples", flush=True)
    return 0


def _write_manifest(root: Path, plan, *, source_sha256: dict[str, str]) -> None:
    filenames = [f"{fold.fold_id}.parquet" for fold in plan.folds]
    completed = [name for name in filenames if (root / name).is_file()]
    payload = {
        "schema_version": 1,
        "data_class": "historical_crossfit_future_max_examples",
        "fold_ids": [fold.fold_id for fold in plan.folds],
        "completed_folds": [name.removesuffix(".parquet") for name in completed],
        "source_sha256": dict(sorted(source_sha256.items())),
        "sha256": {name: _sha256(root / name) for name in completed},
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="ascii"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
