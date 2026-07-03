"""Generate fixed-checkpoint hourly predictions for replay periods."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from training.predictions import build_quantile_prediction_frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frame", type=Path)
    source.add_argument("--data-store", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start-init")
    parser.add_argument("--end-init")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    predict_fn: Callable = build_quantile_prediction_frame,
) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    frame = (
        pd.read_parquet(args.frame)
        if args.frame is not None
        else _build_inference_frame(args)
    )
    frame = _filter_frame(frame, args.start_init, args.end_init)
    if frame.empty:
        raise ValueError("no inference rows remain in the requested range")
    with args.training_dataset.open("rb") as handle:
        training_dataset = pickle.load(handle)
    predictions = predict_fn(
        checkpoint_path=args.checkpoint,
        training_dataset=training_dataset,
        frame=frame,
        batch_size=args.batch_size,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output, index=False)
    source_sha256 = {
        "checkpoint": _sha256(args.checkpoint),
        "training_dataset": _sha256(args.training_dataset),
    }
    if args.frame is not None:
        source_sha256["frame"] = _sha256(args.frame)
    manifest = {
        "schema_version": 1,
        "data_class": "fixed_checkpoint_hourly_predictions",
        "station_id": args.station,
        "requested_init_range": [args.start_init, args.end_init],
        "row_count": len(predictions),
        "init_count": int(
            pd.to_datetime(predictions["init_time"], utc=True).nunique()
        ),
        "source_sha256": source_sha256,
        "sha256": {args.output.name: _sha256(args.output)},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Saved {manifest['row_count']} rows across "
        f"{manifest['init_count']} init cycles to {args.output}",
        flush=True,
    )
    return 0


def _build_inference_frame(args: argparse.Namespace) -> pd.DataFrame:
    if not args.start_init or not args.end_init:
        raise ValueError(
            "--data-store requires --start-init and --end-init"
        )
    from data.manager import DataManager
    from data.storage.parquet_store import ParquetStore
    from training.dataset import TrainingDatasetBuilder, TrainingDecoderCache

    settings = Settings(data_store_dir=args.data_store)
    manager = DataManager(
        [],
        [],
        ParquetStore(settings.forecast_store_dir),
        ParquetStore(settings.obs_store_dir),
        settings=settings,
    )
    builder = TrainingDatasetBuilder(
        manager,
        TrainingDecoderCache(settings.training_decoder_cache_dir),
    )
    init_times = list(
        pd.date_range(
            _utc(args.start_init),
            _utc(args.end_init) + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
            freq="h",
        )
    )

    def progress(completed: int, total: int, built: int, skipped: int) -> None:
        print(
            f"[inference-frame] {completed}/{total} "
            f"built={built} skipped={skipped}",
            flush=True,
        )

    frame = builder.build_inference_frame(
        STATIONS[args.station],
        init_times,
        split="backtest_inference",
        progress_callback=progress,
        progress_every=100,
    )
    if builder.skipped:
        examples = ", ".join(
            f"{row['init_time']}:{row['reason']}"
            for row in builder.skipped[:5]
        )
        raise RuntimeError(
            f"inference frame skipped {len(builder.skipped)} cycles: {examples}"
        )
    return frame


def _filter_frame(
    frame: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if not start and not end:
        return frame
    if "init_time_str" not in frame:
        raise ValueError("inference frame missing init_time_str")
    init_times = pd.to_datetime(frame["init_time_str"], utc=True)
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= init_times >= _utc(start)
    if end:
        mask &= init_times < _utc(end) + pd.Timedelta(days=1)
    return frame.loc[mask].reset_index(drop=True)


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
