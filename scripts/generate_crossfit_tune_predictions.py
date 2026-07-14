"""Generate checkpoint-aligned tune predictions for MC calibration evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from training.historical_crossfit import historical_six_fold_plan
from training.predictions import build_quantile_prediction_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"immutable tune-prediction output exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = historical_six_fold_plan()
    root_manifest = {
        "schema_version": 1,
        "data_class": "crossfit_checkpoint_aligned_tune_predictions",
        "crossfit_manifest_sha256": _sha256(args.crossfit_root / "manifest.json"),
        "fold_ids": [fold.fold_id for fold in plan.folds],
    }
    _write_or_validate_manifest(args.output / "manifest.json", root_manifest)
    print(f"[start] crossfit tune predictions output={args.output}", flush=True)
    for index, fold in enumerate(plan.folds, start=1):
        fold_root = args.crossfit_root / fold.fold_id
        crossfit = json.loads((fold_root / "crossfit_manifest.json").read_text(encoding="ascii"))
        artifacts = crossfit["artifacts"]
        checkpoint = fold_root / artifacts["checkpoint"]
        dataset = fold_root / artifacts["training_dataset"]
        frame = fold_root / "artifacts" / "audit_frames" / "val.parquet"
        output = args.output / fold.fold_id / "tune_predictions.parquet"
        manifest_path = output.with_suffix(".parquet.manifest.json")
        expected = {
            "schema_version": 1,
            "fold": {"fold_id": fold.fold_id, "tune": list(fold.tune)},
            "source_sha256": {
                "checkpoint": _sha256(checkpoint),
                "training_dataset": _sha256(dataset),
                "audit_frame": _sha256(frame),
            },
        }
        if args.resume and output.is_file() and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("fold") == expected["fold"] and existing.get("source_sha256") == expected["source_sha256"] and existing.get("sha256", {}).get(output.name) == _sha256(output):
                print(f"[complete] tune fold={fold.fold_id} resume rows={existing['row_count']}", flush=True)
                continue
        if output.exists() or manifest_path.exists():
            raise FileExistsError(f"invalid or stale tune prediction artifact: {output}")
        print(f"[progress] tune fold={fold.fold_id} stage=inference {index}/{len(plan.folds)}", flush=True)
        with dataset.open("rb") as handle:
            training_dataset = pickle.load(handle)
        predictions = build_quantile_prediction_frame(
            checkpoint_path=checkpoint,
            training_dataset=training_dataset,
            frame=pd.read_parquet(frame),
            batch_size=args.batch_size,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        expected["row_count"] = len(predictions)
        expected["init_count"] = int(pd.to_datetime(predictions["init_time"], utc=True).nunique())
        expected["sha256"] = {output.name: _sha256(output)}
        manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[complete] tune fold={fold.fold_id} rows={len(predictions)}", flush=True)
    print(f"[done] crossfit tune predictions output={args.output}", flush=True)
    return 0


def _write_or_validate_manifest(path: Path, payload: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("tune-prediction root manifest does not match crossfit input")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
