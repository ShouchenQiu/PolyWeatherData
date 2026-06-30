"""Evaluate one standalone A/B validation checkpoint and optionally fit MC."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from training.audit_frames import AuditedFrameCache
from training.config import ExperimentTemplate
from training.mc_calibration import (
    cross_validate_mc_calibration,
    fit_mc_calibration,
)
from training.predictions import build_quantile_prediction_frame
from training.selection import compute_validation_evidence
from training.workspace import TrainingWorkspace


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--fit-mc", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=2_000)
    args = parser.parse_args()

    workspace = TrainingWorkspace(args.run_dir)
    template = ExperimentTemplate.from_json(args.experiment_config)
    version_dir = workspace.checkpoints_dir / args.version
    checkpoints = sorted(version_dir.glob("tft_best*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected exactly one best checkpoint in {version_dir}, "
            f"found {len(checkpoints)}"
        )
    dataset_path = version_dir / "training_dataset.pkl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"training_dataset.pkl not found at {dataset_path}"
        )
    with dataset_path.open("rb") as handle:
        training_dataset = pickle.load(handle)
    val_start, val_end = template.splits["val"]
    cached = AuditedFrameCache(
        workspace.audit_frames_dir,
        workspace.data_store_dir,
    ).load("val", template.station, val_start, val_end)
    if cached is None:
        raise RuntimeError("audited validation frame is missing or stale")
    prediction_frame = build_quantile_prediction_frame(
        checkpoint_path=checkpoints[0],
        training_dataset=training_dataset,
        frame=cached[0],
        batch_size=template.trainer.batch_size,
    )
    prediction_path = version_dir / "validation_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    result: dict[str, object] = {
        "version": args.version,
        "checkpoint_path": str(checkpoints[0]),
        "prediction_path": str(prediction_path),
        "raw_validation": compute_validation_evidence(prediction_frame),
    }
    if args.fit_mc:
        result["cross_validated_mc"] = cross_validate_mc_calibration(
            prediction_frame,
            checkpoints[0],
            folds=4,
            mc_samples=args.mc_samples,
        )
        calibration = fit_mc_calibration(
            prediction_frame,
            checkpoints[0],
            mc_samples=args.mc_samples,
        )
        calibration.save(version_dir)
        result["mc_calibration"] = {
            "max_bias_f": calibration.max_bias_f,
            "sigma_scales": calibration.sigma_scales.tolist(),
            "sample_count": calibration.sample_count,
        }
    output_path = version_dir / "validation_evidence.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
