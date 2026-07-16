"""Run a strict fixed-epoch TFT -> March MC calibration -> April market test."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.observations import PointInTimeObservationDataset
from backtest.signal_study import (
    run_forecast_signal_validation_study,
    signal_study_id,
)
from backtest.study import StudyInputs, validate_signal_inputs
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.daily_high import fit_partial_day_bias
from model.mc_calibration import McCalibration
from training.audit import audit_workspace
from training.audit_frames import AuditedFrameCache
from training.backtest_forecast_generation import (
    build_forecast_cache,
    mc_calibration_sha256,
    partial_bias_sha256,
)
from training.mc_calibration import fit_mc_calibration
from training.preapril_fixed_epoch import (
    PreAprilFixedEpochPlan,
    local_target_window,
    preapril_fixed_epoch_template,
)
from training.predictions import build_quantile_prediction_frame
from training.trainer import train_workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--mc-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mc_samples <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("sample counts must be positive")
    plan = PreAprilFixedEpochPlan()
    station = STATIONS[args.station]
    _prepare_output(args.output, resume=args.resume)
    print(f"[start] strict pre-April MC output={args.output}", flush=True)

    checkpoint, dataset = _prepare_fixed_epoch_tft(args, plan)
    march = _prediction_stage(
        name="March calibration",
        frame_name="val",
        target_window=plan.calibration,
        output=args.output / "march_calibration_predictions.parquet",
        run_root=args.run_output,
        checkpoint=checkpoint,
        dataset=dataset,
        station_timezone=station.timezone,
        resume=args.resume,
    )
    calibration_root = args.output / "mc_calibration"
    calibration, partial_bias = _calibration_stage(
        march=march,
        checkpoint=checkpoint,
        observations_root=args.observations_root,
        calibration_root=calibration_root,
        station_timezone=station.timezone,
        mc_samples=args.mc_samples,
        resume=args.resume,
    )
    april = _prediction_stage(
        name="April test",
        frame_name="test",
        target_window=plan.evaluation,
        output=args.output / "april_hourly_predictions.parquet",
        run_root=args.run_output,
        checkpoint=checkpoint,
        dataset=dataset,
        station_timezone=station.timezone,
        resume=args.resume,
    )
    cache_root = args.output / "forecast_cache"
    cache = _forecast_cache_stage(
        april=april,
        checkpoint=checkpoint,
        calibration=calibration,
        partial_bias=partial_bias,
        observations_root=args.observations_root,
        availability_root=args.availability_root,
        clob_root=args.clob_root,
        output=cache_root,
        station_id=station.station_id,
        station_timezone=station.timezone,
        mc_samples=args.mc_samples,
        sources=_source_hashes(args),
        resume=args.resume,
    )
    _signal_stage(
        cache_root=cache_root,
        quotes_root=args.quotes_root,
        clob_root=args.clob_root,
        actuals=args.actual_highs,
        output=args.output / "signal_study",
        station_id=station.station_id,
        bootstrap_samples=args.bootstrap_samples,
        resume=args.resume,
    )
    _write_report(
        output=args.output,
        plan=plan,
        checkpoint=checkpoint,
        calibration=calibration,
        partial_bias=partial_bias,
        cache=cache,
        sources=_source_hashes(args),
    )
    print(f"[done] strict pre-April MC output={args.output}", flush=True)
    return 0


def _prepare_fixed_epoch_tft(args: argparse.Namespace, plan: PreAprilFixedEpochPlan) -> tuple[Path, Path]:
    root = args.run_output
    reference = args.source_run / "artifacts" / "ab_experiments" / "hist-error-ab-a-seed42.json"
    data_store = args.source_run / "data_store"
    if not reference.is_file() or not data_store.is_dir():
        raise FileNotFoundError("source run must contain data_store and hist-error reference")
    result_path = root / "validation_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="ascii"))
        checkpoint = _checkpoint_from_training_result(result)
        dataset = checkpoint.parent / "training_dataset.pkl"
        if checkpoint.is_file() and dataset.is_file():
            print(f"[complete] fixed-epoch TFT resume checkpoint={checkpoint.name}", flush=True)
            return checkpoint, dataset
        raise ValueError("fixed-epoch training result is incomplete")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"incomplete training workspace requires manual inspection: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "data_store").symlink_to(data_store.resolve(), target_is_directory=True)
    template = preapril_fixed_epoch_template(
        json.loads(reference.read_text(encoding="ascii")), plan=plan
    )
    template_path = root / "experiment_config.json"
    template_path.write_text(json.dumps(template, indent=2, sort_keys=True), encoding="ascii")
    print(f"[start] fixed-epoch TFT audit workspace={root}", flush=True)
    audit_workspace(
        root,
        args.station,
        {"train": plan.train, "val": plan.calibration, "test": plan.evaluation},
        progress=True,
        progress_every=250,
    )
    print("[complete] fixed-epoch TFT audit", flush=True)
    result = train_workspace(
        root,
        station_id=args.station,
        version="preapril-fixed-epoch-v1",
        experiment_config=template_path,
    )
    checkpoint = _checkpoint_from_training_result(result)
    dataset = checkpoint.parent / "training_dataset.pkl"
    if not checkpoint.is_file() or not dataset.is_file():
        raise RuntimeError("fixed-epoch TFT did not produce checkpoint and dataset")
    print(f"[complete] fixed-epoch TFT checkpoint={checkpoint.name}", flush=True)
    return checkpoint, dataset


def _checkpoint_from_training_result(result: dict[str, object]) -> Path:
    """Accept the in-memory wrapper and the direct on-disk result format."""
    train_result = result.get("train_result", result)
    if not isinstance(train_result, dict):
        raise ValueError("fixed-epoch training result has an invalid train_result")
    checkpoint = train_result.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("fixed-epoch training result has no checkpoint_path")
    return Path(checkpoint)


def _prediction_stage(
    *, name: str, frame_name: str, target_window: tuple[str, str], output: Path,
    run_root: Path, checkpoint: Path, dataset: Path, station_timezone: str,
    resume: bool,
) -> pd.DataFrame:
    if output.is_file():
        if not resume:
            raise FileExistsError(output)
        frame = pd.read_parquet(output)
        print(f"[complete] {name} predictions resume rows={len(frame)}", flush=True)
        return frame
    print(f"[start] {name} predictions output={output}", flush=True)
    cache = AuditedFrameCache(run_root / "artifacts" / "audit_frames", run_root / "data_store")
    audited = cache.load(frame_name, "KLGA", *target_window)
    if audited is None:
        raise RuntimeError(f"missing audited {frame_name} frame")
    frame, _ = audited
    with dataset.open("rb") as handle:
        training_dataset = pickle.load(handle)
    predictions = build_quantile_prediction_frame(
        checkpoint_path=checkpoint,
        training_dataset=training_dataset,
        frame=frame,
        batch_size=256,
    )
    predictions = local_target_window(
        predictions,
        start=target_window[0], end=target_window[1], station_timezone=station_timezone,
    )
    if predictions.empty:
        raise ValueError(f"{name} has no trusted local target cells")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, index=False)
    print(f"[complete] {name} predictions rows={len(predictions)}", flush=True)
    return predictions


def _calibration_stage(
    *, march: pd.DataFrame, checkpoint: Path, observations_root: Path,
    calibration_root: Path, station_timezone: str, mc_samples: int, resume: bool,
):
    metadata_path = calibration_root / "partial_day_bias.json"
    if calibration_root.is_dir() and metadata_path.is_file():
        if not resume:
            raise FileExistsError(calibration_root)
        calibration = McCalibration.load(calibration_root, checkpoint)
        payload = json.loads(metadata_path.read_text(encoding="ascii"))
        from model.daily_high import PartialDayBiasCalibration
        bias = PartialDayBiasCalibration(
            tuple(payload["bias_by_forecast_hours_f"]),
            prior_strength=float(payload["prior_strength"]),
            sample_counts=tuple(payload["sample_counts"]),
        )
        print(f"[complete] March MC calibration resume cycles={calibration.sample_count}", flush=True)
        return calibration, bias
    print(f"[start] March MC calibration output={calibration_root}", flush=True)
    observations = PointInTimeObservationDataset.load(observations_root).observations
    calibration = fit_mc_calibration(
        march, checkpoint, station_timezone=station_timezone, mc_samples=mc_samples
    )
    bias = fit_partial_day_bias(
        hourly_predictions=march, observations=observations, calibration=calibration,
        station_timezone=station_timezone, mc_samples=mc_samples, seed=42,
    )
    calibration.save(calibration_root)
    metadata_path.write_text(json.dumps({
        "bias_by_forecast_hours_f": list(bias.bias_by_forecast_hours_f),
        "prior_strength": bias.prior_strength,
        "sample_counts": list(bias.sample_counts),
        "source": "strict_march_oos_same_checkpoint",
    }, indent=2, sort_keys=True), encoding="ascii")
    print(f"[complete] March MC calibration cycles={calibration.sample_count}", flush=True)
    return calibration, bias


def _forecast_cache_stage(
    *, april: pd.DataFrame, checkpoint: Path, calibration: McCalibration, partial_bias: object,
    observations_root: Path, availability_root: Path, clob_root: Path, output: Path,
    station_id: str, station_timezone: str, mc_samples: int, sources: dict[str, str], resume: bool,
) -> ForecastCache:
    if output.is_dir():
        if not resume:
            raise FileExistsError(output)
        cache = ForecastCache.load(output)
        print(f"[complete] April ForecastCache resume snapshots={len(cache.snapshots)}", flush=True)
        return cache
    print(f"[start] April ForecastCache output={output}", flush=True)
    observations = PointInTimeObservationDataset.load(observations_root).observations
    availability = HrrrAvailabilityDataset.load(availability_root).cycles
    clob = HistoricalClobDataset.load(clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date", "event_end_at"]], on="event_id", how="inner", validate="many_to_one"
    )
    lineage_id = "strict-preapril-fixed-epoch-mc"
    lineages = pd.DataFrame([{
        "lineage_id": lineage_id,
        "checkpoint_sha256": calibration.model_sha256,
        "calibration_sha256": hashlib.sha256((mc_calibration_sha256(calibration) + partial_bias_sha256(partial_bias)).encode("ascii")).hexdigest(),
        "split_role": "model_test",
        "mc_seed": 42,
        "mc_samples": mc_samples,
    }])
    assignments = pd.DataFrame({"init_time": sorted(pd.to_datetime(april["init_time"], utc=True).unique()), "lineage_id": lineage_id})
    cache = build_forecast_cache(
        hourly_predictions=april, observations=observations, availability=availability,
        markets=markets, lineages=lineages, cycle_lineages=assignments,
        calibrations={lineage_id: calibration}, partial_biases={lineage_id: partial_bias},
        station_id=station_id, station_timezone=station_timezone,
        target_start="2026-04-01", target_end="2026-04-30",
    )
    cache.save(output, source_sha256=sources)
    print(f"[complete] April ForecastCache snapshots={len(cache.snapshots)} probabilities={len(cache.probabilities)}", flush=True)
    return cache


def _signal_stage(
    *, cache_root: Path, quotes_root: Path, clob_root: Path, actuals: Path, output: Path,
    station_id: str, bootstrap_samples: int, resume: bool,
) -> None:
    if output.is_dir():
        if not resume:
            raise FileExistsError(output)
        print(f"[complete] April market scoring resume output={output}", flush=True)
        return
    print(f"[start] April market scoring output={output}", flush=True)
    inputs = StudyInputs(cache_root, quotes_root, clob_root, actuals)
    # April is a frozen model-test window. It may be scored for diagnostics,
    # but the trading-study validator must continue to reject it.
    validation = validate_signal_inputs(
        study_id="strict-preapril-april-pending", station_id=station_id,
        start_target="2026-04-01", end_target="2026-04-30", inputs=inputs,
        expected_split_role="model_test",
    )
    study_id = signal_study_id(
        station_id=station_id, start_target="2026-04-01", end_target="2026-04-30",
        input_sha256=validation.input_sha256, ece_bins=10, probability_floor=1e-6,
        bootstrap_samples=bootstrap_samples, seed=42,
    )
    run_forecast_signal_validation_study(
        validation.__class__(**{**validation.__dict__, "study_id": study_id}),
        output=output, bootstrap_samples=bootstrap_samples, seed=42,
    )
    print(f"[complete] April market scoring output={output}", flush=True)


def _prepare_output(output: Path, *, resume: bool) -> None:
    if output.exists() and not resume:
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _source_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "source_reference": args.source_run / "artifacts" / "ab_experiments" / "hist-error-ab-a-seed42.json",
        "observations_manifest": args.observations_root / "manifest.json",
        "availability_manifest": args.availability_root / "manifest.json",
        "clob_manifest": args.clob_root / "manifest.json",
        "quotes_manifest": args.quotes_root / "manifest.json",
        "actuals": args.actual_highs,
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _write_report(*, output: Path, plan: PreAprilFixedEpochPlan, checkpoint: Path,
                  calibration: McCalibration, partial_bias: object, cache: ForecastCache,
                  sources: dict[str, str]) -> None:
    signal = json.loads((output / "signal_study" / "summary.json").read_text(encoding="ascii"))
    report = {
        "schema_version": 1,
        "data_class": "strict_preapril_fixed_epoch_mc",
        "protocol": {
            "tft_train": list(plan.train),
            "tft_fixed_epochs": plan.fixed_epochs,
            "tft_validation_loss_used": False,
            "mc_calibration": list(plan.calibration),
            "evaluation": list(plan.evaluation),
            "original_april_oof_reused": False,
        },
        "checkpoint_sha256": McCalibration.sha256(checkpoint),
        "mc_calibration": {
            "fit_start": calibration.fit_start, "fit_end": calibration.fit_end,
            "sample_count": calibration.sample_count,
            "sha256": mc_calibration_sha256(calibration),
            "partial_bias_sha256": partial_bias_sha256(partial_bias),
        },
        "forecast_cache": {"snapshots": len(cache.snapshots), "probabilities": len(cache.probabilities)},
        "market_summary": signal,
        "source_sha256": sources,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="ascii")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
