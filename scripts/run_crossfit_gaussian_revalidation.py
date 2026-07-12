"""Retrain the future-max Gaussian from historical OOF TFT tracks, then diagnose May."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.daily_high_pmf_evaluation import (
    build_crossfit_gaussian_report,
    score_parallel_forecasts,
    summarize_parallel_scores,
)
from backtest.forecast_cache import ForecastCache
from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.storage.parquet_store import ParquetStore
from training.backtest_forecast_generation import build_gaussian_forecast_cache
from training.daily_high_gaussian import (
    FutureMaxExamples,
    build_future_max_examples,
    build_historical_snapshot_cases,
    fit_gaussian_future_max_model,
    predict_gaussian_examples,
    run_expanding_gaussian_oof,
    summarize_future_max_evaluation,
)
from training.historical_crossfit import historical_six_fold_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-examples-root", required=True, type=Path)
    parser.add_argument("--april-predictions", required=True, type=Path)
    parser.add_argument("--may-predictions", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--april-forecast-root", required=True, type=Path)
    parser.add_argument("--may-forecast-root", required=True, type=Path)
    parser.add_argument("--actuals", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--april-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("epochs and bootstrap samples must be positive")
    for path in (args.april_predictions, args.may_predictions):
        if "202606" in str(path):
            raise ValueError("cross-fit Gaussian revalidation must not consume June")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    station = STATIONS[args.station]
    observations = PointInTimeObservationDataset.load(
        args.observations_root
    ).observations
    historical_examples = _load_historical_examples(args.historical_examples_root)
    april_cache = ForecastCache.load(args.april_forecast_root)
    may_cache = ForecastCache.load(args.may_forecast_root)
    april_predictions = pd.read_parquet(args.april_predictions)
    april_examples = build_future_max_examples(
        hourly_predictions=april_predictions,
        observations=observations,
        snapshots=_target_range(april_cache.snapshots, "2026-04-01", "2026-04-30"),
        station_timezone=station.timezone,
    )
    examples = FutureMaxExamples(
        pd.concat([historical_examples.frame, april_examples.frame], ignore_index=True)
    )
    training_manifest = examples.save(
        output / "training_examples",
        source_sha256={
            "historical_examples_manifest": _sha256(
                args.historical_examples_root / "manifest.json"
            ),
            "april_predictions": _sha256(args.april_predictions),
            "april_forecast_manifest": _sha256(args.april_forecast_root / "manifest.json"),
            "observations_manifest": _sha256(args.observations_root / "manifest.json"),
        },
    )
    training_manifest_sha = _sha256(output / "training_examples" / "manifest.json")
    april_checkpoint_sha = _sha256(args.april_checkpoint)
    historical_oof = run_expanding_gaussian_oof(
        examples,
        tft_checkpoint_sha256=april_checkpoint_sha,
        training_manifest_sha256=training_manifest_sha,
        folds=4, epochs=args.epochs, seed=42,
    )
    historical_oof.save(
        output / "historical_gaussian_oof", split_role="oof_development",
        source_sha256={"training_examples_manifest": training_manifest_sha},
    )
    historical_summary = summarize_future_max_evaluation(historical_oof)
    historical_summary.to_csv(output / "historical_future_max_summary.csv", index=False)
    frozen_model = fit_gaussian_future_max_model(
        examples.frame, feature_columns=examples.feature_columns,
        tft_checkpoint_sha256=april_checkpoint_sha,
        training_manifest_sha256=training_manifest_sha,
        epochs=args.epochs, seed=42,
    )
    frozen_model.save(output / "frozen_model")
    frozen_model_sha = _sha256(output / "frozen_model" / "future_max_gaussian.json")
    may_predictions = pd.read_parquet(args.may_predictions)
    may_examples = build_future_max_examples(
        hourly_predictions=may_predictions,
        observations=observations,
        snapshots=_target_range(may_cache.snapshots, "2026-05-01", "2026-05-31"),
        station_timezone=station.timezone,
    )
    may_examples.save(
        output / "may_examples",
        source_sha256={
            "predictions": _sha256(args.may_predictions),
            "forecast_manifest": _sha256(args.may_forecast_root / "manifest.json"),
            "observations_manifest": _sha256(args.observations_root / "manifest.json"),
            "frozen_model": frozen_model_sha,
        },
    )
    may_evaluation = predict_gaussian_examples(
        frozen_model, may_examples, lineage_label="frozen-may-diagnostic"
    )
    may_evaluation.save(
        output / "may_diagnostic", split_role="model_test",
        source_sha256={"frozen_model": frozen_model_sha},
    )
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date"]], on="event_id", how="inner",
        validate="many_to_one",
    )
    markets["target_date"] = markets["target_date"].astype(str)
    candidate_cache = build_gaussian_forecast_cache(
        evaluation=may_evaluation, source_snapshots=may_cache.snapshots, markets=markets,
        station_id=station.station_id, split_role="model_test",
        tft_checkpoint_sha256=april_checkpoint_sha, artifact_sha256=frozen_model_sha,
    )
    candidate_cache.save(
        output / "may_forecast_cache", source_sha256={"frozen_model": frozen_model_sha}
    )
    actuals = pd.read_csv(args.actuals)
    actuals["target_date"] = actuals["target_date"].astype(str)
    may_scores = score_parallel_forecasts(
        evaluation=may_evaluation, candidate_cache=candidate_cache,
        mc_cache=may_cache, markets=markets, actuals=actuals,
    )
    may_scores.to_parquet(output / "may_parallel_scores.parquet", index=False)
    may_summary = summarize_parallel_scores(may_scores)
    may_summary.to_csv(output / "may_summary.csv", index=False)
    report = build_crossfit_gaussian_report(
        historical_oof=historical_oof.snapshots,
        historical_summary=historical_summary,
        may_summary=may_summary,
        may_scores=may_scores,
        bootstrap_samples=args.bootstrap_samples,
        training_manifest=training_manifest,
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="ascii"
    )
    print(json.dumps(report["gates"], indent=2, sort_keys=True), flush=True)
    return 0


def _combined_observations(
    *,
    historical_data_store: Path,
    station_id: str,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct historical IEM visibility without reading post-April data."""
    raw = ParquetStore(historical_data_store / "obs").read(
        station_id,
        pd.Timestamp("2025-03-29", tz="UTC"),
        pd.Timestamp("2026-04-30", tz="UTC"),
    )
    required = {"valid_time", "raw_tmpc_tenths"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            "historical observation source missing columns: "
            + ", ".join(sorted(missing))
        )
    historical = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(raw["valid_time"], utc=True),
            # Source timestamps are the conservative whole-hour visibility contract.
            "available_at": pd.to_datetime(raw["valid_time"], utc=True),
            "temperature_c": pd.to_numeric(raw["raw_tmpc_tenths"], errors="raise"),
        }
    )
    combined = pd.concat([historical, current], ignore_index=True)
    return (
        combined.sort_values(["valid_time", "available_at"], kind="stable")
        .drop_duplicates("valid_time", keep="last")
        .reset_index(drop=True)
    )


def _load_historical_examples(root: Path) -> FutureMaxExamples:
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    plan = historical_six_fold_plan()
    if manifest.get("fold_ids") != [fold.fold_id for fold in plan.folds]:
        raise ValueError("historical example manifest has another cross-fit plan")
    frames = []
    for fold in plan.folds:
        filename = f"{fold.fold_id}.parquet"
        expected = manifest.get("sha256", {}).get(filename)
        if expected != _sha256(root / filename):
            raise ValueError(f"historical example hash mismatch: {filename}")
        frame = pd.read_parquet(root / filename)
        if frame.empty or not set(FutureMaxExamples(frame).feature_columns).issubset(frame):
            raise ValueError(f"historical example contract invalid: {filename}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["init_time", "target_date"]).any():
        raise ValueError("historical cross-fit examples contain duplicate snapshots")
    return FutureMaxExamples(combined)


def _target_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = frame["target_date"].astype(str)
    return frame[(dates >= start) & (dates <= end)].copy()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
