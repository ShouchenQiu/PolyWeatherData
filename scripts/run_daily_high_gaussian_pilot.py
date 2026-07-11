"""Build the April OOF / frozen-May future-maximum Gaussian pilot artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.daily_high_pmf_evaluation import (
    build_gaussian_pilot_report,
    score_parallel_forecasts,
    summarize_parallel_scores,
)
from backtest.forecast_cache import ForecastCache
from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from training.backtest_forecast_generation import build_gaussian_forecast_cache
from training.daily_high_gaussian import (
    build_future_max_examples,
    fit_gaussian_future_max_model,
    predict_gaussian_examples,
    run_expanding_gaussian_oof,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--april-predictions", required=True, type=Path)
    parser.add_argument("--may-predictions", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--april-forecast-root", required=True, type=Path)
    parser.add_argument("--may-forecast-root", required=True, type=Path)
    parser.add_argument("--actuals", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
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
            raise ValueError("Gaussian pilot must not consume June predictions")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    station = STATIONS[args.station]
    checkpoint_sha = _sha256(args.checkpoint)
    observations = PointInTimeObservationDataset.load(args.observations_root).observations
    april_cache = ForecastCache.load(args.april_forecast_root)
    may_cache = ForecastCache.load(args.may_forecast_root)
    april_predictions = pd.read_parquet(args.april_predictions)
    may_predictions = pd.read_parquet(args.may_predictions)
    actuals = pd.read_csv(args.actuals)
    actuals["target_date"] = actuals["target_date"].astype(str)
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date"]], on="event_id", how="inner",
        validate="many_to_one",
    )
    markets["target_date"] = markets["target_date"].astype(str)

    april_examples = build_future_max_examples(
        hourly_predictions=april_predictions,
        observations=observations,
        snapshots=_target_range(april_cache.snapshots, "2026-04-01", "2026-04-30"),
        station_timezone=station.timezone,
    )
    april_manifest = april_examples.save(
        output / "april_examples",
        source_sha256={
            "predictions": _sha256(args.april_predictions),
            "forecast_manifest": _sha256(args.april_forecast_root / "manifest.json"),
            "observations_manifest": _sha256(args.observations_root / "manifest.json"),
        },
    )
    april_manifest_sha = _sha256(output / "april_examples" / "manifest.json")
    april_oof = run_expanding_gaussian_oof(
        april_examples, tft_checkpoint_sha256=checkpoint_sha,
        training_manifest_sha256=april_manifest_sha, folds=4,
        epochs=args.epochs, seed=42,
    )
    april_oof.save(
        output / "april_oof", split_role="oof_development",
        source_sha256={"examples_manifest": april_manifest_sha},
    )
    april_oof_sha = _sha256(output / "april_oof" / "manifest.json")
    april_candidate_cache = build_gaussian_forecast_cache(
        evaluation=april_oof, source_snapshots=april_cache.snapshots, markets=markets,
        station_id=station.station_id, split_role="oof_development",
        tft_checkpoint_sha256=checkpoint_sha, artifact_sha256=april_oof_sha,
    )
    april_candidate_cache.save(
        output / "april_oof_forecast_cache",
        source_sha256={"gaussian_evaluation": april_oof_sha},
    )

    frozen_model = fit_gaussian_future_max_model(
        april_examples.frame, feature_columns=april_examples.feature_columns,
        tft_checkpoint_sha256=checkpoint_sha,
        training_manifest_sha256=april_manifest_sha,
        epochs=args.epochs, seed=42,
    )
    frozen_model.save(output / "frozen_model")
    frozen_model_sha = _sha256(output / "frozen_model" / "future_max_gaussian.json")

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
    may_candidate_cache = build_gaussian_forecast_cache(
        evaluation=may_evaluation, source_snapshots=may_cache.snapshots, markets=markets,
        station_id=station.station_id, split_role="model_test",
        tft_checkpoint_sha256=checkpoint_sha, artifact_sha256=frozen_model_sha,
    )
    may_candidate_cache.save(
        output / "may_forecast_cache",
        source_sha256={"frozen_model": frozen_model_sha},
    )

    april_scores = score_parallel_forecasts(
        evaluation=april_oof, candidate_cache=april_candidate_cache,
        mc_cache=april_cache, markets=markets, actuals=actuals,
    )
    may_scores = score_parallel_forecasts(
        evaluation=may_evaluation, candidate_cache=may_candidate_cache,
        mc_cache=may_cache, markets=markets, actuals=actuals,
    )
    april_scores.to_parquet(output / "april_parallel_scores.parquet", index=False)
    may_scores.to_parquet(output / "may_parallel_scores.parquet", index=False)
    april_summary = summarize_parallel_scores(april_scores)
    may_summary = summarize_parallel_scores(may_scores)
    april_summary.to_csv(output / "april_summary.csv", index=False)
    may_summary.to_csv(output / "may_summary.csv", index=False)
    report = build_gaussian_pilot_report(
        april_oof=april_oof.snapshots, april_summary=april_summary,
        may_summary=may_summary, april_scores=april_scores, may_scores=may_scores,
        bootstrap_samples=args.bootstrap_samples, april_manifest=april_manifest,
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="ascii"
    )
    print(json.dumps(report["gates"], indent=2, sort_keys=True), flush=True)
    return 0


def _target_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = frame["target_date"].astype(str)
    return frame[(dates >= start) & (dates <= end)].copy()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
