"""Run the training-only conditional cross-hour residual dependence pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.observations import PointInTimeObservationDataset
from backtest.forecast_cache import ForecastCache
from backtest.conditional_residual_evaluation import (
    actual_daily_highs,
    score_parallel_bucket_distributions,
    summarize_conditional_diagnostics,
)
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from data.storage.parquet_store import ParquetStore
from model.conditional_residual import ConditionalPathSampler, ConditionalResidualPathLibrary
from model.daily_high import fit_partial_day_bias
from model.mc import McSampler
from model.mc_calibration import McCalibration
from training.conditional_residual import (
    CONDITIONAL_STATE_FEATURES,
    build_complete_residual_path_library,
    build_conditional_forecast_cache,
    conditional_state,
    fit_conditional_partial_day_bias,
)
from training.historical_crossfit import load_historical_crossfit_predictions
from training.mc_attribution import historical_attribution_folds
from training.mc_calibration import fit_mc_calibration, local_calendar_day_groups


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--historical-data-store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--april-predictions", required=True, type=Path)
    parser.add_argument("--may-predictions", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--april-mc-cache", required=True, type=Path)
    parser.add_argument("--may-mc-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--mc-samples", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mc_samples <= 0:
        raise ValueError("mc-samples must be positive")
    if "202606" in str(args.april_predictions) or "202606" in str(args.may_predictions):
        raise ValueError("conditional dependence pilot must not consume June predictions")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[start] conditional residual pilot output={args.output}", flush=True)
    station = STATIONS[args.station]
    checkpoint_sha = McCalibration.sha256(args.checkpoint)
    frozen_calibration = McCalibration.load(args.calibration_root, args.checkpoint)
    print("[start] loading strict historical cross-fit predictions", flush=True)
    predictions = load_historical_crossfit_predictions(args.crossfit_root)
    raw_observations = _historical_observations(args.historical_data_store, args.station, predictions)
    print(f"[complete] historical inputs prediction_rows={len(predictions)} observation_rows={len(raw_observations)}", flush=True)

    library_root = args.output / "frozen_pre_april_path_library"
    if library_root.exists():
        print("[start] validating frozen residual path library", flush=True)
        library = ConditionalResidualPathLibrary.load(library_root, expected_lineage=_frozen_lineage(args, checkpoint_sha))
    else:
        print("[start] building frozen residual path library through 2026-03-31", flush=True)
        library = build_complete_residual_path_library(
            predictions, frozen_calibration, station_timezone=station.timezone, source_fold_id="historical-crossfit"
        )
        library.save(library_root, lineage=_frozen_lineage(args, checkpoint_sha))
    print(f"[complete] frozen residual path library paths={len(library.paths)}", flush=True)

    print("[start] strict expanding historical OOF scoring", flush=True)
    historical_scores = _historical_expanding_scores(
        args.crossfit_root, args.checkpoint, raw_observations, station.timezone, args.mc_samples
    )
    historical_scores.to_parquet(args.output / "historical_scores.parquet", index=False)
    historical_summary = _summarize(historical_scores)
    historical_summary.to_csv(args.output / "historical_summary.csv", index=False)
    print(f"[complete] historical OOF scoring rows={len(historical_scores)}", flush=True)

    observations = PointInTimeObservationDataset.load(args.observations_root).observations
    availability = HrrrAvailabilityDataset.load(args.availability_root).cycles
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date", "event_end_at"]], on="event_id", how="inner", validate="many_to_one"
    )
    print("[start] fitting frozen conditional partial-day bias", flush=True)
    bias = fit_conditional_partial_day_bias(
        hourly_predictions=predictions, observations=raw_observations,
        calibration=frozen_calibration, library=library, station_timezone=station.timezone,
        mc_samples=min(args.mc_samples, 2_000), seed=42,
    )
    print(f"[complete] frozen conditional partial-day bias samples={sum(bias.sample_counts)}", flush=True)
    source_sha = {
        "path_library": library.manifest_sha256,
        "checkpoint": checkpoint_sha,
        "april_predictions": _sha256(args.april_predictions),
        "may_predictions": _sha256(args.may_predictions),
    }
    for label, path, start, end, role in (
        ("april", args.april_predictions, "2026-04-01", "2026-04-30", "oof_development"),
        ("may", args.may_predictions, "2026-05-01", "2026-05-31", "model_test"),
    ):
        cache_root = args.output / f"{label}_conditional_forecast_cache"
        if cache_root.exists():
            print(f"[complete] {label} candidate cache resume-existing", flush=True)
            continue
        print(f"[start] building {label} conditional ForecastCache", flush=True)
        cache = build_conditional_forecast_cache(
            hourly_predictions=pd.read_parquet(path), observations=observations,
            availability=availability, markets=markets, station_id=station.station_id,
            station_timezone=station.timezone, split_role=role, checkpoint_sha256=checkpoint_sha,
            calibration=frozen_calibration, bias=bias, library=library,
            mc_samples=args.mc_samples, target_start=start, target_end=end,
        )
        cache.save(cache_root, source_sha256=source_sha)
        print(f"[complete] {label} candidate cache snapshots={len(cache.snapshots)} probabilities={len(cache.probabilities)}", flush=True)
    print("[start] paired frozen April/May outcome diagnostics", flush=True)
    paired_reports = _paired_market_outcomes(
        output=args.output, observations=observations, timezone=station.timezone,
        markets=markets, april_candidate=args.output / "april_conditional_forecast_cache",
        may_candidate=args.output / "may_conditional_forecast_cache",
        april_baseline=args.april_mc_cache, may_baseline=args.may_mc_cache,
    )
    print("[complete] paired frozen April/May outcome diagnostics", flush=True)
    report = {
        "schema_version": 1,
        "data_class": "conditional_cross_hour_residual_dependence_pilot",
        "production_change": "none",
        "june_consumed": False,
        "frozen_library_cutoff": "2026-03-31",
        "path_library_sha256": library.manifest_sha256,
        "state_feature_contract": list(CONDITIONAL_STATE_FEATURES),
        "historical": _summary_records(historical_summary),
        "frozen_outcome_diagnostics": paired_reports,
        "artifacts": {"april_cache": "april_conditional_forecast_cache", "may_cache": "may_conditional_forecast_cache"},
        "market_scoring": "Bucket proper scores are paired with current MC; CLOB midpoint scoring remains a separate Forecast Signal Study artifact.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="ascii")
    print(f"[done] conditional residual pilot report={args.output / 'report.json'}", flush=True)
    return 0


def _historical_expanding_scores(crossfit_root: Path, checkpoint: Path, observations: pd.DataFrame, timezone: str, samples: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for boundary in historical_attribution_folds(crossfit_root):
        print(f"[start] historical fold={boundary.evaluation_fold_id} calibration_folds={','.join(boundary.calibration_fold_ids)}", flush=True)
        training = pd.concat([pd.read_parquet(crossfit_root / fold / "predictions.parquet") for fold in boundary.calibration_fold_ids], ignore_index=True)
        evaluation = pd.read_parquet(crossfit_root / boundary.evaluation_fold_id / "predictions.parquet")
        calibration = fit_mc_calibration(training, checkpoint, station_timezone=timezone, mc_samples=min(samples, 2_000))
        library = build_complete_residual_path_library(training, calibration, station_timezone=timezone, source_fold_id="|".join(boundary.calibration_fold_ids))
        bias = fit_conditional_partial_day_bias(hourly_predictions=training, observations=observations, calibration=calibration, library=library, station_timezone=timezone, mc_samples=min(samples, 2_000), seed=42)
        static_bias = fit_partial_day_bias(hourly_predictions=training, observations=observations, calibration=calibration, station_timezone=timezone, mc_samples=min(samples, 2_000), seed=42)
        for group_index, group in enumerate(local_calendar_day_groups(evaluation, timezone)):
            ordered = group.sort_values("fxx")
            raw_sigma = (ordered["p90"].to_numpy(float) - ordered["p10"].to_numpy(float)) / (2.0 * 1.2815515655446004)
            indices = [int(value) - 1 for value in ordered["fxx"]]
            sigma = calibration.scale_sigma(raw_sigma, hour_indices=indices)
            mu = ordered["base_temp_c"].to_numpy(float) + ordered["p50"].to_numpy(float)
            actual = float((ordered["base_temp_c"] + ordered["target"]).max() * 1.8 + 32.0)
            static = McSampler(n_samp=samples, rng_seed=42 + group_index)
            static.set_calibration(calibration)
            static_samples = static.sample_daily_high(mu, raw_sigma, max_bias=static_bias.for_forecast_hours(len(ordered)), hour_indices=indices)
            conditional = ConditionalPathSampler(library, n_samp=samples, rng_seed=42 + group_index)
            try:
                conditional_samples = conditional.sample_daily_high(mu, sigma, indices, state=conditional_state(ordered, target_date=str(ordered["target_date"].iloc[0]), station_timezone=timezone, observed_hour_count=0, observed_high_f=None), max_bias=bias.for_forecast_hours(len(ordered)))
            except ValueError:
                continue
            rows.extend([
                _score("static_mc", boundary.evaluation_fold_id, ordered, static_samples, actual, None),
                _score("conditional_path_mc", boundary.evaluation_fold_id, ordered, conditional_samples, actual, conditional.last_diagnostics),
            ])
        print(f"[complete] historical fold={boundary.evaluation_fold_id} cumulative_rows={len(rows)}", flush=True)
    return pd.DataFrame(rows)


def _paired_market_outcomes(
    *, output: Path, observations: pd.DataFrame, timezone: str, markets: pd.DataFrame,
    april_candidate: Path, may_candidate: Path, april_baseline: Path, may_baseline: Path,
) -> dict[str, object]:
    periods = (
        ("april", april_candidate, april_baseline, "2026-04-01", "2026-04-30"),
        ("may_early", may_candidate, may_baseline, "2026-05-01", "2026-05-16"),
        ("may_late", may_candidate, may_baseline, "2026-05-19", "2026-05-31"),
    )
    reports: dict[str, object] = {}
    for label, candidate_root, baseline_root, start, end in periods:
        actuals = actual_daily_highs(observations, station_timezone=timezone, start_target=start, end_target=end)
        scores = score_parallel_bucket_distributions(
            candidate=ForecastCache.load(candidate_root), baseline=ForecastCache.load(baseline_root),
            markets=markets, actuals=actuals,
        )
        scores = scores[scores["target_date"].between(start, end)].copy()
        scores.to_parquet(output / f"{label}_paired_bucket_scores.parquet", index=False)
        reports[label] = summarize_conditional_diagnostics(scores)
    return reports


def _score(variant: str, fold_id: str, group: pd.DataFrame, samples: np.ndarray, actual: float, diagnostics: object) -> dict[str, object]:
    integer = np.floor(samples + 0.5).astype(int)
    support = np.arange(min(integer.min(), int(np.floor(actual + 0.5))), max(integer.max(), int(np.floor(actual + 0.5))) + 1)
    cdf = np.array([(integer <= value).mean() for value in support])
    truth = (support >= int(np.floor(actual + 0.5))).astype(float)
    result = {
        "variant_id": variant, "fold_id": fold_id, "init_time": pd.Timestamp(group["init_time"].iloc[0]),
        "target_date": str(group["target_date"].iloc[0]), "actual_high_f": actual,
        "p10_f": float(np.quantile(samples, .1)), "p50_f": float(np.quantile(samples, .5)), "p90_f": float(np.quantile(samples, .9)),
        "rps": float(np.mean((cdf - truth) ** 2)), "log_score": float(-np.log(max((integer == int(np.floor(actual + .5))).mean(), 1e-6))),
        "true_integer_probability": float((integer == int(np.floor(actual + .5))).mean()),
        "coverage_type": "forecast_only", "forecast_hour_count": len(group),
    }
    if diagnostics is not None:
        result.update({"nearest_path_distance": diagnostics.nearest_distance, "path_effective_sample_size": diagnostics.effective_sample_size, "path_ood": diagnostics.is_ood})
    return result


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        raise ValueError("conditional historical evaluation produced no scores")
    return scores.groupby("variant_id", as_index=False).agg(
        snapshots=("rps", "size"), event_days=("target_date", "nunique"), rps=("rps", "mean"),
        log_score=("log_score", "mean"), true_integer_probability=("true_integer_probability", "mean"),
        p50_mae_f=("p50_f", lambda values: float(np.mean(np.abs(values - scores.loc[values.index, "actual_high_f"])))),
        p10_p90_coverage=("actual_high_f", lambda values: float(np.mean((values >= scores.loc[values.index, "p10_f"]) & (values <= scores.loc[values.index, "p90_f"])))),
        p10_p90_width_f=("p90_f", lambda values: float(np.mean(values - scores.loc[values.index, "p10_f"]))),
    )


def _historical_observations(root: Path, station: str, predictions: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(predictions["valid_time"], utc=True).min() - pd.Timedelta(days=2)
    end = pd.to_datetime(predictions["valid_time"], utc=True).max() + pd.Timedelta(days=2)
    raw = ParquetStore(root / "obs").read(station, start, end)
    return pd.DataFrame({"valid_time": pd.to_datetime(raw["valid_time"], utc=True), "temperature_c": pd.to_numeric(raw["raw_tmpc_tenths"], errors="raise")}).drop_duplicates("valid_time", keep="last")


def _frozen_lineage(args: argparse.Namespace, checkpoint_sha: str) -> dict[str, str]:
    return {"crossfit_manifest": _sha256(args.crossfit_root / "manifest.json"), "checkpoint": checkpoint_sha, "cutoff": "2026-03-31"}


def _summary_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
