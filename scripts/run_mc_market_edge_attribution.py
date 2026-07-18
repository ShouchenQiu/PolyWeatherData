"""PIT MC oracle ceiling and frozen market-edge attribution.

This is intentionally separate from ``run_prequential_mc_optuna``: its
objective is paired normalized-synthetic-midpoint edge on validated market
vectors, not full-day temperature RPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import optuna
import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.observations import PointInTimeObservationDataset
from backtest.signal_study import build_signal_panel, load_signal_study_artifact
from backtest.study import StudyInputs, validate_study_inputs
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.daily_high import fit_partial_day_bias
from training.backtest_forecast_generation import build_forecast_cache, mc_calibration_sha256, partial_bias_sha256
from training.historical_crossfit import historical_six_fold_plan
from training.historical_market_panel import _fold_checkpoint, _load_checkpoint_aligned_tune_predictions, _right_closed_target_dates, _target_dates
from training.mc_calibration import fit_mc_calibration
from training.mc_market_edge import rebuild_market_candidate_probabilities, score_by_strata, score_market_edge_panel, selected_fold_ids
from training.mc_prequential_optuna import apply_mc_tune_parameters, enqueue_identity_mc_trial, identity_mc_tune_parameters, mc_tune_component_subsets, shapley_rps_attribution, suggest_mc_tune_parameters


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--tune-predictions-root", required=True, type=Path)
    parser.add_argument("--static-cache-root", required=True, type=Path)
    parser.add_argument("--static-signal-root", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("oracle_market_edge", "frozen_market_edge"), required=True)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--calibration-mc-samples", type=int, default=2000)
    parser.add_argument("--trial-mc-samples", type=int, default=256)
    parser.add_argument("--evaluation-mc-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(args.trials, args.calibration_mc_samples, args.trial_mc_samples, args.evaluation_mc_samples, args.bootstrap_samples) <= 0:
        raise ValueError("all trial, sample and bootstrap counts must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_or_validate_contract(args)
    print(f"[start] MC market edge mode={args.mode} output={args.output}", flush=True)
    station = STATIONS[args.station]
    signal = load_signal_study_artifact(args.static_signal_root)
    static_cache = ForecastCache.load(args.static_cache_root)
    eval_panel_by_fold = _evaluation_panels(signal.panel, static_cache.snapshots)
    observations = PointInTimeObservationDataset.load(args.observations_root).observations
    availability = HrrrAvailabilityDataset.load(args.availability_root).cycles
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(clob.events[["event_id", "target_date", "event_end_at"]], on="event_id", how="inner", validate="many_to_one")
    all_folds = historical_six_fold_plan().folds
    folds = [fold for fold in all_folds if fold.fold_id in selected_fold_ids(args.mode, tuple(item.fold_id for item in all_folds))]
    all_scores: list[pd.DataFrame] = []
    fold_report: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        eval_panel = eval_panel_by_fold.get(fold.fold_id, pd.DataFrame())
        if eval_panel.empty:
            raise ValueError(f"static signal panel has no market vectors for {fold.fold_id}")
        evaluation = _evaluation_predictions(args.crossfit_root, fold, station.timezone)
        checkpoint = _fold_checkpoint(args.crossfit_root, fold.fold_id)
        tuning = _load_checkpoint_aligned_tune_predictions(root=args.crossfit_root, tune_root=args.tune_predictions_root, fold_id=fold.fold_id, checkpoint=checkpoint, station_timezone=station.timezone)
        base_calibration = fit_mc_calibration(tuning, checkpoint, station_timezone=station.timezone, mc_samples=args.calibration_mc_samples)
        base_bias = fit_partial_day_bias(hourly_predictions=tuning, observations=observations, calibration=base_calibration, station_timezone=station.timezone, mc_samples=args.calibration_mc_samples, seed=42)
        if args.mode == "oracle_market_edge":
            selection_panel, selection_predictions, selection_snapshots = eval_panel, evaluation, _snapshots_for_panel(static_cache.snapshots, eval_panel)
        else:
            selection_panel, selection_snapshots = _build_or_load_tune_panel(args, fold=fold, tuning=tuning, calibration=base_calibration, bias=base_bias, observations=observations, availability=availability, markets=markets, station_id=station.station_id, timezone=station.timezone)
            selection_predictions = tuning
            if selection_panel.empty:
                raise ValueError(f"frozen selection has no complete market vectors for {fold.fold_id}")
        vector_count = selection_panel[["event_id", "timestamp"]].drop_duplicates().shape[0]
        print(f"[start] fold={fold.fold_id} selection={args.mode} vectors={vector_count}", flush=True)
        subset_scores: dict[str, float] = {}
        for subset_index, components in enumerate(mc_tune_component_subsets()):
            result = _load_subset_result(args.output, fold.fold_id, components.subset_id) if args.resume else None
            if result is None:
                result = _run_subset(args=args, fold_id=fold.fold_id, components=components, selection_panel=selection_panel, selection_predictions=selection_predictions, selection_snapshots=selection_snapshots, evaluation_panel=eval_panel, evaluation_predictions=evaluation, evaluation_snapshots=_snapshots_for_panel(static_cache.snapshots, eval_panel), observations=observations, base_calibration=base_calibration, base_bias=base_bias, timezone=station.timezone, seed=42 + fold_index * 100 + subset_index)
                _save_best(args.output, fold.fold_id, components.subset_id, result)
            all_scores.append(result["scores"])
            subset_scores[components.subset_id] = float(result["summary"]["event_day_equal"]["normalized_rps_delta"])
        shapley = shapley_rps_attribution(subset_scores)
        fold_report.append({"fold_id": fold.fold_id, "subset_market_edge": subset_scores, "shapley_market_edge_improvement": shapley})
        print(f"[complete] fold={fold.fold_id} subsets=8", flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(args.output / "candidate_scores.parquet", index=False)
    report = _report(args, scores, fold_report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="ascii")
    print(f"[done] MC market edge mode={args.mode} scores={len(scores)} output={args.output}", flush=True)
    return 0


def _run_subset(*, args: argparse.Namespace, fold_id: str, components: object, selection_panel: pd.DataFrame, selection_predictions: pd.DataFrame, selection_snapshots: pd.DataFrame, evaluation_panel: pd.DataFrame, evaluation_predictions: pd.DataFrame, evaluation_snapshots: pd.DataFrame, observations: pd.DataFrame, base_calibration: object, base_bias: object, timezone: str, seed: int) -> dict[str, object]:
    subset_id = components.subset_id
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    enqueue_identity_mc_trial(study, components)
    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_mc_tune_parameters(trial, components)
        calibration, bias = apply_mc_tune_parameters(base_calibration, base_bias, components=components, parameters=parameters)
        probabilities = rebuild_market_candidate_probabilities(selection_panel, selection_snapshots, hourly_predictions=selection_predictions, observations=observations, calibration=calibration, bias=bias, station_timezone=timezone, mc_samples=args.trial_mc_samples, seed=42)
        _, summary = score_market_edge_panel(selection_panel, candidate_probability=probabilities, bootstrap_samples=1)
        return float(summary["event_day_equal"]["normalized_rps_delta"])
    if subset_id == "none":
        parameters = identity_mc_tune_parameters(components)
    else:
        study.optimize(objective, n_trials=args.trials)
        parameters = dict(study.best_params)
    calibration, bias = apply_mc_tune_parameters(base_calibration, base_bias, components=components, parameters=parameters)
    probabilities = rebuild_market_candidate_probabilities(evaluation_panel, evaluation_snapshots, hourly_predictions=evaluation_predictions, observations=observations, calibration=calibration, bias=bias, station_timezone=timezone, mc_samples=args.evaluation_mc_samples, seed=42)
    scores, summary = score_market_edge_panel(evaluation_panel, candidate_probability=probabilities, bootstrap_samples=args.bootstrap_samples, seed=seed)
    scores["fold_id"], scores["subset_id"], scores["selection_mode"] = fold_id, subset_id, args.mode
    scores["candidate_lineage"] = f"{args.mode}:{fold_id}:{subset_id}"
    return {"scores": scores, "summary": summary, "parameters": parameters, "identity_won": parameters == identity_mc_tune_parameters(components), "selection_objective": float(study.best_value) if subset_id != "none" else None, "evaluation_market_edge": float(summary["event_day_equal"]["normalized_rps_delta"]), "selection_mode": args.mode, "calibration_sha256": mc_calibration_sha256(calibration), "partial_bias_sha256": partial_bias_sha256(bias)}


def _evaluation_panels(panel: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mapping = snapshots[["snapshot_id", "lineage_id", "coverage_type", "forecast_hour_count"]].copy()
    mapping["fold_id"] = mapping["lineage_id"].astype(str).str.extract(r"(fold-\d+)$", expand=False)
    merged = panel.merge(
        mapping[["snapshot_id", "fold_id", "coverage_type", "forecast_hour_count"]],
        on="snapshot_id",
        how="inner",
        validate="many_to_one",
    )
    if len(merged) != len(panel) or merged["fold_id"].isna().any():
        raise ValueError("static signal panel snapshot lineage cannot be mapped to a fold")
    plan = {fold.fold_id: fold for fold in historical_six_fold_plan().folds}
    invalid = merged.loc[
        ~merged.apply(
            lambda row: plan[str(row.fold_id)].predict[0]
            <= str(row.target_date)
            <= plan[str(row.fold_id)].predict[1],
            axis=1,
        ),
        ["snapshot_id", "fold_id", "target_date"],
    ].drop_duplicates()
    if not invalid.empty:
        first = invalid.iloc[0]
        raise ValueError(
            "static market panel contains cross-fit target tail "
            f"snapshot={first.snapshot_id} fold={first.fold_id} "
            f"target_date={first.target_date}; rebuild it with strict target filtering"
        )
    return {str(key): value.drop(columns="fold_id").copy() for key, value in merged.groupby("fold_id", sort=True)}


def _snapshots_for_panel(snapshots: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    needed = set(panel["snapshot_id"].astype(str))
    result = snapshots[snapshots["snapshot_id"].astype(str).isin(needed)].copy()
    if len(result) != len(needed):
        raise ValueError("signal panel snapshot is missing from ForecastCache")
    return result


def _evaluation_predictions(root: Path, fold: object, timezone: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / fold.fold_id / "predictions.parquet")
    dates = _right_closed_target_dates(frame, timezone)
    return frame.loc[dates.between(*fold.predict)].copy()


def _build_or_load_tune_panel(args: argparse.Namespace, *, fold: object, tuning: pd.DataFrame, calibration: object, bias: object, observations: pd.DataFrame, availability: pd.DataFrame, markets: pd.DataFrame, station_id: str, timezone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = args.output / "frozen_selection" / fold.fold_id
    cache_root = root / "forecast_cache"
    panel_path = root / "panel.parquet"
    if panel_path.is_file() and cache_root.exists():
        cache = ForecastCache.load(cache_root)
        return pd.read_parquet(panel_path), cache.snapshots
    root.mkdir(parents=True, exist_ok=True)
    lineage_id = f"market-edge-frozen-tune-{fold.fold_id}"
    checkpoint = _fold_checkpoint(args.crossfit_root, fold.fold_id)
    lineage = pd.DataFrame([{"lineage_id": lineage_id, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "calibration_sha256": hashlib.sha256((mc_calibration_sha256(calibration) + partial_bias_sha256(bias)).encode()).hexdigest(), "split_role": "oof_development", "mc_seed": 42, "mc_samples": args.evaluation_mc_samples}])
    assignments = pd.DataFrame({"init_time": sorted(pd.to_datetime(tuning["init_time"], utc=True).unique()), "lineage_id": lineage_id})
    dates = _right_closed_target_dates(tuning, timezone)
    target_dates = sorted(dates.unique())
    cache = build_forecast_cache(hourly_predictions=tuning, observations=observations, availability=availability, markets=markets, lineages=lineage, cycle_lineages=assignments, calibrations={lineage_id: calibration}, partial_biases={lineage_id: bias}, station_id=station_id, station_timezone=timezone, target_start=target_dates[0], target_end=target_dates[-1])
    cache.save(cache_root, source_sha256=_tune_cache_source_hashes(args))
    validation = validate_study_inputs(study_id="market-edge-tune", station_id=station_id, start_target=target_dates[0], end_target=target_dates[-1], inputs=StudyInputs(forecast_root=cache_root, quotes_root=args.quotes_root, clob_root=args.clob_root, actuals_path=args.actual_highs), expected_split_role="oof_development")
    panel = build_signal_panel(validation).merge(
        cache.snapshots[["snapshot_id", "coverage_type", "forecast_hour_count"]],
        on="snapshot_id",
        how="inner",
        validate="many_to_one",
    )
    panel.to_parquet(panel_path, index=False)
    return panel, cache.snapshots


def _save_best(output: Path, fold_id: str, subset_id: str, result: dict[str, object]) -> None:
    root = output / fold_id / subset_id
    root.mkdir(parents=True, exist_ok=True)
    result["scores"].to_parquet(root / "candidate_scores.parquet", index=False)
    (root / "best_parameters.json").write_text(json.dumps({key: value for key, value in result.items() if key != "scores"}, indent=2, sort_keys=True, default=str), encoding="ascii")


def _load_subset_result(output: Path, fold_id: str, subset_id: str) -> dict[str, object] | None:
    root = output / fold_id / subset_id
    metadata_path, scores_path = root / "best_parameters.json", root / "candidate_scores.parquet"
    if not metadata_path.is_file() and not scores_path.is_file():
        return None
    if not metadata_path.is_file() or not scores_path.is_file():
        raise ValueError(f"incomplete resumable subset artifact: {root}")
    payload = json.loads(metadata_path.read_text(encoding="ascii"))
    return {**payload, "scores": pd.read_parquet(scores_path)}


def _report(args: argparse.Namespace, scores: pd.DataFrame, folds: list[dict[str, object]]) -> dict[str, object]:
    subset_reports: dict[str, object] = {}
    for subset_id, group in scores.groupby("subset_id", sort=True):
        subset_reports[str(subset_id)] = score_by_strata(group, bootstrap_samples=args.bootstrap_samples)
    return {"schema_version": 1, "data_class": "mc_market_edge_attribution", "protocol": {"mode": args.mode, "oracle_leakage": args.mode == "oracle_market_edge", "comparator": "normalized_synthetic_midpoint", "objective": "event_day_equal_model_minus_midpoint_normalized_rps", "frozen_excludes_fold_00": args.mode == "frozen_market_edge"}, "subsets": subset_reports, "folds": folds}


def _write_or_validate_contract(args: argparse.Namespace) -> None:
    contract = {"schema_version": 1, "mode": args.mode, "station": args.station, "comparator": "normalized_synthetic_midpoint", "crossfit_manifest": _hash(args.crossfit_root / "manifest.json"), "tune_manifest": _hash(args.tune_predictions_root / "manifest.json"), "static_signal_manifest": _hash(args.static_signal_root / "manifest.json"), "static_cache_manifest": _hash(args.static_cache_root / "manifest.json"), "trials": args.trials, "trial_mc_samples": args.trial_mc_samples, "evaluation_mc_samples": args.evaluation_mc_samples}
    path = args.output / "run_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="ascii")) != contract:
            raise ValueError("resume output contract does not match requested market-edge run")
    else:
        path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="ascii")


def _tune_cache_source_hashes(args: argparse.Namespace) -> dict[str, str]:
    """Bind frozen selection snapshots to the exact market input contract."""
    return {
        "purpose": "frozen_market_edge_selection",
        "crossfit_manifest": _hash(args.crossfit_root / "manifest.json"),
        "tune_predictions_manifest": _hash(args.tune_predictions_root / "manifest.json"),
        "observations_manifest": _hash(args.observations_root / "manifest.json"),
        "availability_manifest": _hash(args.availability_root / "manifest.json"),
        "clob_manifest": _hash(args.clob_root / "manifest.json"),
        "quotes_manifest": _hash(args.quotes_root / "manifest.json"),
        "actuals_manifest": _hash(
            args.actual_highs.with_suffix(args.actual_highs.suffix + ".manifest.json")
        ),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
