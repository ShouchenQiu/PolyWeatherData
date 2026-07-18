"""Checkpoint-aligned full-day MC parameter attribution.

Calibration always uses an earlier tune block from the same TFT checkpoint.
``frozen_tune`` selects parameters on that tune block and evaluates later OOF
data. ``oracle_eval`` deliberately selects on evaluation outcomes to estimate
a leakage-labelled parameter ceiling. Neither mode measures true point-in-time
partial-day bias behavior or real-market edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from data.storage.parquet_store import ParquetStore
from model.daily_high import fit_partial_day_bias
from training.historical_crossfit import historical_six_fold_plan
from training.mc_calibration import fit_mc_calibration
from training.mc_prequential_optuna import (
    apply_mc_tune_parameters,
    enqueue_identity_mc_trial,
    event_day_rps,
    mc_tune_component_subsets,
    score_daily_high_rps,
    shapley_rps_attribution,
    suggest_mc_tune_parameters,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--tune-predictions-root", required=True, type=Path)
    parser.add_argument("--historical-data-store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--calibration-mc-samples", type=int, default=2_000)
    parser.add_argument("--trial-mc-samples", type=int, default=256)
    parser.add_argument("--evaluation-mc-samples", type=int, default=4_000)
    parser.add_argument("--fold-id", action="append", dest="fold_ids")
    parser.add_argument(
        "--selection-mode",
        choices=("frozen_tune", "oracle_eval"),
        default="frozen_tune",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(args.trials, args.calibration_mc_samples, args.trial_mc_samples, args.evaluation_mc_samples) <= 0:
        raise ValueError("trial and MC sample counts must be positive")
    print(f"[start] checkpoint-aligned MC Optuna output={args.output}", flush=True)
    _prepare_output(args.output, resume=args.resume)
    folds = _planned_folds(args.crossfit_root, requested_fold_ids=args.fold_ids)
    fold_ids = tuple(fold.fold_id for fold in folds)
    predictions = {
        fold.fold_id: filter_evaluation_prediction_window(
            pd.read_parquet(args.crossfit_root / fold.fold_id / "predictions.parquet"),
            predict=fold.predict,
            station_timezone=args.timezone,
        )
        for fold in folds
    }
    tuning = {
        fold_id: _load_checkpoint_aligned_tune_predictions(
            crossfit_root=args.crossfit_root,
            tune_root=args.tune_predictions_root,
            fold_id=fold_id,
            predict_start=next(fold.predict[0] for fold in folds if fold.fold_id == fold_id),
            station_timezone=args.timezone,
        )
        for fold_id in fold_ids
    }
    _write_or_validate_contract(args, fold_ids)
    observations = _load_observations(
        args.historical_data_store,
        station_id=args.station,
        frames=(*predictions.values(), *tuning.values()),
    )
    all_scores: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    for fold_index, fold_id in enumerate(fold_ids):
        calibration_frame = tuning[fold_id]
        evaluation_frame = predictions[fold_id]
        selection_frame = selection_frame_for_mode(
            args.selection_mode, calibration_frame, evaluation_frame
        )
        checkpoint = _fold_checkpoint(args.crossfit_root / fold_id / "manifest.json")
        base_calibration = fit_mc_calibration(
            calibration_frame,
            checkpoint,
            station_timezone=args.timezone,
            mc_samples=args.calibration_mc_samples,
        )
        base_bias = fit_partial_day_bias(
            hourly_predictions=calibration_frame,
            observations=observations,
            calibration=base_calibration,
            station_timezone=args.timezone,
            mc_samples=args.calibration_mc_samples,
            seed=42,
        )
        fold_root = args.output / fold_id
        fold_root.mkdir(parents=True, exist_ok=True)
        print(
            f"[start] fold={fold_id} calibration_source=same_checkpoint_tune",
            flush=True,
        )
        subset_scores: dict[str, float] = {}
        for subset_index, components in enumerate(mc_tune_component_subsets()):
            result = _run_subset(
                fold_root=fold_root,
                fold_id=fold_id,
                components=components,
                calibration_frame=calibration_frame,
                selection_frame=selection_frame,
                evaluation_frame=evaluation_frame,
                base_calibration=base_calibration,
                base_bias=base_bias,
                station_timezone=args.timezone,
                trials=args.trials,
                trial_mc_samples=args.trial_mc_samples,
                evaluation_mc_samples=args.evaluation_mc_samples,
                seed=42 + fold_index * 100 + subset_index,
                resume=args.resume,
            )
            all_scores.append(result)
            subset_scores[components.subset_id] = event_day_rps(result)
        fold_shapley = shapley_rps_attribution(subset_scores)
        fold_summaries.extend(
            {
                "fold_id": fold_id,
                "subset_id": subset_id,
                "rps_event_day": score,
                **{f"shapley_{name}": fold_shapley[name] for name in fold_shapley},
            }
            for subset_id, score in subset_scores.items()
        )
        print(f"[complete] fold={fold_id}", flush=True)

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(args.output / "evaluation_scores.parquet", index=False)
    summary = _summarize(scores)
    summary.to_csv(args.output / "summary.csv", index=False)
    fold_summary = pd.DataFrame(fold_summaries)
    fold_summary.to_csv(args.output / "fold_summary.csv", index=False)
    totals = {
        row["subset_id"]: float(row["rps_event_day"])
        for row in summary.to_dict("records")
    }
    report = {
        "schema_version": 4,
        "data_class": "checkpoint_aligned_mc_optuna_attribution",
        "station_id": args.station,
        "protocol": {
            "selection_mode": args.selection_mode,
            "oracle_ceiling": args.selection_mode == "oracle_eval",
            "objective": "event_day_equal_fixed_integer_daily_high_rps",
            "evaluation_folds": fold_ids,
            "calibration_evidence": "same_checkpoint_tune_precedes_evaluation",
            "selection_evidence": (
                "evaluation_outcomes_leaked_oracle"
                if args.selection_mode == "oracle_eval"
                else "same_checkpoint_tune"
            ),
            "excluded_periods": ["2026-04", "2026-05", "2026-06"],
            "trials_per_fold_subset": args.trials,
            "calibration_mc_samples": args.calibration_mc_samples,
            "trial_mc_samples": args.trial_mc_samples,
            "evaluation_mc_samples": args.evaluation_mc_samples,
        },
        "parameter_family": {
            "sigma": "five fxx-band multipliers in [0.6, 1.6]",
            "correlation": "convex empirical/AR1 blend with rho in [0, 0.98]",
            "partial_bias": "full forecast-window max-bias proxy; not point-in-time partial-day stitching",
        },
        "aggregate_rps": totals,
        "aggregate_shapley_rps_improvement": shapley_rps_attribution(totals),
        "source_sha256": _source_hashes(args.crossfit_root, args.tune_predictions_root, fold_ids),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="ascii"
    )
    print(f"[complete] checkpoint-aligned MC Optuna rows={len(scores)}", flush=True)
    print(f"[done] checkpoint-aligned MC Optuna output={args.output}", flush=True)
    return 0


def _run_subset(
    *,
    fold_root: Path,
    fold_id: str,
    components: object,
    calibration_frame: pd.DataFrame,
    selection_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    base_calibration: object,
    base_bias: object,
    station_timezone: str,
    trials: int,
    trial_mc_samples: int,
    evaluation_mc_samples: int,
    seed: int,
    resume: bool,
) -> pd.DataFrame:
    from training.mc_prequential_optuna import McTuneComponents

    assert isinstance(components, McTuneComponents)
    from model.daily_high import PartialDayBiasCalibration
    from model.mc_calibration import McCalibration

    assert isinstance(base_calibration, McCalibration)
    assert isinstance(base_bias, PartialDayBiasCalibration)
    root = fold_root / components.subset_id
    scores_path = root / "evaluation_scores.parquet"
    if scores_path.is_file():
        if not resume:
            raise FileExistsError(scores_path)
        print(
            f"[complete] fold={fold_id} subset={components.subset_id} resume",
            flush=True,
        )
        return pd.read_parquet(scores_path)
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError(f"incomplete subset output requires resume inspection: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if components.subset_id == "none":
        parameters: dict[str, float] = {}
        optimization_rps = event_day_rps(
            score_daily_high_rps(
                selection_frame, base_calibration, base_bias,
                station_timezone=station_timezone, mc_samples=trial_mc_samples,
            )
        )
    else:
        storage = f"sqlite:///{(root / 'optuna.sqlite').resolve()}"
        study = optuna.create_study(
            study_name=f"prequential-{fold_id}-{components.subset_id}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        enqueue_identity_mc_trial(study, components)

        def objective(trial: optuna.Trial) -> float:
            parameters = suggest_mc_tune_parameters(trial, components)
            calibration, bias = apply_mc_tune_parameters(
                base_calibration, base_bias, components=components, parameters=parameters
            )
            rows = score_daily_high_rps(
                selection_frame, calibration, bias,
                station_timezone=station_timezone, mc_samples=trial_mc_samples,
            )
            return event_day_rps(rows)

        def progress(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
            complete = sum(
                item.state == optuna.trial.TrialState.COMPLETE
                for item in study.trials
            )
            if complete == 1 or complete % 5 == 0:
                print(
                    f"[progress] fold={fold_id} subset={components.subset_id} "
                    f"trials={complete}/{trials} best_rps={study.best_value:.6f}",
                    flush=True,
                )

        remaining = max(0, trials - sum(
            item.state == optuna.trial.TrialState.COMPLETE for item in study.trials
        ))
        if remaining:
            print(
                f"[start] fold={fold_id} subset={components.subset_id} trials={trials}",
                flush=True,
            )
            study.optimize(objective, n_trials=remaining, callbacks=[progress])
        parameters = {name: float(value) for name, value in study.best_params.items()}
        optimization_rps = float(study.best_value)
    calibration, bias = apply_mc_tune_parameters(
        base_calibration, base_bias, components=components, parameters=parameters
    )
    rows = score_daily_high_rps(
        evaluation_frame, calibration, bias,
        station_timezone=station_timezone, mc_samples=evaluation_mc_samples,
    )
    rows.insert(0, "fold_id", fold_id)
    rows.insert(1, "subset_id", components.subset_id)
    rows.to_parquet(scores_path, index=False)
    (root / "best_parameters.json").write_text(
        json.dumps(
            {
                "fold_id": fold_id,
                "subset_id": components.subset_id,
                "optimization_rps_event_day": optimization_rps,
                "parameters": parameters,
                "evaluation_rps_event_day": event_day_rps(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="ascii",
    )
    print(
        f"[complete] fold={fold_id} subset={components.subset_id} "
        f"evaluation_rows={len(rows)}",
        flush=True,
    )
    return rows


def _load_observations(
    data_store: Path,
    *,
    station_id: str,
    frames: object,
) -> pd.DataFrame:
    all_frames = list(frames)
    earliest = min(pd.to_datetime(frame["valid_time"], utc=True).min() for frame in all_frames)
    latest = max(pd.to_datetime(frame["valid_time"], utc=True).max() for frame in all_frames)
    raw = ParquetStore(data_store / "obs").read(
        station_id, earliest - pd.Timedelta(days=2), latest + pd.Timedelta(days=2)
    )
    return pd.DataFrame(
        {
            "valid_time": pd.to_datetime(raw["valid_time"], utc=True),
            "temperature_c": pd.to_numeric(raw["raw_tmpc_tenths"], errors="raise"),
        }
    ).drop_duplicates("valid_time", keep="last")


def _fold_checkpoint(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    value = manifest.get("train_result", {}).get("checkpoint_path")
    if not isinstance(value, str) or not Path(value).is_file():
        raise FileNotFoundError(f"crossfit fold checkpoint missing: {manifest_path}")
    return Path(value)


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_id, group in scores.groupby("subset_id", sort=True):
        work = group.copy()
        work["p50_abs_error_f"] = (work["p50_f"] - work["actual_high_f"]).abs()
        work["interval_width_f"] = work["p90_f"] - work["p10_f"]
        work["p10_p90_covered"] = (
            (work["actual_high_f"] >= work["p10_f"])
            & (work["actual_high_f"] <= work["p90_f"])
        ).astype(float)
        day_means = work.groupby("target_date", as_index=False)[
            ["p50_abs_error_f", "interval_width_f", "p10_p90_covered", "true_integer_probability"]
        ].mean()
        rows.append(
            {
                "subset_id": subset_id,
                "snapshots": len(group),
                "event_days": group["target_date"].nunique(),
                "rps_event_day": event_day_rps(group),
                "p50_mae_f": float(day_means["p50_abs_error_f"].mean()),
                "p10_p90_coverage": float(day_means["p10_p90_covered"].mean()),
                "p10_p90_width_f": float(day_means["interval_width_f"].mean()),
                "true_integer_probability": float(day_means["true_integer_probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _planned_folds(
    crossfit_root: Path,
    *,
    requested_fold_ids: list[str] | None = None,
) -> tuple[object, ...]:
    actual = tuple(sorted(
        path.name for path in crossfit_root.glob("fold-*")
        if (path / "predictions.parquet").is_file()
    ))
    planned = historical_six_fold_plan().folds
    expected = tuple(fold.fold_id for fold in planned)
    if actual != expected:
        raise ValueError("crossfit root does not match the fixed six-fold plan")
    if requested_fold_ids is None:
        return planned
    requested = tuple(dict.fromkeys(requested_fold_ids))
    unknown = sorted(set(requested) - set(expected))
    if unknown:
        raise ValueError("unknown requested fold ids: " + ", ".join(unknown))
    selected = tuple(fold for fold in planned if fold.fold_id in requested)
    if not selected:
        raise ValueError("at least one fold id must be requested")
    return selected


def _load_checkpoint_aligned_tune_predictions(
    *, crossfit_root: Path, tune_root: Path, fold_id: str, predict_start: str,
    station_timezone: str,
) -> pd.DataFrame:
    prediction_path = tune_root / fold_id / "tune_predictions.parquet"
    manifest_path = prediction_path.with_suffix(".parquet.manifest.json")
    if not prediction_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint-aligned tune predictions missing: {fold_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = _fold_checkpoint(crossfit_root / fold_id / "manifest.json")
    if manifest.get("fold", {}).get("fold_id") != fold_id:
        raise ValueError(f"tune prediction fold mismatch: {fold_id}")
    if manifest.get("source_sha256", {}).get("checkpoint") != _sha256(checkpoint):
        raise ValueError(f"tune prediction checkpoint mismatch: {fold_id}")
    if manifest.get("sha256", {}).get(prediction_path.name) != _sha256(prediction_path):
        raise ValueError(f"tune prediction hash mismatch: {fold_id}")
    frame = pd.read_parquet(prediction_path)
    dates = _right_closed_target_dates(frame, station_timezone)
    tune_start, tune_end = manifest["fold"]["tune"]
    filtered = frame.loc[dates.between(tune_start, tune_end)].copy()
    if filtered.empty:
        raise ValueError(f"tune prediction artifact has no trusted target days: {fold_id}")
    if str(tune_end) >= str(predict_start):
        raise ValueError(f"tune/evaluation target-date overlap: {fold_id}")
    return filtered


def _right_closed_target_dates(frame: pd.DataFrame, timezone: str) -> pd.Series:
    return (
        (pd.to_datetime(frame["valid_time"], utc=True) - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(timezone).dt.date.astype(str)
    )


def filter_evaluation_prediction_window(
    frame: pd.DataFrame,
    *, predict: tuple[str, str], station_timezone: str,
) -> pd.DataFrame:
    """Remove 48h horizon tails outside the declared cross-fit test block."""
    dates = _right_closed_target_dates(frame, station_timezone)
    result = frame.loc[dates.between(*predict)].copy()
    if result.empty:
        raise ValueError("evaluation prediction window has no target cells")
    return result


def selection_frame_for_mode(
    selection_mode: str,
    calibration_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Select tuning evidence, or deliberately leaked evaluation evidence."""
    if selection_mode == "frozen_tune":
        return calibration_frame
    if selection_mode == "oracle_eval":
        return evaluation_frame
    raise ValueError(f"unknown MC parameter selection mode: {selection_mode}")


def _source_hashes(crossfit_root: Path, tune_root: Path, fold_ids: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for fold_id in fold_ids:
        hashes[f"{fold_id}_predictions"] = _sha256(crossfit_root / fold_id / "predictions.parquet")
        hashes[f"{fold_id}_tune_predictions"] = _sha256(tune_root / fold_id / "tune_predictions.parquet")
        hashes[f"{fold_id}_tune_manifest"] = _sha256(tune_root / fold_id / "tune_predictions.parquet.manifest.json")
    return hashes


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _prepare_output(output: Path, *, resume: bool) -> None:
    if output.exists() and not resume:
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _write_or_validate_contract(args: argparse.Namespace, fold_ids: tuple[str, ...]) -> None:
    path = args.output / "run_contract.json"
    payload = {
        "schema_version": 4,
        "fold_ids": list(fold_ids),
        "station": args.station,
        "timezone": args.timezone,
        "trials": args.trials,
        "calibration_mc_samples": args.calibration_mc_samples,
        "trial_mc_samples": args.trial_mc_samples,
        "evaluation_mc_samples": args.evaluation_mc_samples,
        "requested_fold_ids": args.fold_ids,
        "selection_mode": args.selection_mode,
        "fixed_integer_support_f": [-100, 160],
        "evaluation_window_contract": "right_closed_local_target_date_exact_predict_range",
        "identity_candidate_contract": "every_tuned_subset_enqueues_exact_static_mc_point",
        "source_sha256": _source_hashes(args.crossfit_root, args.tune_predictions_root, fold_ids),
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="ascii"))
        if existing != payload:
            raise ValueError("resume output contract does not match current inputs or settings")
        return
    if any(args.output.iterdir()):
        raise ValueError("output has payloads but no valid run contract")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="ascii")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
