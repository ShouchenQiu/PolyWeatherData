"""Strict expanding static-MC ForecastCache construction for historical panels."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from model.daily_high import fit_partial_day_bias
from training.backtest_forecast_generation import (
    build_forecast_cache,
    mc_calibration_sha256,
    partial_bias_sha256,
)
from training.historical_crossfit import load_historical_crossfit_predictions
from training.historical_crossfit import historical_six_fold_plan
from training.mc_attribution import historical_attribution_folds
from training.mc_calibration import fit_mc_calibration


@dataclass(frozen=True)
class HistoricalMarketPanelResult:
    cache: ForecastCache
    fold_audit: pd.DataFrame


def build_checkpoint_aligned_historical_market_panel_cache(
    *,
    crossfit_root: Path | str,
    tune_predictions_root: Path | str,
    observations: pd.DataFrame,
    availability: pd.DataFrame,
    markets: pd.DataFrame,
    station_id: str,
    station_timezone: str,
    mc_samples: int = 2_000,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> HistoricalMarketPanelResult:
    """Build MC caches from tune residuals of the same TFT checkpoint.

    Each tune block precedes its fold's prediction block.  Unlike the legacy
    expanding implementation, no residual from another TFT checkpoint may
    calibrate the evaluated checkpoint.
    """
    root = Path(crossfit_root)
    tune_root = Path(tune_predictions_root)
    all_predictions = load_historical_crossfit_predictions(root)
    folds_by_id = {fold.fold_id: fold for fold in historical_six_fold_plan().folds}
    caches: list[ForecastCache] = []
    audits: list[dict[str, object]] = []
    for fold_id in sorted(all_predictions["fold_id"].unique()):
        if str(fold_id) not in folds_by_id:
            raise ValueError(f"cross-fit fold is absent from the fixed plan: {fold_id}")
        evaluation = filter_evaluation_prediction_window(
            all_predictions[all_predictions["fold_id"] == fold_id],
            predict=folds_by_id[str(fold_id)].predict,
            station_timezone=station_timezone,
        )
        if evaluation.empty:
            raise ValueError(f"evaluation prediction window is empty: {fold_id}")
        checkpoint = _fold_checkpoint(root, str(fold_id))
        tuning = _load_checkpoint_aligned_tune_predictions(
            root=root,
            tune_root=tune_root,
            fold_id=str(fold_id),
            checkpoint=checkpoint,
            station_timezone=station_timezone,
        )
        calibration = fit_mc_calibration(
            tuning,
            checkpoint,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
        )
        bias = fit_partial_day_bias(
            hourly_predictions=tuning,
            observations=observations,
            calibration=calibration,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
            seed=42,
        )
        lineage_id = f"checkpoint-aligned-static-mc-{fold_id}"
        lineage = pd.DataFrame([{
            "lineage_id": lineage_id,
            "checkpoint_sha256": calibration.model_sha256,
            "calibration_sha256": _lineage_sha256(calibration, bias, (str(fold_id),)),
            "split_role": "oof_development",
            "mc_seed": 42,
            "mc_samples": mc_samples,
        }])
        assignments = pd.DataFrame({
            "init_time": sorted(evaluation["init_time"].unique()),
            "lineage_id": lineage_id,
        })
        target_dates = _strict_target_dates(evaluation, station_timezone)
        cache = build_forecast_cache(
            hourly_predictions=evaluation,
            observations=observations,
            availability=availability,
            markets=markets,
            lineages=lineage,
            cycle_lineages=assignments,
            calibrations={lineage_id: calibration},
            partial_biases={lineage_id: bias},
            station_id=station_id,
            station_timezone=station_timezone,
            target_start=min(target_dates),
            target_end=max(target_dates),
        )
        caches.append(cache)
        if progress_callback is not None:
            progress_callback(str(fold_id), len(cache.snapshots), len(cache.probabilities))
        audits.append({
            "evaluation_fold_id": str(fold_id),
            "calibration_source": "same_checkpoint_tune",
            "calibration_target_start": _right_closed_target_dates(tuning, station_timezone).min(),
            "calibration_target_end": _right_closed_target_dates(tuning, station_timezone).max(),
            "evaluation_target_start": min(target_dates),
            "evaluation_target_end": max(target_dates),
            "tuning_prediction_rows": len(tuning),
            "evaluation_prediction_rows": len(evaluation),
            "snapshot_count": len(cache.snapshots),
            "probability_count": len(cache.probabilities),
            "checkpoint_sha256": calibration.model_sha256,
            "mc_calibration_sha256": mc_calibration_sha256(calibration),
            "partial_bias_sha256": partial_bias_sha256(bias),
        })
    return HistoricalMarketPanelResult(
        ForecastCache(
            snapshots=pd.concat([item.snapshots for item in caches], ignore_index=True),
            probabilities=pd.concat([item.probabilities for item in caches], ignore_index=True),
            lineages=pd.concat([item.lineages for item in caches], ignore_index=True),
        ),
        pd.DataFrame(audits),
    )


def build_historical_market_panel_cache(
    *,
    crossfit_root: Path | str,
    observations: pd.DataFrame,
    availability: pd.DataFrame,
    markets: pd.DataFrame,
    station_id: str,
    station_timezone: str,
    mc_samples: int = 2_000,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> HistoricalMarketPanelResult:
    """Build each OOF block only from calibration evidence in earlier blocks."""
    root = Path(crossfit_root)
    all_predictions = load_historical_crossfit_predictions(root)
    caches: list[ForecastCache] = []
    audits: list[dict[str, object]] = []
    for boundary in historical_attribution_folds(root):
        training = pd.concat(
            [
                all_predictions[all_predictions["fold_id"] == fold_id]
                for fold_id in boundary.calibration_fold_ids
            ],
            ignore_index=True,
        )
        evaluation = all_predictions[
            all_predictions["fold_id"] == boundary.evaluation_fold_id
        ].copy()
        if training.empty or evaluation.empty:
            raise ValueError(f"missing cross-fit predictions for {boundary}")
        checkpoint = _fold_checkpoint(root, boundary.evaluation_fold_id)
        calibration = fit_mc_calibration(
            training,
            checkpoint,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
        )
        bias = fit_partial_day_bias(
            hourly_predictions=training,
            observations=observations,
            calibration=calibration,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
            seed=42,
        )
        lineage_id = f"historical-static-mc-{boundary.evaluation_fold_id}"
        lineage = pd.DataFrame(
            [
                {
                    "lineage_id": lineage_id,
                    "checkpoint_sha256": calibration.model_sha256,
                    "calibration_sha256": _lineage_sha256(
                        calibration, bias, boundary.calibration_fold_ids
                    ),
                    "split_role": "oof_development",
                    "mc_seed": 42,
                    "mc_samples": mc_samples,
                }
            ]
        )
        cycle_lineages = pd.DataFrame(
            {
                "init_time": sorted(evaluation["init_time"].unique()),
                "lineage_id": lineage_id,
            }
        )
        target_dates = _strict_target_dates(evaluation, station_timezone)
        cache = build_forecast_cache(
            hourly_predictions=evaluation,
            observations=observations,
            availability=availability,
            markets=markets,
            lineages=lineage,
            cycle_lineages=cycle_lineages,
            calibrations={lineage_id: calibration},
            partial_biases={lineage_id: bias},
            station_id=station_id,
            station_timezone=station_timezone,
            target_start=min(target_dates),
            target_end=max(target_dates),
        )
        caches.append(cache)
        if progress_callback is not None:
            progress_callback(
                boundary.evaluation_fold_id,
                len(cache.snapshots),
                len(cache.probabilities),
            )
        audits.append(
            {
                "evaluation_fold_id": boundary.evaluation_fold_id,
                "calibration_fold_ids": ",".join(boundary.calibration_fold_ids),
                "calibration_last_init": pd.to_datetime(
                    training["init_time"], utc=True
                ).max(),
                "evaluation_first_init": pd.to_datetime(
                    evaluation["init_time"], utc=True
                ).min(),
                "evaluation_target_start": min(target_dates),
                "evaluation_target_end": max(target_dates),
                "training_prediction_rows": len(training),
                "evaluation_prediction_rows": len(evaluation),
                "snapshot_count": len(cache.snapshots),
                "probability_count": len(cache.probabilities),
                "checkpoint_sha256": calibration.model_sha256,
                "mc_calibration_sha256": mc_calibration_sha256(calibration),
                "partial_bias_sha256": partial_bias_sha256(bias),
            }
        )
    if not caches:
        raise ValueError("historical market panel has no evaluation folds")
    combined = ForecastCache(
        snapshots=pd.concat([item.snapshots for item in caches], ignore_index=True),
        probabilities=pd.concat(
            [item.probabilities for item in caches], ignore_index=True
        ),
        lineages=pd.concat([item.lineages for item in caches], ignore_index=True),
    )
    return HistoricalMarketPanelResult(combined, pd.DataFrame(audits))


def _fold_checkpoint(root: Path, fold_id: str) -> Path:
    payload = json.loads(
        (root / fold_id / "crossfit_manifest.json").read_text(encoding="ascii")
    )
    relative = payload.get("artifacts", {}).get("checkpoint")
    if not relative:
        raise ValueError(f"cross-fit fold missing checkpoint artifact: {fold_id}")
    path = root / fold_id / str(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_checkpoint_aligned_tune_predictions(
    *,
    root: Path,
    tune_root: Path,
    fold_id: str,
    checkpoint: Path,
    station_timezone: str,
) -> pd.DataFrame:
    manifest_path = tune_root / fold_id / "tune_predictions.parquet.manifest.json"
    prediction_path = tune_root / fold_id / "tune_predictions.parquet"
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"checkpoint-aligned tune predictions missing: {fold_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        raise ValueError(f"tune predictions have no target cells in tune window: {fold_id}")
    return filtered


def _right_closed_target_dates(frame: pd.DataFrame, timezone: str) -> pd.Series:
    return (
        (pd.to_datetime(frame["valid_time"], utc=True) - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(timezone)
        .dt.date.astype(str)
    )


def filter_evaluation_prediction_window(
    frame: pd.DataFrame,
    *,
    predict: tuple[str, str],
    station_timezone: str,
) -> pd.DataFrame:
    """Keep only hourly cells belonging to a fold's declared local dates.

    Cross-fit prediction files can include the first hour of the following
    local day.  Letting that tail into ForecastCache creates an undeclared
    target day, then later PIT candidate reconstruction cannot legitimately
    find its full 24-hour forecast.
    """
    dates = _right_closed_target_dates(frame, station_timezone)
    return frame.loc[dates.between(*predict)].copy()


def _strict_target_dates(frame: pd.DataFrame, timezone: str) -> list[str]:
    """Return local target days using the same right-closed contract as folds."""
    dates = sorted(_right_closed_target_dates(frame, timezone).unique().tolist())
    if not dates:
        raise ValueError("evaluation predictions have no strict target dates")
    return dates


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_dates(frame: pd.DataFrame, timezone: str) -> list[str]:
    dates = (
        pd.to_datetime(frame["valid_time"], utc=True)
        .dt.tz_convert(timezone)
        .dt.date.astype(str)
        .unique()
        .tolist()
    )
    if not dates:
        raise ValueError("evaluation predictions have no target dates")
    return sorted(dates)


def _lineage_sha256(calibration: object, bias: object, fold_ids: tuple[str, ...]) -> str:
    payload = {
        "calibration": mc_calibration_sha256(calibration),
        "partial_bias": partial_bias_sha256(bias),
        "calibration_folds": list(fold_ids),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
