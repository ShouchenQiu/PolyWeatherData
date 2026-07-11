"""Read-only loading and metric tables for offline model evaluations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.signal_study import load_signal_study_artifact
from backtest.forecast_cache import ForecastCache
from model.mc_calibration import McCalibration


@dataclass(frozen=True)
class EvaluationArtifact:
    run_id: str
    version: str
    path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class EvaluationData:
    artifact: EvaluationArtifact
    hourly: pd.DataFrame
    daily: pd.DataFrame
    calibration: McCalibration | None
    calibration_evidence: dict[str, object]


@dataclass(frozen=True)
class SignalEvaluationData:
    roots: tuple[Path, ...]
    events: pd.DataFrame
    panel: pd.DataFrame


@dataclass(frozen=True)
class DailyHighPmfPilotData:
    root: Path
    report: dict[str, object]
    april_summary: pd.DataFrame
    may_summary: pd.DataFrame
    may_scores: pd.DataFrame


def load_daily_high_pmf_pilot(root: Path | str) -> DailyHighPmfPilotData:
    """Load read-only PMF pilot evidence; June must remain unconsumed."""
    path = Path(root)
    report = json.loads((path / "report.json").read_text(encoding="ascii"))
    if report.get("schema_version") != 1:
        raise ValueError("unsupported daily-high PMF pilot report")
    if report.get("june_consumed") is not False:
        raise ValueError("daily-high PMF pilot must not consume June")
    return DailyHighPmfPilotData(
        root=path,
        report=report,
        april_summary=pd.read_csv(path / "april_summary.csv"),
        may_summary=pd.read_csv(path / "may_summary.csv"),
        may_scores=pd.read_parquet(path / "may_parallel_scores.parquet"),
    )


def discover_evaluations(runs_root: Path | str) -> list[EvaluationArtifact]:
    """Find complete local-day calibrated validation artifact directories."""
    root = Path(runs_root)
    found: list[EvaluationArtifact] = []
    if not root.exists():
        return found
    required = {
        "validation_predictions.parquet",
        "mc_calibration_predictions.parquet",
        "mc_calibration_evidence.json",
        "mc_calibration.npz",
        "mc_calibration.json",
    }
    for path in root.glob("*/checkpoints/*"):
        if not path.is_dir() or not required <= {p.name for p in path.iterdir()}:
            continue
        checkpoints = sorted(path.glob("tft_best*.ckpt"))
        if len(checkpoints) != 1:
            continue
        found.append(
            EvaluationArtifact(
                run_id=path.parents[1].name,
                version=path.name,
                path=path,
                checkpoint_path=checkpoints[0],
            )
        )
    return sorted(
        found,
        key=lambda item: item.path.stat().st_mtime,
        reverse=True,
    )


def load_evaluation(artifact: EvaluationArtifact) -> EvaluationData:
    """Load and validate one complete evaluation."""
    hourly = pd.read_parquet(artifact.path / "validation_predictions.parquet")
    daily = pd.read_parquet(
        artifact.path / "mc_calibration_predictions.parquet"
    )
    evidence = json.loads(
        (artifact.path / "mc_calibration_evidence.json").read_text(
            encoding="ascii"
        )
    )
    calibration = McCalibration.load(
        artifact.path,
        artifact.checkpoint_path,
    )
    hourly = _prepare_hourly(hourly, calibration.station_timezone)
    daily["init_time"] = pd.to_datetime(daily["init_time"], utc=True)
    return EvaluationData(
        artifact=artifact,
        hourly=hourly,
        daily=daily,
        calibration=calibration,
        calibration_evidence=evidence,
    )


def load_backtest_model_evaluation(
    *,
    prediction_paths: list[Path] | tuple[Path, ...],
    forecast_root: Path,
    run_id: str = "KLGA May model-test",
    version: str = "202605 fixed hourly + MC snapshots",
    station_timezone: str = "America/New_York",
    target_start: str | None = "2026-05-01",
    target_end: str | None = "2026-05-31",
    actuals_path: Path | None = None,
) -> EvaluationData:
    """Load fixed backtest predictions as Model Evaluation data."""
    existing = [Path(path) for path in prediction_paths if Path(path).exists()]
    if not existing or not Path(forecast_root).exists():
        raise FileNotFoundError("May backtest model evaluation artifacts missing")
    raw = pd.concat(
        [pd.read_parquet(path) for path in existing],
        ignore_index=True,
    )
    hourly = _prepare_hourly(raw, station_timezone)
    hourly["target_date"] = (
        (hourly["valid_time"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(station_timezone)
        .dt.date.astype(str)
    )
    if target_start is not None:
        hourly = hourly[hourly["target_date"] >= target_start]
    if target_end is not None:
        hourly = hourly[hourly["target_date"] <= target_end]

    cache = ForecastCache.load(forecast_root)
    snapshots = cache.snapshots.copy()
    snapshots["init_time"] = pd.to_datetime(snapshots["init_time"], utc=True)
    snapshots["target_date"] = snapshots["target_date"].astype(str)
    if target_start is not None:
        snapshots = snapshots[snapshots["target_date"] >= target_start]
    if target_end is not None:
        snapshots = snapshots[snapshots["target_date"] <= target_end]
    complete_hours = hourly[hourly["weight"].astype(float) > 0]
    cycle_highs = (
        complete_hours.groupby(["init_time", "target_date"], as_index=False)
        .agg(
            hrrr_future_high_f=("hrrr_f", "max"),
            tft_future_high_f=("tft_p50_f", "max"),
        )
    )
    daily = snapshots.merge(
        cycle_highs,
        on=["init_time", "target_date"],
        how="inner",
        validate="one_to_one",
    )
    observed_high = daily["observed_high_f"].astype(float)
    daily["hrrr_high_f"] = daily["hrrr_future_high_f"].where(
        observed_high.isna(),
        daily[["hrrr_future_high_f", "observed_high_f"]].max(axis=1),
    )
    daily["tft_high_f"] = daily["tft_future_high_f"].where(
        observed_high.isna(),
        daily[["tft_future_high_f", "observed_high_f"]].max(axis=1),
    )
    if actuals_path is not None and Path(actuals_path).exists():
        actuals = pd.read_csv(actuals_path)
        actuals["target_date"] = actuals["target_date"].astype(str)
        daily = daily.drop(columns=["actual_high_f"], errors="ignore").merge(
            actuals[["target_date", "actual_high_f"]],
            on="target_date",
            how="left",
            validate="many_to_one",
        )
    else:
        actual_by_day = (
            hourly.groupby("target_date", as_index=False)
            .agg(actual_high_f=("actual_f", "max"))
        )
        daily = daily.merge(
            actual_by_day,
            on="target_date",
            how="left",
            validate="many_to_one",
        )
    daily = daily.rename(
        columns={
            "p10_f": "mc_p10_f",
            "p50_f": "mc_p50_f",
            "p90_f": "mc_p90_f",
        }
    )
    daily = daily[
        [
            "init_time",
            "target_date",
            "actual_high_f",
            "hrrr_high_f",
            "tft_high_f",
            "mc_p10_f",
            "mc_p50_f",
            "mc_p90_f",
            "coverage_type",
            "observed_hour_count",
            "forecast_hour_count",
        ]
    ].sort_values(["target_date", "init_time"])
    return EvaluationData(
        artifact=EvaluationArtifact(
            run_id=run_id,
            version=version,
            path=Path(forecast_root),
            checkpoint_path=Path(forecast_root) / "manifest.json",
        ),
        hourly=hourly,
        daily=daily,
        calibration=None,
        calibration_evidence={"source": "backtest_model_test"},
    )


def evaluation_metrics(data: EvaluationData) -> dict[str, float]:
    hourly = data.hourly
    daily = data.daily
    hrrr_mae = float(hourly["hrrr_abs_error_f"].mean())
    tft_mae = float(hourly["tft_abs_error_f"].mean())
    covered = (
        (daily["actual_high_f"] >= daily["mc_p10_f"])
        & (daily["actual_high_f"] <= daily["mc_p90_f"])
    )
    return {
        "hrrr_hourly_mae_f": hrrr_mae,
        "tft_hourly_mae_f": tft_mae,
        "hourly_skill_pct": (
            float((1.0 - tft_mae / hrrr_mae) * 100.0)
            if hrrr_mae > 0
            else float("nan")
        ),
        "hourly_coverage": float(hourly["covered"].mean()),
        "hourly_interval_width_f": float(
            (hourly["tft_p90_f"] - hourly["tft_p10_f"]).mean()
        ),
        "hrrr_high_mae_f": float(
            (daily["hrrr_high_f"] - daily["actual_high_f"]).abs().mean()
        ),
        "tft_high_mae_f": float(
            (daily["tft_high_f"] - daily["actual_high_f"]).abs().mean()
        ),
        "mc_high_mae_f": float(
            (daily["mc_p50_f"] - daily["actual_high_f"]).abs().mean()
        ),
        "mc_coverage": float(covered.mean()),
        "mc_interval_width_f": float(
            (daily["mc_p90_f"] - daily["mc_p10_f"]).mean()
        ),
    }


def daily_target_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse cycle-day forecasts to one median forecast per target day."""
    summary = (
        daily.groupby("target_date", as_index=False)
        .agg(
            actual_high_f=("actual_high_f", "first"),
            hrrr_high_f=("hrrr_high_f", "median"),
            tft_high_f=("tft_high_f", "median"),
            mc_p10_f=("mc_p10_f", "median"),
            mc_p50_f=("mc_p50_f", "median"),
            mc_p90_f=("mc_p90_f", "median"),
            cycle_count=("actual_high_f", "size"),
        )
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    summary["target_date"] = pd.to_datetime(summary["target_date"])
    return summary


def model_performance_table(data: EvaluationData) -> pd.DataFrame:
    """Return a compact TFT/MC comparison using each model's direct baseline."""
    metrics = evaluation_metrics(data)
    tft_high_mae = metrics["tft_high_mae_f"]
    mc_target = (
        "Daily high (model_test cycle-day)"
        if data.calibration is None
        else "Daily high (OOF cycle-day)"
    )
    mc_skill = (
        (1.0 - metrics["mc_high_mae_f"] / tft_high_mae) * 100.0
        if tft_high_mae > 0
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "Model": "TFT",
                "Target": "Hourly temperature",
                "Baseline": "HRRR",
                "MAE (°F)": metrics["tft_hourly_mae_f"],
                "Baseline MAE (°F)": metrics["hrrr_hourly_mae_f"],
                "Skill vs baseline (%)": metrics["hourly_skill_pct"],
                "P10–P90 coverage (%)": metrics["hourly_coverage"] * 100.0,
                "Mean interval width (°F)": metrics["hourly_interval_width_f"],
                "Samples": len(data.hourly),
            },
            {
                "Model": "MC",
                "Target": mc_target,
                "Baseline": "TFT",
                "MAE (°F)": metrics["mc_high_mae_f"],
                "Baseline MAE (°F)": tft_high_mae,
                "Skill vs baseline (%)": mc_skill,
                "P10–P90 coverage (%)": metrics["mc_coverage"] * 100.0,
                "Mean interval width (°F)": metrics["mc_interval_width_f"],
                "Samples": len(data.daily),
            },
        ]
    )


def horizon_metrics(data: EvaluationData) -> pd.DataFrame:
    return (
        data.hourly.groupby("fxx", as_index=False)
        .agg(
            hrrr_mae_f=("hrrr_abs_error_f", "mean"),
            tft_mae_f=("tft_abs_error_f", "mean"),
            hrrr_bias_f=("hrrr_error_f", "mean"),
            tft_bias_f=("tft_error_f", "mean"),
            coverage=("covered", "mean"),
        )
        .sort_values("fxx")
    )


def skill_heatmap(data: EvaluationData) -> pd.DataFrame:
    work = data.hourly.copy()
    work["fxx_band"] = pd.cut(
        work["fxx"],
        [0, 6, 12, 24, 36, 48],
        labels=["1-6", "7-12", "13-24", "25-36", "37-48"],
    )
    grouped = work.groupby(
        ["init_hour", "fxx_band"],
        observed=True,
    ).agg(
        hrrr=("hrrr_abs_error_f", "mean"),
        tft=("tft_abs_error_f", "mean"),
    )
    grouped["skill_pct"] = (1.0 - grouped["tft"] / grouped["hrrr"]) * 100
    return grouped["skill_pct"].unstack("fxx_band")


def load_signal_evaluation(roots: list[Path] | tuple[Path, ...]) -> SignalEvaluationData:
    """Load model-vs-midpoint signal artifacts into event diagnostics."""
    loaded = []
    for root in roots:
        path = Path(root)
        if path.exists():
            loaded.append((path, load_signal_study_artifact(path)))
    if not loaded:
        return SignalEvaluationData(tuple(), pd.DataFrame(), pd.DataFrame())

    event_frames = []
    panel_frames = []
    for root, artifact in loaded:
        label = root.name
        by_event = artifact.by_event.copy()
        pivot = by_event.pivot(index=["event_id", "target_date"], columns="source")
        events = pd.DataFrame(
            {
                "event_id": [str(index[0]) for index in pivot.index],
                "target_date": [str(index[1]) for index in pivot.index],
                "artifact": label,
                "model_brier": pivot["multiclass_brier"]["model_raw"],
                "midpoint_brier": pivot["multiclass_brier"][
                    "synthetic_midpoint"
                ],
                "model_log_score": pivot["log_score"]["model_raw"],
                "midpoint_log_score": pivot["log_score"][
                    "synthetic_midpoint"
                ],
                "model_rps": pivot["normalized_rps"]["model_raw"],
                "midpoint_rps": pivot["normalized_rps"][
                    "synthetic_midpoint"
                ],
                "model_true_bucket_probability": pivot[
                    "true_bucket_probability"
                ]["model_raw"],
                "midpoint_true_bucket_probability": pivot[
                    "true_bucket_probability"
                ]["synthetic_midpoint"],
                "model_top1_accuracy": pivot["top1_accuracy"]["model_raw"],
                "midpoint_top1_accuracy": pivot["top1_accuracy"][
                    "synthetic_midpoint"
                ],
            }
        ).reset_index(drop=True)
        actuals = (
            artifact.panel[
                ["event_id", "target_date", "actual_high_f"]
            ]
            .drop_duplicates()
            .assign(event_id=lambda frame: frame["event_id"].astype(str))
        )
        events = events.merge(
            actuals,
            on=["event_id", "target_date"],
            how="left",
            validate="one_to_one",
        )
        events["brier_delta"] = events["model_brier"] - events["midpoint_brier"]
        events["log_score_delta"] = (
            events["model_log_score"] - events["midpoint_log_score"]
        )
        events["rps_delta"] = events["model_rps"] - events["midpoint_rps"]
        events["true_bucket_probability_delta"] = (
            events["model_true_bucket_probability"]
            - events["midpoint_true_bucket_probability"]
        )
        event_frames.append(events)
        panel_frames.append(artifact.panel.assign(artifact=label))

    event_frame = pd.concat(event_frames, ignore_index=True)
    event_frame["target_date"] = pd.to_datetime(event_frame["target_date"])
    panel = pd.concat(panel_frames, ignore_index=True)
    panel["target_date"] = pd.to_datetime(panel["target_date"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    return SignalEvaluationData(
        roots=tuple(root for root, _artifact in loaded),
        events=event_frame.sort_values("target_date").reset_index(drop=True),
        panel=panel,
    )


def _prepare_hourly(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    work = frame[frame["weight"].astype(float) > 0].copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    work["actual_f"] = (work["base_temp_c"] + work["target"]) * 1.8 + 32
    work["hrrr_f"] = work["base_temp_c"] * 1.8 + 32
    for quantile in ("p10", "p50", "p90"):
        work[f"tft_{quantile}_f"] = (
            work["base_temp_c"] + work[quantile]
        ) * 1.8 + 32
    work["hrrr_error_f"] = work["hrrr_f"] - work["actual_f"]
    work["tft_error_f"] = work["tft_p50_f"] - work["actual_f"]
    work["hrrr_abs_error_f"] = work["hrrr_error_f"].abs()
    work["tft_abs_error_f"] = work["tft_error_f"].abs()
    work["covered"] = (
        (work["actual_f"] >= work["tft_p10_f"])
        & (work["actual_f"] <= work["tft_p90_f"])
    )
    work["init_hour"] = work["init_time"].dt.hour
    work["local_hour"] = work["valid_time"].dt.tz_convert(timezone).dt.hour
    return work
