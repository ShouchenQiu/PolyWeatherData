"""Read-only loading and metric tables for offline model evaluations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    calibration: McCalibration
    calibration_evidence: dict[str, object]


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
                "Target": "Daily high (OOF cycle-day)",
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
