"""Offline HRRR decoder verification against canonical IEM observations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.forecast.hrrr_fill import MAJOR_HOURS
from data.storage.parquet_store import ParquetStore
from training.workspace import TrainingWorkspace

SCHEMA_VERSION = 2
OUTLIER_ROBUST_Z_THRESHOLD = 3.5

_CELL_COLUMNS = [
    "forecast_kind",
    "station_id",
    "init_time",
    "fxx",
    "valid_time",
    "forecast_c",
    "observed_c",
    "forecast_error_c",
    "absolute_error_c",
    "source_kind",
    "is_extrapolated",
    "has_observation",
    "forecast_is_valid",
]


@dataclass(frozen=True)
class DecoderBaselineMonth:
    cells: pd.DataFrame
    per_init: pd.DataFrame
    by_fxx: pd.DataFrame
    summary: dict


def verify_decoder_baseline(
    run_dir: Path | str,
    station_id: str,
    *,
    month: str | None = None,
) -> dict[str, Any]:
    """Generate persisted HRRR decoder baseline artifacts for one workspace."""
    started_at = datetime.now(timezone.utc).isoformat()
    workspace = TrainingWorkspace(run_dir)
    decoder_dir = workspace.data_store_dir / "_decoder_cache" / station_id
    available = sorted(path.stem for path in decoder_dir.glob("*.parquet"))
    months = [month] if month is not None else available
    missing = [value for value in months if value not in available]
    if missing:
        raise FileNotFoundError(
            f"decoder baseline month files not found for {station_id}: {missing}"
        )

    output_dir = workspace.artifacts_dir / "decoder_baseline" / station_id
    cells_dir = output_dir / "cells"
    per_init_dir = output_dir / "per_init"
    by_fxx_dir = output_dir / "by_fxx"
    for path in (cells_dir, per_init_dir, by_fxx_dir):
        path.mkdir(parents=True, exist_ok=True)

    obs_store = ParquetStore(workspace.data_store_dir / "obs")
    summaries = []
    input_fingerprints: dict[str, Any] = {}
    for value in months:
        decoder_path = decoder_dir / f"{value}.parquet"
        decoder = pd.read_parquet(decoder_path)
        observations = _read_observations_for_decoder(
            obs_store,
            station_id,
            decoder,
        )
        result = build_month_baseline(
            decoder,
            observations,
            station_id=station_id,
            month=value,
        )
        _write_parquet_atomic(result.cells, cells_dir / f"{value}.parquet")
        _write_parquet_atomic(result.per_init, per_init_dir / f"{value}.parquet")
        _write_parquet_atomic(result.by_fxx, by_fxx_dir / f"{value}.parquet")
        summaries.append(result.summary)
        input_fingerprints[value] = {
            "decoder": _path_fingerprint(decoder_path, workspace.run_dir),
            "observations": [
                _path_fingerprint(path, workspace.run_dir)
                for path in _observation_paths_for_decoder(
                    workspace.data_store_dir / "obs" / station_id,
                    decoder,
                )
            ],
        }

    summary_path = output_dir / "monthly_summary.parquet"
    existing = (
        pd.read_parquet(summary_path)
        if summary_path.exists()
        else pd.DataFrame()
    )
    updated = pd.concat([existing, pd.DataFrame(summaries)], ignore_index=True)
    if not updated.empty:
        updated = (
            updated.drop_duplicates("month", keep="last")
            .sort_values("month")
            .reset_index(drop=True)
        )
    _write_parquet_atomic(updated, summary_path)

    manifest_path = output_dir / "manifest.json"
    old_manifest = (
        workspace.read_json(manifest_path)
        if manifest_path.exists()
        else {}
    )
    all_fingerprints = dict(old_manifest.get("input_fingerprints") or {})
    all_fingerprints.update(input_fingerprints)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "forecast_kind": "hrrr_raw",
        "station_id": station_id,
        "error_definition": "forecast_c - observed_c",
        "outlier_robust_z_threshold": OUTLIER_ROBUST_Z_THRESHOLD,
        "months": updated["month"].tolist() if "month" in updated.columns else [],
        "input_fingerprints": all_fingerprints,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    workspace.write_json(manifest_path, manifest)
    result_summary = {
        "status": "verified",
        "station": station_id,
        "processed_months": months,
        "artifact_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    workspace.record_stage("decoder_baseline", result_summary)
    return result_summary


def build_month_baseline(
    decoder: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    station_id: str,
    month: str,
) -> DecoderBaselineMonth:
    """Build cell, per-init, and per-fxx HRRR baseline metrics for one UTC month."""
    work = decoder.copy()
    structural_required = {"init_time", "fxx", "valid_time"}
    if work.empty or not structural_required.issubset(work.columns):
        empty = pd.DataFrame(columns=_CELL_COLUMNS)
        return DecoderBaselineMonth(empty, pd.DataFrame(), pd.DataFrame(), _empty_summary(month))
    if "TMP_2m" not in work.columns:
        work["TMP_2m"] = np.nan

    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True).dt.floor("h")
    work["fxx"] = pd.to_numeric(work["fxx"], errors="coerce")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    work = work[
        (work["init_time"] >= month_start)
        & (work["init_time"] < month_end)
        & work["fxx"].between(1, 48)
    ].copy()
    if work.empty:
        empty = pd.DataFrame(columns=_CELL_COLUMNS)
        return DecoderBaselineMonth(empty, pd.DataFrame(), pd.DataFrame(), _empty_summary(month))

    work["fxx"] = work["fxx"].astype(int)
    duplicate_counts = (
        work.groupby(["init_time", "fxx"], sort=False).size().sub(1).clip(lower=0)
    )
    duplicates_by_init = duplicate_counts.groupby("init_time").sum()
    work = (
        work.sort_values(["init_time", "fxx", "valid_time"])
        .drop_duplicates(["init_time", "fxx"], keep="last")
        .reset_index(drop=True)
    )
    work["forecast_c"] = pd.to_numeric(work["TMP_2m"], errors="coerce")
    work["forecast_is_valid"] = np.isfinite(work["forecast_c"])
    if "is_extrapolated" not in work.columns:
        work["is_extrapolated"] = False
    work["is_extrapolated"] = work["is_extrapolated"].fillna(False).astype(bool)
    work["source_kind"] = [
        _source_kind(init, fxx, extrapolated)
        for init, fxx, extrapolated in zip(
            work["init_time"],
            work["fxx"],
            work["is_extrapolated"],
            strict=True,
        )
    ]

    obs = observations.copy()
    if obs.empty or not {"valid_time", "raw_tmpc_tenths"}.issubset(obs.columns):
        obs_hourly = pd.DataFrame(columns=["valid_time", "observed_c"])
    else:
        obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True).dt.floor("h")
        obs["observed_c"] = pd.to_numeric(obs["raw_tmpc_tenths"], errors="coerce")
        obs_hourly = (
            obs.sort_values("valid_time")
            .drop_duplicates("valid_time", keep="last")
            [["valid_time", "observed_c"]]
        )
    cells = work.merge(obs_hourly, on="valid_time", how="left")
    cells["has_observation"] = np.isfinite(cells["observed_c"])
    cells["forecast_error_c"] = cells["forecast_c"] - cells["observed_c"]
    cells["absolute_error_c"] = cells["forecast_error_c"].abs()
    cells["forecast_kind"] = "hrrr_raw"
    cells["station_id"] = station_id
    cells = cells[_CELL_COLUMNS].sort_values(["init_time", "fxx"]).reset_index(drop=True)

    per_init = _per_init_metrics(cells, duplicates_by_init)
    per_init = _add_outlier_scores(per_init)
    per_init.insert(0, "month", month)
    per_init.insert(0, "station_id", station_id)
    per_init.insert(0, "forecast_kind", "hrrr_raw")
    eligible_inits = set(
        per_init.loc[per_init["data_status"] == "complete", "init_time"]
    )
    eligible_cells = cells[cells["init_time"].isin(eligible_inits)].copy()
    by_fxx = _by_fxx_metrics(eligible_cells, month, station_id)
    summary = _monthly_summary(per_init, eligible_cells, month, station_id)
    return DecoderBaselineMonth(cells, per_init, by_fxx, summary)


def _per_init_metrics(
    cells: pd.DataFrame,
    duplicates_by_init: pd.Series,
) -> pd.DataFrame:
    rows = []
    expected = set(range(1, 49))
    for init_time, group in cells.groupby("init_time", sort=True):
        actual = set(group["fxx"].astype(int))
        duplicate_steps = int(duplicates_by_init.get(init_time, 0))
        missing_steps = len(expected - actual)
        invalid_forecast_steps = int((~group["forecast_is_valid"]).sum())
        matched = group[group["forecast_is_valid"] & group["has_observation"]]
        matched_steps = len(matched)
        if duplicate_steps or missing_steps or invalid_forecast_steps:
            status = "invalid_decoder"
        elif matched_steps < 48:
            status = "incomplete_observations"
        else:
            status = "complete"
        errors = matched["forecast_error_c"].astype(float)
        rows.append(
            {
                "init_time": init_time,
                "expected_steps": 48,
                "matched_steps": matched_steps,
                "missing_observation_steps": 48 - matched_steps,
                "missing_forecast_steps": missing_steps,
                "duplicate_steps": duplicate_steps,
                "invalid_forecast_steps": invalid_forecast_steps,
                "extrapolated_steps": int(group["is_extrapolated"].sum()),
                "bias_c": float(errors.mean()) if not errors.empty else np.nan,
                "mae_c": float(errors.abs().mean()) if not errors.empty else np.nan,
                "rmse_c": float(np.sqrt(np.mean(np.square(errors)))) if not errors.empty else np.nan,
                "error_variance_c2": float(np.var(errors, ddof=0)) if not errors.empty else np.nan,
                "error_std_c": float(np.std(errors, ddof=0)) if not errors.empty else np.nan,
                "max_absolute_error_c": float(errors.abs().max()) if not errors.empty else np.nan,
                "data_status": status,
            }
        )
    return pd.DataFrame(rows)


def _add_outlier_scores(per_init: pd.DataFrame) -> pd.DataFrame:
    result = per_init.copy()
    complete = result["data_status"] == "complete"
    for metric, output in (
        ("mae_c", "mae_robust_z"),
        ("error_std_c", "error_std_robust_z"),
    ):
        values = result.loc[complete, metric]
        median = values.median()
        mad = (values - median).abs().median()
        result[output] = np.nan
        if pd.notna(mad) and mad > 0:
            result.loc[complete, output] = 0.6745 * (values - median) / mad
    result["performance_outlier"] = complete & (
        (result["mae_robust_z"] > OUTLIER_ROBUST_Z_THRESHOLD)
        | (result["error_std_robust_z"] > OUTLIER_ROBUST_Z_THRESHOLD)
    )
    return result


def _by_fxx_metrics(
    cells: pd.DataFrame,
    month: str,
    station_id: str,
) -> pd.DataFrame:
    rows = []
    for fxx, group in cells.groupby("fxx", sort=True):
        errors = group["forecast_error_c"].astype(float)
        rows.append(
            {
                "forecast_kind": "hrrr_raw",
                "station_id": station_id,
                "month": month,
                "fxx": int(fxx),
                "sample_count": len(errors),
                **_error_metrics(errors),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "forecast_kind",
                "station_id",
                "month",
                "fxx",
                "sample_count",
                "bias_c",
                "mae_c",
                "rmse_c",
                "error_variance_c2",
                "error_std_c",
                "max_absolute_error_c",
                "mae_robust_z",
                "error_std_robust_z",
                "performance_outlier",
            ]
        )
    for metric, output in (
        ("mae_c", "mae_robust_z"),
        ("error_std_c", "error_std_robust_z"),
    ):
        values = result[metric]
        median = values.median()
        mad = (values - median).abs().median()
        result[output] = np.nan
        if pd.notna(mad) and mad > 0:
            result[output] = 0.6745 * (values - median) / mad
    result["performance_outlier"] = (
        (result["mae_robust_z"] > OUTLIER_ROBUST_Z_THRESHOLD)
        | (result["error_std_robust_z"] > OUTLIER_ROBUST_Z_THRESHOLD)
    )
    return result


def _monthly_summary(
    per_init: pd.DataFrame,
    cells: pd.DataFrame,
    month: str,
    station_id: str,
) -> dict:
    complete = per_init[per_init["data_status"] == "complete"]
    errors = cells["forecast_error_c"].astype(float)
    return {
        "schema_version": SCHEMA_VERSION,
        "forecast_kind": "hrrr_raw",
        "station_id": station_id,
        "month": month,
        "init_count": len(per_init),
        "complete_init_count": len(complete),
        "incomplete_observation_init_count": int(
            (per_init["data_status"] == "incomplete_observations").sum()
        ),
        "invalid_decoder_init_count": int(
            (per_init["data_status"] == "invalid_decoder").sum()
        ),
        "performance_outlier_count": int(per_init["performance_outlier"].sum()),
        **_error_metrics(errors),
        "per_init_mae_p50_c": _quantile(complete["mae_c"], 0.50),
        "per_init_mae_p90_c": _quantile(complete["mae_c"], 0.90),
        "per_init_mae_p95_c": _quantile(complete["mae_c"], 0.95),
        "per_init_mae_p99_c": _quantile(complete["mae_c"], 0.99),
        "per_init_mae_max_c": float(complete["mae_c"].max()) if not complete.empty else np.nan,
    }


def _error_metrics(errors: pd.Series) -> dict[str, float]:
    if errors.empty:
        return {
            "bias_c": np.nan,
            "mae_c": np.nan,
            "rmse_c": np.nan,
            "error_variance_c2": np.nan,
            "error_std_c": np.nan,
            "max_absolute_error_c": np.nan,
        }
    return {
        "bias_c": float(errors.mean()),
        "mae_c": float(errors.abs().mean()),
        "rmse_c": float(np.sqrt(np.mean(np.square(errors)))),
        "error_variance_c2": float(np.var(errors, ddof=0)),
        "error_std_c": float(np.std(errors, ddof=0)),
        "max_absolute_error_c": float(errors.abs().max()),
    }


def _quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if not values.empty else np.nan


def _source_kind(
    init_time: pd.Timestamp,
    fxx: int,
    is_extrapolated: bool,
) -> str:
    if is_extrapolated:
        return "extrapolated"
    if pd.Timestamp(init_time).hour in MAJOR_HOURS or int(fxx) <= 18:
        return "native"
    return "spliced_prior_major"


def _empty_summary(month: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "forecast_kind": "hrrr_raw",
        "month": month,
        "init_count": 0,
        "complete_init_count": 0,
        "incomplete_observation_init_count": 0,
        "invalid_decoder_init_count": 0,
        "performance_outlier_count": 0,
    }


def _read_observations_for_decoder(
    store: ParquetStore,
    station_id: str,
    decoder: pd.DataFrame,
) -> pd.DataFrame:
    if decoder.empty or "valid_time" not in decoder.columns:
        return pd.DataFrame()
    valid_times = pd.to_datetime(decoder["valid_time"], utc=True)
    return store.read(station_id, valid_times.min(), valid_times.max())


def _observation_paths_for_decoder(
    station_dir: Path,
    decoder: pd.DataFrame,
) -> list[Path]:
    if decoder.empty or "valid_time" not in decoder.columns:
        return []
    valid_times = pd.to_datetime(decoder["valid_time"], utc=True)
    start = valid_times.min()
    end = valid_times.max()
    paths = []
    for path in sorted(station_dir.glob("*.parquet")):
        year_text, quarter_text = path.stem.split("-Q", 1)
        quarter_start = pd.Timestamp(
            year=int(year_text),
            month=(int(quarter_text) - 1) * 3 + 1,
            day=1,
            tz="UTC",
        )
        quarter_end = quarter_start + pd.offsets.QuarterBegin(startingMonth=1)
        if quarter_start <= end and quarter_end > start:
            paths.append(path)
    return paths


def _path_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp_path, index=False)
    temp_path.replace(path)
