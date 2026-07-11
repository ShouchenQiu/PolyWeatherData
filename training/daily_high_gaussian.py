"""Leakage-safe future-maximum examples for the Gaussian daily-high pilot."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from model.daily_high_gaussian import (
    GaussianFutureMaxModel,
    GaussianFutureMaxNetwork,
    SUPPORT_MAX_F,
    SUPPORT_MIN_F,
    TruncatedNormal,
    compose_daily_high,
)
from training.daily_high_pmf import expanding_target_day_folds


FEATURE_COLUMNS: tuple[str, ...] = (
    "tft_future_peak_p10_f",
    "tft_future_peak_p50_f",
    "tft_future_peak_p90_f",
    "hrrr_future_peak_f",
    "tft_future_mean_p50_f",
    "tft_future_peak_spread_f",
    "future_peak_slot_fraction",
    "forecast_hour_fraction",
    "observed_high_f",
    "observed_high_mask",
    "observed_hour_fraction",
    "hours_to_settlement_scaled",
    "day_of_year_sin",
    "day_of_year_cos",
)


@dataclass(frozen=True)
class FutureMaxExamples:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def save(
        self,
        root: Path | str,
        *,
        source_sha256: dict[str, str],
    ) -> dict[str, object]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        frame_path = directory / "examples.parquet"
        self.frame.to_parquet(frame_path, index=False)
        dates = sorted(self.frame["target_date"].astype(str).unique())
        manifest = {
            "schema_version": 1,
            "data_class": "point_in_time_future_max_examples",
            "feature_columns": list(self.feature_columns),
            "feature_contract_sha256": _json_sha256(list(self.feature_columns)),
            "row_count": len(self.frame),
            "target_date_range": [dates[0], dates[-1]] if dates else [],
            "target_day_count": len(dates),
            "coverage_type_counts": {
                str(key): int(value)
                for key, value in self.frame["coverage_type"].value_counts().items()
            },
            "source_sha256": dict(sorted(source_sha256.items())),
            "sha256": {"examples.parquet": _file_sha256(frame_path)},
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
        )
        return manifest


@dataclass(frozen=True)
class GaussianEvaluation:
    snapshots: pd.DataFrame

    def save(
        self,
        root: Path | str,
        *,
        split_role: str,
        source_sha256: dict[str, str],
    ) -> dict[str, object]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        snapshots_path = directory / "snapshots.parquet"
        self.snapshots.to_parquet(snapshots_path, index=False)
        manifest = {
            "schema_version": 1,
            "data_class": "future_max_gaussian_evaluation",
            "distribution_source": "future_max_truncated_gaussian_pilot",
            "split_role": split_role,
            "row_count": len(self.snapshots),
            "source_sha256": dict(sorted(source_sha256.items())),
            "sha256": {"snapshots.parquet": _file_sha256(snapshots_path)},
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
        )
        return manifest


def fit_gaussian_future_max_model(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    tft_checkpoint_sha256: str,
    training_manifest_sha256: str,
    epochs: int = 80,
    seed: int = 42,
) -> GaussianFutureMaxModel:
    """Fit a linear truncated-normal future-maximum model on one training fold."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    required = set(feature_columns) | {"future_max_f"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Gaussian training frame missing: " + ", ".join(missing))
    values = frame[list(feature_columns)].to_numpy(dtype=np.float32)
    labels = frame["future_max_f"].to_numpy(dtype=np.float32)
    if len(values) == 0 or not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("Gaussian training values must be finite and non-empty")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (values - mean) / scale
    torch.manual_seed(seed)
    network = GaussianFutureMaxNetwork(len(feature_columns))
    with torch.no_grad():
        network.linear.weight.zero_()
        network.linear.bias[:] = torch.tensor([0.0, 1.0])
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-2, weight_decay=1e-3)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels)
    base = torch.from_numpy(frame["tft_future_peak_p50_f"].to_numpy(dtype=np.float32))
    for _ in range(epochs):
        optimizer.zero_grad()
        output = network(x)
        mu = base + output[:, 0]
        sigma = torch.nn.functional.softplus(output[:, 1]) + 0.25
        z = (y - mu) / sigma
        log_pdf = -0.5 * z.square() - torch.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        lower = torch.special.ndtr((SUPPORT_MIN_F - mu) / sigma)
        upper = torch.special.ndtr((SUPPORT_MAX_F - mu) / sigma)
        loss = -(log_pdf - torch.log((upper - lower).clamp_min(1e-12))).mean()
        loss.backward()
        optimizer.step()
    return GaussianFutureMaxModel(network=network, feature_columns=feature_columns, feature_mean=mean, feature_scale=scale, tft_checkpoint_sha256=tft_checkpoint_sha256, training_manifest_sha256=training_manifest_sha256)


def run_expanding_gaussian_oof(
    examples: FutureMaxExamples,
    *,
    tft_checkpoint_sha256: str,
    training_manifest_sha256: str,
    folds: int = 4,
    epochs: int = 80,
    seed: int = 42,
) -> GaussianEvaluation:
    rows: list[dict[str, object]] = []
    for fold, (train, validation) in enumerate(
        expanding_target_day_folds(examples.frame, folds=folds)
    ):
        model = fit_gaussian_future_max_model(
            train, feature_columns=examples.feature_columns,
            tft_checkpoint_sha256=tft_checkpoint_sha256,
            training_manifest_sha256=training_manifest_sha256,
            epochs=epochs, seed=seed + fold,
        )
        climatology = TruncatedNormal(
            float(train["future_max_f"].mean()),
            max(float(train["future_max_f"].std(ddof=0)), 0.5),
        )
        rows.extend(
            _evaluation_row(model, pd.Series(row._asdict()), fold, climatology)
            for row in validation.itertuples(index=False)
        )
    return GaussianEvaluation(pd.DataFrame(rows))


def predict_gaussian_examples(
    model: GaussianFutureMaxModel,
    examples: FutureMaxExamples,
    *,
    lineage_label: str,
) -> GaussianEvaluation:
    return GaussianEvaluation(
        pd.DataFrame([
            _evaluation_row(model, pd.Series(row._asdict()), lineage_label, None)
            for row in examples.frame.itertuples(index=False)
        ])
    )


def _evaluation_row(
    model: GaussianFutureMaxModel,
    row: pd.Series,
    fold: int | str,
    climatology: TruncatedNormal | None,
) -> dict[str, object]:
    future = model.predict_future(row)
    observed = float(row["observed_high_f"]) if float(row["observed_high_mask"]) > 0 else None
    daily = compose_daily_high(future, observed_high_f=observed)
    snapshot_id = hashlib.sha256(
        f"{row['init_time']}|{row['target_date']}|{fold}".encode("ascii")
    ).hexdigest()
    result: dict[str, object] = {
        "snapshot_id": snapshot_id,
        "fold": fold,
        "init_time": pd.Timestamp(row["init_time"]),
        "target_date": str(row["target_date"]),
        "coverage_type": str(row["coverage_type"]),
        "observed_high_f": float(row["observed_high_f"]),
        "observed_high_mask": float(row["observed_high_mask"]),
        "observed_hour_count": int(row["observed_hour_count"]),
        "forecast_hour_count": int(row["forecast_hour_count"]),
        "target_hour_count": int(row["target_hour_count"]),
        "future_max_f": float(row["future_max_f"]),
        "mean_f": future.mean_f,
        "sigma_f": future.sigma_f,
        "p10_f": daily.p10_f,
        "p50_f": daily.p50_f,
        "p90_f": daily.p90_f,
    }
    result["future_nll"] = _nll(future, float(row["future_max_f"]))
    if climatology is not None:
        result["climatology_future_nll"] = _nll(climatology, float(row["future_max_f"]))
    for optional in ("source_snapshot_id", "available_at", "observed_through", "forecast_start_at", "forecast_through"):
        if optional in row:
            result[optional] = row[optional]
    return result


def _nll(distribution: TruncatedNormal, value: float) -> float:
    sigma = distribution.sigma_f
    z = (value - distribution.mean_f) / sigma
    lower = _normal_cdf((distribution.lower_f - distribution.mean_f) / sigma)
    upper = _normal_cdf((distribution.upper_f - distribution.mean_f) / sigma)
    return float(0.5 * z * z + np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + np.log(upper - lower))


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / np.sqrt(2.0))))


def build_future_max_examples(
    *,
    hourly_predictions: pd.DataFrame,
    observations: pd.DataFrame,
    snapshots: pd.DataFrame,
    station_timezone: str,
) -> FutureMaxExamples:
    """Build examples labelled by the actual maximum after each init time."""
    _require(
        hourly_predictions,
        {"init_time", "valid_time", "fxx", "p10", "p50", "p90", "base_temp_c"},
        "hourly_predictions",
    )
    _require(observations, {"valid_time", "available_at", "temperature_c"}, "observations")
    _require(snapshots, {"init_time", "target_date"}, "snapshots")
    predictions = hourly_predictions.copy()
    predictions["init_time"] = pd.to_datetime(predictions["init_time"], utc=True)
    predictions["valid_time"] = pd.to_datetime(predictions["valid_time"], utc=True)
    obs = observations.copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs["available_at"] = pd.to_datetime(obs["available_at"], utc=True)
    all_truth = obs.sort_values("available_at").drop_duplicates("valid_time", keep="last")
    truth_by_time = all_truth.set_index("valid_time")["temperature_c"]
    cases = snapshots.copy()
    cases["init_time"] = pd.to_datetime(cases["init_time"], utc=True)
    cases["target_date"] = cases["target_date"].astype(str)
    if cases.duplicated(["init_time", "target_date"]).any():
        raise ValueError("snapshots contain duplicate init_time/target_date")
    by_init = {key: group for key, group in predictions.groupby("init_time", sort=False)}
    rows: list[dict[str, object]] = []
    for case in cases.itertuples(index=False):
        init = pd.Timestamp(case.init_time)
        if init not in by_init:
            continue
        target_date = str(case.target_date)
        expected = _target_hours(target_date, station_timezone)
        expected_set = set(expected)
        visible = obs[
            obs["valid_time"].isin(expected_set)
            & (obs["valid_time"] <= init)
            & (obs["available_at"] <= init)
        ].sort_values("valid_time").drop_duplicates("valid_time", keep="last")
        future = by_init[init].copy()
        extrapolated = future.get(
            "is_extrapolated", pd.Series(False, index=future.index)
        ).fillna(False).astype(bool)
        future = future[
            future["valid_time"].isin(expected_set)
            & (future["valid_time"] > init)
            & ~extrapolated
        ].sort_values("valid_time").drop_duplicates("valid_time", keep="last")
        if set(visible["valid_time"]) | set(future["valid_time"]) != expected_set:
            continue
        if future.empty or not set(future["valid_time"]).issubset(truth_by_time.index):
            continue
        base = future["base_temp_c"].to_numpy(dtype=float)
        p10 = (base + future["p10"].to_numpy(dtype=float)) * 1.8 + 32.0
        p50 = (base + future["p50"].to_numpy(dtype=float)) * 1.8 + 32.0
        p90 = (base + future["p90"].to_numpy(dtype=float)) * 1.8 + 32.0
        observed_values = visible["temperature_c"].to_numpy(dtype=float) * 1.8 + 32.0
        future_truth = truth_by_time.loc[list(future["valid_time"])].to_numpy(dtype=float) * 1.8 + 32.0
        peak_index = int(np.argmax(p50))
        angle = 2.0 * np.pi * float(pd.Timestamp(target_date).dayofyear) / 366.0
        row: dict[str, object] = {
            "init_time": init,
            "target_date": target_date,
            "future_max_f": float(future_truth.max()),
            "observed_high_f": float(observed_values.max()) if len(observed_values) else 0.0,
            "observed_high_mask": float(len(observed_values) > 0),
            "observed_hour_count": len(visible),
            "forecast_hour_count": len(future),
            "target_hour_count": len(expected),
            "coverage_type": "forecast_only" if visible.empty else "observed_plus_forecast",
            "tft_future_peak_p10_f": float(p10.max()),
            "tft_future_peak_p50_f": float(p50.max()),
            "tft_future_peak_p90_f": float(p90.max()),
            "hrrr_future_peak_f": float((base * 1.8 + 32.0).max()),
            "tft_future_mean_p50_f": float(p50.mean()),
            "tft_future_peak_spread_f": float(p90.max() - p10.max()),
            "future_peak_slot_fraction": peak_index / max(len(expected) - 1, 1),
            "forecast_hour_fraction": len(future) / len(expected),
            "observed_hour_fraction": len(visible) / len(expected),
            "hours_to_settlement_scaled": max(
                0.0, (expected[-1] - init).total_seconds() / 3600.0
            ) / 48.0,
            "day_of_year_sin": float(np.sin(angle)),
            "day_of_year_cos": float(np.cos(angle)),
        }
        for optional in ("snapshot_id", "available_at", "observed_through", "forecast_start_at", "forecast_through"):
            if hasattr(case, optional):
                row["source_snapshot_id" if optional == "snapshot_id" else optional] = getattr(case, optional)
        rows.append(row)
    return FutureMaxExamples(pd.DataFrame(rows))


def _target_hours(target_date: str, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(target_date).tz_localize(ZoneInfo(timezone))
    return pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="right").tz_convert("UTC")


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(missing)}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
