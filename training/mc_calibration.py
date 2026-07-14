"""Fit and evaluate checkpoint-bound Monte Carlo calibration artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.mc import McSampler
from model.mc_calibration import FXX_BANDS, McCalibration


_NORMAL_P90 = 1.2815515655446004
_REQUIRED_COLUMNS = {
    "init_time",
    "valid_time",
    "fxx",
    "target",
    "p10",
    "p50",
    "p90",
    "base_temp_c",
    "weight",
}


def local_calendar_day_groups(
    frame: pd.DataFrame,
    station_timezone: str,
) -> list[pd.DataFrame]:
    """Return complete right-closed local-day windows used by daily-high MC."""
    work = _validated_frame(frame)
    work["target_date"] = (
        (work["valid_time"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(station_timezone)
        .dt.date.astype(str)
    )
    groups: list[pd.DataFrame] = []
    for (_init_time, target_date), group in work.groupby(
        ["init_time", "target_date"],
        sort=True,
    ):
        start = pd.Timestamp(target_date, tz=station_timezone)
        end = start + pd.DateOffset(days=1)
        expected = pd.date_range(
            start,
            end,
            freq="h",
            inclusive="right",
        ).tz_convert("UTC")
        actual = pd.DatetimeIndex(group["valid_time"].sort_values().unique())
        if len(actual) != len(expected) or not actual.equals(expected):
            continue
        groups.append(group.sort_values("valid_time").reset_index(drop=True))
    return groups


def fit_mc_calibration(
    frame: pd.DataFrame,
    checkpoint_path: Path | str,
    *,
    station_timezone: str,
    mc_samples: int = 2_000,
    shrinkage: float = 0.1,
) -> McCalibration:
    """Fit sigma scales, standardized-error correlation, and max bias."""
    work = _validated_frame(frame)
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    if not 0.0 < shrinkage < 1.0:
        raise ValueError("shrinkage must be between zero and one")

    sigma = _raw_sigma(work)
    error = work["target"].to_numpy(dtype=float) - work["p50"].to_numpy(
        dtype=float
    )
    global_scale = _coverage_scale(error, sigma)
    scales = []
    for start, end in FXX_BANDS:
        mask = work["fxx"].between(start, end).to_numpy()
        scales.append(
            _coverage_scale(error[mask], sigma[mask])
            if int(mask.sum()) >= 30
            else global_scale
        )
    sigma_scales = np.asarray(scales, dtype=float)
    row_scales = np.array(
        [sigma_scales[_band_index(int(fxx))] for fxx in work["fxx"]],
        dtype=float,
    )
    standardized = error / (sigma * row_scales)
    matrix = (
        work.assign(standardized_error=standardized)
        .pivot_table(
            index="init_time",
            columns="fxx",
            values="standardized_error",
            aggfunc="mean",
        )
        .reindex(columns=range(1, 49))
    )
    correlation = matrix.corr(min_periods=10).to_numpy(dtype=float)
    correlation = _regularize_correlation(correlation, shrinkage)

    provisional = McCalibration(
        correlation=correlation,
        sigma_scales=sigma_scales,
        max_bias_f=0.0,
        model_sha256=McCalibration.sha256(checkpoint_path),
        fit_split="val",
        fit_start=work["init_time"].min().date().isoformat(),
        fit_end=work["init_time"].max().date().isoformat(),
        sample_count=int(work["init_time"].nunique()),
        station_timezone=station_timezone,
    )
    max_bias_f = _fit_max_bias(
        work,
        provisional,
        mc_samples,
        station_timezone=station_timezone,
    )
    return McCalibration(
        correlation=correlation,
        sigma_scales=sigma_scales,
        max_bias_f=max_bias_f,
        model_sha256=provisional.model_sha256,
        fit_split=provisional.fit_split,
        fit_start=provisional.fit_start,
        fit_end=provisional.fit_end,
        sample_count=provisional.sample_count,
        station_timezone=station_timezone,
    )


def cross_validate_mc_calibration(
    frame: pd.DataFrame,
    checkpoint_path: Path | str,
    *,
    station_timezone: str,
    folds: int = 4,
    mc_samples: int = 2_000,
) -> dict[str, float | int]:
    """Evaluate calibration with contiguous cycle folds held out once each."""
    return evaluate_mc_calibration_oof(
        frame,
        checkpoint_path,
        station_timezone=station_timezone,
        folds=folds,
        mc_samples=mc_samples,
    )[0]


def evaluate_mc_calibration_oof(
    frame: pd.DataFrame,
    checkpoint_path: Path | str,
    *,
    station_timezone: str,
    folds: int = 4,
    mc_samples: int = 2_000,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Return contiguous-fold local-day MC summary and cycle-day predictions."""
    work = _validated_frame(frame)
    init_times = np.array(sorted(work["init_time"].unique()))
    if folds < 2 or folds > len(init_times):
        raise ValueError("folds must be between 2 and the number of cycles")

    rows: list[dict[str, Any]] = []
    for held_out in np.array_split(init_times, folds):
        held_set = set(held_out.tolist())
        train = work[~work["init_time"].isin(held_set)]
        validation = work[work["init_time"].isin(held_set)]
        calibration = fit_mc_calibration(
            train,
            checkpoint_path,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
        )
        sampler = McSampler(n_samp=mc_samples, rng_seed=42)
        sampler.set_calibration(calibration)
        for group in local_calendar_day_groups(
            validation,
            station_timezone,
        ):
            actual_high_f, samples = _sample_cycle(group, sampler)
            p10, p50, p90 = np.quantile(samples, [0.1, 0.5, 0.9])
            base = group["base_temp_c"].to_numpy(dtype=float)
            tft = base + group["p50"].to_numpy(dtype=float)
            rows.append(
                {
                    "init_time": group["init_time"].iloc[0],
                    "target_date": group["target_date"].iloc[0],
                    "actual_high_f": actual_high_f,
                    "hrrr_high_f": float(base.max() * 1.8 + 32.0),
                    "tft_high_f": float(tft.max() * 1.8 + 32.0),
                    "mc_p10_f": float(p10),
                    "mc_p50_f": float(p50),
                    "mc_p90_f": float(p90),
                    "min_fxx": int(group["fxx"].min()),
                    "max_fxx": int(group["fxx"].max()),
                }
            )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("MC calibration has no complete local calendar days")
    covered = (
        (predictions["actual_high_f"] >= predictions["mc_p10_f"])
        & (predictions["actual_high_f"] <= predictions["mc_p90_f"])
    )
    summary = {
        "folds": folds,
        "cycles": len(predictions),
        "p10_p90_coverage": float(covered.mean()),
        "mean_interval_width_f": float(
            (predictions["mc_p90_f"] - predictions["mc_p10_f"]).mean()
        ),
        "daily_high_mae_f": float(
            (predictions["mc_p50_f"] - predictions["actual_high_f"]).abs().mean()
        ),
    }
    return summary, predictions


def _validated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "MC calibration prediction frame missing columns: "
            + ", ".join(missing)
        )
    work = frame.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    work["fxx"] = work["fxx"].astype(int)
    work = work[work["weight"].astype(float) > 0].copy()
    extrapolated = work.get(
        "is_extrapolated",
        pd.Series(False, index=work.index),
    ).fillna(False).astype(bool)
    work = work[~extrapolated].copy()
    if work.empty:
        raise ValueError("MC calibration has no trusted target cells")
    if not work["fxx"].between(1, 48).all():
        raise ValueError("MC calibration fxx must be in 1..48")
    numeric = ["target", "p10", "p50", "p90", "base_temp_c"]
    if not np.isfinite(work[numeric].to_numpy(dtype=float)).all():
        raise ValueError("MC calibration prediction frame contains non-finite values")
    if (work["p90"] <= work["p10"]).any():
        raise ValueError("MC calibration requires p90 greater than p10")
    return work.sort_values(["init_time", "fxx"]).reset_index(drop=True)


def _raw_sigma(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["p90"] - frame["p10"]).to_numpy(dtype=float)
        / (2.0 * _NORMAL_P90)
    )


def _coverage_scale(error: np.ndarray, sigma: np.ndarray) -> float:
    ratio = np.abs(error) / np.clip(sigma, 0.01, None)
    scale = float(np.quantile(ratio, 0.8) / _NORMAL_P90)
    return float(np.clip(scale, 0.25, 4.0))


def _regularize_correlation(
    correlation: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    matrix = np.asarray(correlation, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    matrix = (1.0 - shrinkage) * matrix + shrinkage * np.eye(48)
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.clip(values, 1e-6, None)) @ vectors.T
    diagonal = np.sqrt(np.diag(matrix))
    matrix = matrix / np.outer(diagonal, diagonal)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _fit_max_bias(
    frame: pd.DataFrame,
    calibration: McCalibration,
    mc_samples: int,
    *,
    station_timezone: str,
) -> float:
    sampler = McSampler(n_samp=mc_samples, rng_seed=42)
    sampler.set_correlation(calibration.correlation)
    offsets = []
    for group in local_calendar_day_groups(frame, station_timezone):
        actual_high_f, samples = _sample_cycle(
            group,
            sampler,
            sigma_scales=calibration.sigma_scales,
        )
        offsets.append(float(np.median(samples)) - actual_high_f)
    if not offsets:
        raise ValueError("MC calibration has no complete local calendar days")
    return float(np.median(offsets))


def _sample_cycle(
    group: pd.DataFrame,
    sampler: McSampler,
    *,
    sigma_scales: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    ordered = group.sort_values("fxx")
    indices = [int(fxx) - 1 for fxx in ordered["fxx"]]
    sigma = _raw_sigma(ordered)
    if sigma_scales is not None:
        sigma = sigma * np.array(
            [sigma_scales[_band_index(index + 1)] for index in indices]
        )
    predicted_c = (
        ordered["base_temp_c"].to_numpy(dtype=float)
        + ordered["p50"].to_numpy(dtype=float)
    )
    actual_c = (
        ordered["base_temp_c"].to_numpy(dtype=float)
        + ordered["target"].to_numpy(dtype=float)
    )
    samples = sampler.sample_daily_high(
        predicted_c,
        sigma,
        hour_indices=indices,
    )
    actual_high_f = float(actual_c.max() * 1.8 + 32.0)
    return actual_high_f, samples


def _band_index(fxx: int) -> int:
    for index, (start, end) in enumerate(FXX_BANDS):
        if start <= fxx <= end:
            return index
    raise ValueError(f"fxx must be in 1..48, got {fxx}")
