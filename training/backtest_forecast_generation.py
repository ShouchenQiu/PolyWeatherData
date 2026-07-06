"""Convert audited training evidence into replay-ready forecast artifacts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.forecast_cache import ForecastCache
from model.daily_high import (
    IncompleteDailyHighCoverage,
    PartialDayBiasCalibration,
    build_daily_high_distribution,
    fit_partial_day_bias,
)
from model.mc import McSampler
from model.mc_calibration import McCalibration
from training.mc_calibration import (
    fit_mc_calibration,
    local_calendar_day_groups,
)


_NORMAL_P90 = 1.2815515655446004


def contiguous_oof_assignments(
    init_times: object,
    *,
    folds: int,
) -> pd.DataFrame:
    """Assign sorted forecast cycles to one contiguous held-out fold each."""
    cycles = pd.DatetimeIndex(pd.to_datetime(list(init_times), utc=True)).unique()
    cycles = cycles.sort_values()
    if folds < 2 or folds > len(cycles):
        raise ValueError("folds must be between 2 and the number of cycles")
    rows: list[dict] = []
    for fold, held_out in enumerate(np.array_split(cycles.to_numpy(), folds)):
        rows.extend(
            {
                "init_time": pd.Timestamp(init_time),
                "lineage_id": f"oof-fold-{fold}",
            }
            for init_time in held_out
        )
    return pd.DataFrame(rows).sort_values("init_time").reset_index(drop=True)


def fit_oof_calibrations(
    hourly_predictions: pd.DataFrame,
    *,
    observations: pd.DataFrame,
    checkpoint_path: Path | str,
    station_timezone: str,
    folds: int = 4,
    mc_samples: int = 2_000,
    fit_calibration: Callable[..., McCalibration] = fit_mc_calibration,
    fit_partial_bias: Callable[..., PartialDayBiasCalibration] = (
        fit_partial_day_bias
    ),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, McCalibration],
    dict[str, PartialDayBiasCalibration],
]:
    """Fit each calibration only on cycles outside its contiguous OOF fold."""
    _require_columns(hourly_predictions, {"init_time"}, "hourly_predictions")
    work = hourly_predictions.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    assignments = contiguous_oof_assignments(
        work["init_time"].unique(),
        folds=folds,
    )
    calibration_by_lineage: dict[str, McCalibration] = {}
    bias_by_lineage: dict[str, PartialDayBiasCalibration] = {}
    lineage_rows: list[dict[str, Any]] = []
    checkpoint_sha256 = McCalibration.sha256(checkpoint_path)
    for lineage_id in assignments["lineage_id"].drop_duplicates():
        held_out = set(
            assignments.loc[
                assignments["lineage_id"] == lineage_id,
                "init_time",
            ]
        )
        training = work[~work["init_time"].isin(held_out)].copy()
        calibration = fit_calibration(
            training,
            checkpoint_path,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
        )
        if calibration.model_sha256 != checkpoint_sha256:
            raise ValueError(
                f"OOF calibration {lineage_id} is bound to another checkpoint"
            )
        calibration_by_lineage[str(lineage_id)] = calibration
        partial_bias = fit_partial_bias(
            hourly_predictions=training,
            observations=observations,
            calibration=calibration,
            station_timezone=station_timezone,
            mc_samples=mc_samples,
            seed=42,
        )
        bias_by_lineage[str(lineage_id)] = partial_bias
        lineage_rows.append(
            {
                "lineage_id": str(lineage_id),
                "checkpoint_sha256": checkpoint_sha256,
                "calibration_sha256": _calibration_sha256(
                    calibration,
                    partial_bias,
                ),
                "split_role": "oof_development",
                "mc_seed": 42,
                "mc_samples": int(mc_samples),
            }
        )
    return (
        pd.DataFrame(lineage_rows),
        assignments,
        calibration_by_lineage,
        bias_by_lineage,
    )


def build_forecast_cache(
    *,
    hourly_predictions: pd.DataFrame,
    observations: pd.DataFrame,
    availability: pd.DataFrame,
    markets: pd.DataFrame,
    lineages: pd.DataFrame,
    cycle_lineages: pd.DataFrame,
    calibrations: Mapping[str, McCalibration],
    partial_biases: Mapping[str, PartialDayBiasCalibration],
    station_id: str,
    station_timezone: str,
    target_start: str | None = None,
    target_end: str | None = None,
) -> ForecastCache:
    """Build fully covered point-in-time local-day forecast snapshots."""
    availability_by_init = _unique_index(
        availability,
        required={"init_time", "complete", "available_at"},
        name="availability",
    )
    lineage_by_init = _unique_index(
        cycle_lineages,
        required={"init_time", "lineage_id"},
        name="cycle_lineages",
    )
    market_columns = {
        "target_date",
        "condition_id",
        "bucket_lo_f",
        "bucket_hi_f",
        "event_end_at",
    }
    _require_columns(markets, market_columns, "markets")
    lineage_rows = lineages.set_index("lineage_id", drop=False)

    snapshot_rows: list[dict] = []
    probability_rows: list[dict] = []
    predictions = hourly_predictions.copy()
    predictions["init_time"] = pd.to_datetime(
        predictions["init_time"], utc=True
    )
    predictions["valid_time"] = pd.to_datetime(
        predictions["valid_time"], utc=True
    )
    target_markets = markets.copy()
    target_markets["target_date"] = pd.to_datetime(
        target_markets["target_date"]
    ).dt.date.astype(str)
    target_markets["event_end_at"] = pd.to_datetime(
        target_markets["event_end_at"], utc=True
    )
    if target_start is not None:
        target_markets = target_markets[
            target_markets["target_date"] >= str(target_start)
        ]
    if target_end is not None:
        target_markets = target_markets[
            target_markets["target_date"] <= str(target_end)
        ]
    for init_time, init_predictions in predictions.groupby(
        "init_time", sort=True
    ):
        init_time = _utc(init_time)
        if init_time not in availability_by_init.index:
            raise ValueError(
                f"availability missing forecast init_time={init_time.isoformat()}"
            )
        available = availability_by_init.loc[init_time]
        if not bool(available["complete"]):
            continue
        if init_time not in lineage_by_init.index:
            raise ValueError(
                f"cycle_lineages missing init_time={init_time.isoformat()}"
            )
        lineage_id = str(lineage_by_init.loc[init_time, "lineage_id"])
        if (
            lineage_id not in calibrations
            or lineage_id not in partial_biases
            or lineage_id not in lineage_rows.index
        ):
            raise ValueError(f"missing calibration lineage {lineage_id!r}")
        lineage = lineage_rows.loc[lineage_id]
        calibration = calibrations[lineage_id]
        if calibration.model_sha256 != str(lineage["checkpoint_sha256"]):
            raise ValueError(
                f"calibration/checkpoint mismatch for lineage {lineage_id!r}"
            )

        for target_date, day_markets in target_markets.groupby(
            "target_date", sort=True
        ):
            available_at = _utc(available["available_at"])
            target_day_end = (
                pd.Timestamp(str(target_date))
                .tz_localize(ZoneInfo(station_timezone))
                + pd.DateOffset(days=1)
            ).tz_convert("UTC")
            if available_at >= target_day_end:
                continue
            sampler = McSampler(
                n_samp=int(lineage["mc_samples"]),
                rng_seed=int(lineage["mc_seed"]),
            )
            sampler.set_calibration(calibration)
            try:
                distribution = build_daily_high_distribution(
                    init_time=init_time,
                    target_date=str(target_date),
                    station_timezone=station_timezone,
                    predictions=init_predictions,
                    observations=observations,
                    sampler=sampler,
                    bias=partial_biases[lineage_id],
                )
            except IncompleteDailyHighCoverage:
                continue
            daily_highs = distribution.daily_high_samples_f
            resolved_highs = np.floor(daily_highs + 0.5)
            snapshot_id = _snapshot_id(
                station_id,
                init_time,
                str(target_date),
                lineage_id,
            )
            p10, p50, p90 = np.quantile(daily_highs, [0.1, 0.5, 0.9])
            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "station_id": station_id,
                    "init_time": init_time,
                    "target_date": str(target_date),
                    "available_at": available_at,
                    "p10_f": float(p10),
                    "p50_f": float(p50),
                    "p90_f": float(p90),
                    "lineage_id": lineage_id,
                    "coverage_type": distribution.coverage_type,
                    "observed_high_f": distribution.observed_high_f,
                    "observed_hour_count": (
                        distribution.observed_hour_count
                    ),
                    "observed_through": distribution.observed_through,
                    "forecast_hour_count": (
                        distribution.forecast_hour_count
                    ),
                    "forecast_start_at": distribution.forecast_start_at,
                    "forecast_through": distribution.forecast_through,
                    "target_hour_count": distribution.target_hour_count,
                }
            )
            for market in day_markets.itertuples(index=False):
                lo_f = float(market.bucket_lo_f)
                hi_f = float(market.bucket_hi_f)
                if lo_f >= hi_f:
                    raise ValueError(
                        f"invalid market bucket for {market.condition_id}: "
                        f"[{lo_f}, {hi_f})"
                    )
                probability_rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "condition_id": str(market.condition_id),
                        "yes_probability": float(
                            (
                                (resolved_highs >= lo_f)
                                & (resolved_highs < hi_f)
                            ).mean()
                        ),
                    }
                )

    snapshots = pd.DataFrame(
        snapshot_rows,
        columns=[
            "snapshot_id",
            "station_id",
            "init_time",
            "target_date",
            "available_at",
            "p10_f",
            "p50_f",
            "p90_f",
            "lineage_id",
            "coverage_type",
            "observed_high_f",
            "observed_hour_count",
            "observed_through",
            "forecast_hour_count",
            "forecast_start_at",
            "forecast_through",
            "target_hour_count",
        ],
    )
    probabilities = pd.DataFrame(
        probability_rows,
        columns=["snapshot_id", "condition_id", "yes_probability"],
    )
    return ForecastCache(snapshots, probabilities, lineages)


def _unique_index(
    frame: pd.DataFrame,
    *,
    required: set[str],
    name: str,
) -> pd.DataFrame:
    _require_columns(frame, required, name)
    work = frame.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    if work["init_time"].duplicated().any():
        raise ValueError(f"{name} contains duplicate init_time")
    return work.set_index("init_time", drop=False)


def _prediction_calendar_day_groups(
    frame: pd.DataFrame,
    station_timezone: str,
) -> list[pd.DataFrame]:
    """Return complete local days without consulting validation-label weight."""
    required = {
        "init_time",
        "valid_time",
        "fxx",
        "p10",
        "p50",
        "p90",
        "base_temp_c",
    }
    _require_columns(frame, required, "hourly_predictions")
    work = frame.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    work["fxx"] = work["fxx"].astype(int)
    numeric = ["p10", "p50", "p90", "base_temp_c"]
    if not np.isfinite(work[numeric].to_numpy(dtype=float)).all():
        raise ValueError("hourly_predictions contain non-finite model outputs")
    if (work["p90"] <= work["p10"]).any():
        raise ValueError("hourly_predictions require p90 greater than p10")
    work["target_date"] = (
        work["valid_time"]
        .dt.tz_convert(station_timezone)
        .dt.date.astype(str)
    )
    groups: list[pd.DataFrame] = []
    for (_init_time, target_date), group in work.groupby(
        ["init_time", "target_date"],
        sort=True,
    ):
        start = pd.Timestamp(target_date, tz=station_timezone)
        expected = pd.date_range(
            start,
            start + pd.DateOffset(days=1),
            freq="h",
            inclusive="left",
        ).tz_convert("UTC")
        actual = pd.DatetimeIndex(group["valid_time"].sort_values().unique())
        if len(actual) == len(expected) and actual.equals(expected):
            groups.append(
                group.sort_values("valid_time").reset_index(drop=True)
            )
    return groups


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(missing)}")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _snapshot_id(
    station_id: str,
    init_time: pd.Timestamp,
    target_date: str,
    lineage_id: str,
) -> str:
    value = "|".join(
        [station_id, init_time.isoformat(), target_date, lineage_id]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _calibration_sha256(
    calibration: McCalibration,
    partial_bias: PartialDayBiasCalibration | None = None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(mc_calibration_sha256(calibration).encode("ascii"))
    if partial_bias is not None:
        hasher.update(partial_bias_sha256(partial_bias).encode("ascii"))
    return hasher.hexdigest()


def mc_calibration_sha256(calibration: McCalibration) -> str:
    """Hash one in-memory MC calibration deterministically."""
    hasher = hashlib.sha256()
    hasher.update(calibration.correlation.astype("<f8").tobytes())
    hasher.update(calibration.sigma_scales.astype("<f8").tobytes())
    metadata = {
        "max_bias_f": calibration.max_bias_f,
        "model_sha256": calibration.model_sha256,
        "fit_split": calibration.fit_split,
        "fit_start": calibration.fit_start,
        "fit_end": calibration.fit_end,
        "sample_count": calibration.sample_count,
        "station_timezone": calibration.station_timezone,
        "target_window": calibration.target_window,
    }
    hasher.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return hasher.hexdigest()


def partial_bias_sha256(
    partial_bias: PartialDayBiasCalibration,
) -> str:
    """Hash one remaining-hour bias calibration deterministically."""
    hasher = hashlib.sha256()
    hasher.update(
        np.asarray(
            partial_bias.bias_by_forecast_hours_f,
            dtype="<f8",
        ).tobytes()
    )
    hasher.update(
        np.asarray(partial_bias.sample_counts, dtype="<i8").tobytes()
    )
    hasher.update(str(partial_bias.prior_strength).encode("ascii"))
    return hasher.hexdigest()
