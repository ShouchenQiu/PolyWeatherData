"""Point-in-time local-calendar-day high temperature distributions."""
from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from model.mc import McSampler


_NORMAL_P90 = 1.2815515655446004


class IncompleteDailyHighCoverage(ValueError):
    """Raised when visible observations and trusted forecasts leave a gap."""


@dataclass(frozen=True)
class PartialDayBiasCalibration:
    """Daily-high max bias indexed by the number of future forecast hours."""

    bias_by_forecast_hours_f: tuple[float, ...]
    prior_strength: float = 20.0
    sample_counts: tuple[int, ...] = (0,) * 25

    def __post_init__(self) -> None:
        if len(self.bias_by_forecast_hours_f) != 25:
            raise ValueError(
                "bias_by_forecast_hours_f must contain indices 0..24"
            )
        values = np.asarray(self.bias_by_forecast_hours_f, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("partial-day bias values must be finite")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        if len(self.sample_counts) != 25 or any(
            int(value) < 0 for value in self.sample_counts
        ):
            raise ValueError("sample_counts must contain 25 non-negative values")

    @classmethod
    def zero(cls) -> "PartialDayBiasCalibration":
        return cls((0.0,) * 25)

    def for_forecast_hours(self, hours: int) -> float:
        if not 0 <= int(hours) <= 25:
            raise ValueError("forecast hours must be between 0 and 25")
        # A 25-hour DST fall-back day shares the 24-hour full-day prior bin.
        return float(self.bias_by_forecast_hours_f[min(int(hours), 24)])


def fit_partial_day_bias(
    *,
    hourly_predictions: pd.DataFrame,
    observations: pd.DataFrame,
    calibration: object,
    station_timezone: str,
    prior_strength: float = 20.0,
    mc_samples: int = 2_000,
    seed: int = 42,
) -> PartialDayBiasCalibration:
    """Fit future-maximum bias by remaining trusted forecast-hour count."""
    if prior_strength < 0 or mc_samples <= 0:
        raise ValueError("bias fit strengths and samples must be valid")
    required = {
        "init_time",
        "valid_time",
        "fxx",
        "p10",
        "p50",
        "p90",
        "base_temp_c",
    }
    missing = required - set(hourly_predictions)
    if missing:
        raise ValueError(
            "hourly predictions missing columns: "
            + ", ".join(sorted(missing))
        )
    work = hourly_predictions.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    extrapolated = work.get(
        "is_extrapolated",
        pd.Series(False, index=work.index),
    ).fillna(False).astype(bool)
    work = work[~extrapolated].copy()
    work["target_date"] = (
        (work["valid_time"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(station_timezone)
        .dt.date.astype(str)
    )
    obs = observations.copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs = obs[["valid_time", "temperature_c"]].drop_duplicates(
        "valid_time", keep="last"
    )
    offsets: dict[int, list[float]] = {
        hours: [] for hours in range(1, 25)
    }
    for group_index, (_, group) in enumerate(
        work.groupby(["init_time", "target_date"], sort=True)
    ):
        group = group.sort_values("valid_time")
        if not 1 <= len(group) <= 25:
            continue
        truth = group[["valid_time"]].merge(
            obs,
            on="valid_time",
            how="inner",
            validate="one_to_one",
        )
        if len(truth) != len(group):
            continue
        sigma = (
            group["p90"].to_numpy(dtype=float)
            - group["p10"].to_numpy(dtype=float)
        ) / (2.0 * _NORMAL_P90)
        sampler = McSampler(n_samp=mc_samples, rng_seed=seed + group_index)
        sampler.set_calibration(calibration)
        samples = sampler.sample_daily_high(
            group["base_temp_c"].to_numpy(dtype=float)
            + group["p50"].to_numpy(dtype=float),
            sigma,
            max_bias=0.0,
            hour_indices=[int(value) - 1 for value in group["fxx"]],
        )
        actual_future_high_f = float(
            truth["temperature_c"].astype(float).max() * 1.8 + 32.0
        )
        offsets[min(len(group), 24)].append(
            float(np.median(samples)) - actual_future_high_f
        )
    global_bias = float(calibration.max_bias_f)
    values = [0.0]
    counts = [0]
    for hours in range(1, 25):
        cell = offsets[hours]
        count = len(cell)
        raw = float(np.median(cell)) if cell else global_bias
        shrunk = (
            count * raw + float(prior_strength) * global_bias
        ) / (count + float(prior_strength))
        values.append(float(shrunk))
        counts.append(count)
    return PartialDayBiasCalibration(
        tuple(values),
        prior_strength=float(prior_strength),
        sample_counts=tuple(counts),
    )


@dataclass(frozen=True)
class DailyHighDistribution:
    daily_high_samples_f: np.ndarray
    coverage_type: str
    observed_high_f: float | None
    observed_hour_count: int
    observed_through: pd.Timestamp | None
    forecast_hour_count: int
    forecast_start_at: pd.Timestamp | None
    forecast_through: pd.Timestamp | None
    target_hour_count: int


def build_daily_high_distribution(
    *,
    init_time: object,
    target_date: str,
    station_timezone: str,
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    sampler: McSampler,
    bias: PartialDayBiasCalibration,
) -> DailyHighDistribution:
    """Combine observations visible by init with trusted future predictions."""
    init = _utc(init_time)
    expected = _target_hours(target_date, station_timezone)
    expected_set = set(expected)

    obs = observations.copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs["available_at"] = pd.to_datetime(obs["available_at"], utc=True)
    obs = obs[
        obs["valid_time"].isin(expected_set)
        & (obs["valid_time"] <= init)
        & (obs["available_at"] <= init)
    ].sort_values("valid_time")
    obs = obs.drop_duplicates("valid_time", keep="last")

    forecast = predictions.copy()
    forecast["valid_time"] = pd.to_datetime(
        forecast["valid_time"], utc=True
    )
    extrapolated = forecast.get(
        "is_extrapolated",
        pd.Series(False, index=forecast.index),
    ).fillna(False).astype(bool)
    forecast = forecast[
        forecast["valid_time"].isin(expected_set)
        & (forecast["valid_time"] > init)
        & ~extrapolated
    ].sort_values("valid_time")
    forecast = forecast.drop_duplicates("valid_time", keep="last")

    covered = set(obs["valid_time"]) | set(forecast["valid_time"])
    if covered != expected_set:
        missing = sorted(expected_set - covered)
        raise IncompleteDailyHighCoverage(
            "point-in-time daily high has uncovered local-day hours: "
            + ", ".join(stamp.isoformat() for stamp in missing)
        )

    observed_high_f: float | None = None
    if not obs.empty:
        observed_high_f = float(
            obs["temperature_c"].astype(float).max() * 1.8 + 32.0
        )

    if forecast.empty:
        if observed_high_f is None:
            raise IncompleteDailyHighCoverage(
                "daily high has neither observations nor forecasts"
            )
        samples = np.full(sampler.n_samp, observed_high_f, dtype=float)
    else:
        mu = (
            forecast["base_temp_c"].to_numpy(dtype=float)
            + forecast["p50"].to_numpy(dtype=float)
        )
        sigma = (
            forecast["p90"].to_numpy(dtype=float)
            - forecast["p10"].to_numpy(dtype=float)
        ) / (2.0 * _NORMAL_P90)
        samples = sampler.sample_daily_high(
            mu,
            sigma,
            max_bias=bias.for_forecast_hours(len(forecast)),
            hour_indices=[
                int(value) - 1 for value in forecast["fxx"].astype(int)
            ],
        )
        if observed_high_f is not None:
            samples = np.maximum(samples, observed_high_f)

    if obs.empty:
        coverage_type = "forecast_only"
    elif forecast.empty:
        coverage_type = "observed_only"
    else:
        coverage_type = "observed_plus_forecast"
    return DailyHighDistribution(
        daily_high_samples_f=np.asarray(samples, dtype=float),
        coverage_type=coverage_type,
        observed_high_f=observed_high_f,
        observed_hour_count=len(obs),
        observed_through=(
            pd.Timestamp(obs["valid_time"].max()) if not obs.empty else None
        ),
        forecast_hour_count=len(forecast),
        forecast_start_at=(
            pd.Timestamp(forecast["valid_time"].min())
            if not forecast.empty
            else None
        ),
        forecast_through=(
            pd.Timestamp(forecast["valid_time"].max())
            if not forecast.empty
            else None
        ),
        target_hour_count=len(expected),
    )


def _target_hours(target_date: str, timezone: str) -> pd.DatetimeIndex:
    zone = ZoneInfo(timezone)
    start = pd.Timestamp(target_date).tz_localize(zone)
    end = start + pd.DateOffset(days=1)
    return pd.date_range(
        start,
        end,
        freq="h",
        inclusive="right",
    ).tz_convert("UTC")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
