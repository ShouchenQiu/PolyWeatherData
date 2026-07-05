from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.daily_high import (
    IncompleteDailyHighCoverage,
    PartialDayBiasCalibration,
    build_daily_high_distribution,
    fit_partial_day_bias,
)
from model.mc import McSampler
from model.mc_calibration import McCalibration


def test_partial_day_combines_visible_observed_high_with_future_mc() -> None:
    init_time = pd.Timestamp("2026-01-01T11:00:00Z")
    observed_times = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-01T11:00:00Z",
        freq="h",
    )
    observations = pd.DataFrame(
        {
            "valid_time": observed_times,
            "available_at": observed_times,
            "temperature_c": [20.0] * 11 + [30.0],
        }
    )
    forecast_times = pd.date_range(
        "2026-01-01T12:00:00Z",
        "2026-01-01T23:00:00Z",
        freq="h",
    )
    predictions = pd.DataFrame(
        {
            "valid_time": forecast_times,
            "fxx": range(1, 13),
            "p10": [9.0] * 12,
            "p50": [10.0] * 12,
            "p90": [11.0] * 12,
            "base_temp_c": [0.0] * 12,
            "is_extrapolated": [False] * 12,
        }
    )

    result = build_daily_high_distribution(
        init_time=init_time,
        target_date="2026-01-01",
        station_timezone="UTC",
        predictions=predictions,
        observations=observations,
        sampler=McSampler(n_samp=100, rng_seed=42),
        bias=PartialDayBiasCalibration.zero(),
    )

    assert result.coverage_type == "observed_plus_forecast"
    assert result.observed_hour_count == 12
    assert result.forecast_hour_count == 12
    assert result.target_hour_count == 24
    assert result.observed_high_f == 86.0
    assert np.all(result.daily_high_samples_f == 86.0)


def test_partial_day_rejects_future_observation_instead_of_leaking() -> None:
    hours = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    observations = pd.DataFrame(
        {
            "valid_time": hours,
            "available_at": hours + pd.Timedelta(hours=1),
            "temperature_c": [20.0] * 24,
        }
    )

    with pytest.raises(
        IncompleteDailyHighCoverage,
        match="uncovered local-day hours",
    ):
        build_daily_high_distribution(
            init_time="2026-01-01T23:00:00Z",
            target_date="2026-01-01",
            station_timezone="UTC",
            predictions=pd.DataFrame(
                columns=[
                    "valid_time",
                    "fxx",
                    "p10",
                    "p50",
                    "p90",
                    "base_temp_c",
                    "is_extrapolated",
                ]
            ),
            observations=observations,
            sampler=McSampler(n_samp=10, rng_seed=42),
            bias=PartialDayBiasCalibration.zero(),
        )


def test_partial_bias_shrinks_remaining_hour_cell_to_full_day_prior() -> None:
    predictions = pd.DataFrame(
        {
            "init_time": pd.Timestamp("2026-01-01T21:00:00Z"),
            "valid_time": pd.date_range(
                "2026-01-01T22:00:00Z",
                periods=2,
                freq="h",
            ),
            "fxx": [1, 2],
            "p10": [11.999, 11.999],
            "p50": [12.0, 12.0],
            "p90": [12.001, 12.001],
            "base_temp_c": [0.0, 0.0],
            "is_extrapolated": [False, False],
        }
    )
    observations = pd.DataFrame(
        {
            "valid_time": predictions["valid_time"],
            "available_at": predictions["valid_time"],
            "temperature_c": [10.0, 10.0],
        }
    )
    calibration = McCalibration(
        correlation=np.eye(48),
        sigma_scales=np.ones(5),
        max_bias_f=1.0,
        model_sha256="a" * 64,
        fit_split="development",
        fit_start="2026-01-01",
        fit_end="2026-01-01",
        sample_count=1,
        station_timezone="UTC",
    )

    bias = fit_partial_day_bias(
        hourly_predictions=predictions,
        observations=observations,
        calibration=calibration,
        station_timezone="UTC",
        prior_strength=20.0,
        mc_samples=100,
        seed=42,
    )

    assert bias.sample_counts[2] == 1
    assert bias.for_forecast_hours(2) == pytest.approx(
        (3.6 + 20.0) / 21.0,
        abs=0.01,
    )


@pytest.mark.parametrize(
    ("target_date", "start_utc", "hours"),
    [
        ("2026-03-08", "2026-03-08T05:00:00Z", 23),
        ("2026-11-01", "2026-11-01T04:00:00Z", 25),
    ],
)
def test_forecast_only_uses_dst_local_day_hour_count(
    target_date: str,
    start_utc: str,
    hours: int,
) -> None:
    valid_times = pd.date_range(start_utc, periods=hours, freq="h")
    predictions = pd.DataFrame(
        {
            "valid_time": valid_times,
            "fxx": range(1, hours + 1),
            "p10": [9.0] * hours,
            "p50": [10.0] * hours,
            "p90": [11.0] * hours,
            "base_temp_c": [0.0] * hours,
            "is_extrapolated": [False] * hours,
        }
    )

    result = build_daily_high_distribution(
        init_time=pd.Timestamp(start_utc) - pd.Timedelta(hours=1),
        target_date=target_date,
        station_timezone="America/New_York",
        predictions=predictions,
        observations=pd.DataFrame(
            columns=["valid_time", "available_at", "temperature_c"]
        ),
        sampler=McSampler(n_samp=10, rng_seed=42),
        bias=PartialDayBiasCalibration.zero(),
    )

    assert result.target_hour_count == hours
    assert result.forecast_hour_count == hours
