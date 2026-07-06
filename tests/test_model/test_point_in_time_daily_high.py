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
        "2026-01-01T01:00:00Z",
        "2026-01-01T11:00:00Z",
        freq="h",
    )
    observations = pd.DataFrame(
        {
            "valid_time": observed_times,
            "available_at": observed_times,
            "temperature_c": [20.0] * 10 + [30.0],
        }
    )
    forecast_times = pd.date_range(
        "2026-01-01T12:00:00Z",
        "2026-01-02T00:00:00Z",
        freq="h",
    )
    predictions = pd.DataFrame(
        {
            "valid_time": forecast_times,
            "fxx": range(1, 14),
            "p10": [9.0] * 13,
            "p50": [10.0] * 13,
            "p90": [11.0] * 13,
            "base_temp_c": [0.0] * 13,
            "is_extrapolated": [False] * 13,
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
    assert result.observed_hour_count == 11
    assert result.forecast_hour_count == 13
    assert result.target_hour_count == 24
    assert result.observed_high_f == 86.0
    assert np.all(result.daily_high_samples_f == 86.0)


def test_local_day_excludes_previous_2351_slot_and_includes_next_midnight() -> None:
    observations = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                [
                    "2026-05-28T04:00:00Z",
                    "2026-05-28T05:00:00Z",
                ]
            ),
            "available_at": pd.to_datetime(
                [
                    "2026-05-28T04:00:00Z",
                    "2026-05-28T05:00:00Z",
                ]
            ),
            "temperature_c": [25.0, 23.3333333333],
        }
    )
    forecast_times = pd.date_range(
        "2026-05-28T06:00:00Z",
        "2026-05-29T04:00:00Z",
        freq="h",
    )
    predictions = pd.DataFrame(
        {
            "valid_time": forecast_times,
            "fxx": range(1, len(forecast_times) + 1),
            "p10": [9.999] * len(forecast_times),
            "p50": [10.0] * len(forecast_times),
            "p90": [10.001] * len(forecast_times),
            "base_temp_c": [0.0] * len(forecast_times),
            "is_extrapolated": [False] * len(forecast_times),
        }
    )

    result = build_daily_high_distribution(
        init_time="2026-05-28T05:00:00Z",
        target_date="2026-05-28",
        station_timezone="America/New_York",
        predictions=predictions,
        observations=observations,
        sampler=McSampler(n_samp=100, rng_seed=42),
        bias=PartialDayBiasCalibration.zero(),
    )

    assert result.observed_hour_count == 1
    assert result.observed_high_f == pytest.approx(74.0)
    assert result.forecast_through == pd.Timestamp("2026-05-29T04:00:00Z")


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
        ("2026-03-08", "2026-03-08T06:00:00Z", 23),
        ("2026-11-01", "2026-11-01T05:00:00Z", 25),
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
