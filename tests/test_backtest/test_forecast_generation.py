from __future__ import annotations

import numpy as np
import pandas as pd

from training.backtest_forecast_generation import (
    build_pmf_forecast_cache,
    build_forecast_cache,
    contiguous_oof_assignments,
    fit_oof_calibrations,
)
from training.daily_high_pmf import DailyHighPmfEvaluation
from model.daily_high import PartialDayBiasCalibration
from model.mc_calibration import McCalibration


def test_build_pmf_forecast_cache_keeps_parallel_distribution_lineage() -> None:
    evaluation = DailyHighPmfEvaluation(
        snapshots=pd.DataFrame(
            [
                {
                    "snapshot_id": "pmf-1",
                    "source_snapshot_id": "mc-1",
                    "init_time": "2026-05-01T00:00:00Z",
                    "target_date": "2026-05-01",
                    "p10_f": 69.0,
                    "p50_f": 70.0,
                    "p90_f": 71.0,
                    "coverage_type": "forecast_only",
                    "observed_high_f": 0.0,
                    "observed_hour_count": 0,
                    "forecast_hour_count": 24,
                    "target_hour_count": 24,
                }
            ]
        ),
        temperature_probabilities=pd.DataFrame(
            {
                "snapshot_id": ["pmf-1", "pmf-1"],
                "temperature_f": [69, 70],
                "pmf_probability": [0.25, 0.75],
            }
        ),
    )
    source = pd.DataFrame(
        [
            {
                "snapshot_id": "mc-1",
                "available_at": "2026-05-01T01:00:00Z",
                "observed_high_f": float("nan"),
                "observed_through": pd.NaT,
                "forecast_start_at": "2026-05-01T01:00:00Z",
                "forecast_through": "2026-05-02T00:00:00Z",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "target_date": "2026-05-01",
                "condition_id": "below-70",
                "bucket_lo_f": float("-inf"),
                "bucket_hi_f": 70.0,
            },
            {
                "target_date": "2026-05-01",
                "condition_id": "at-least-70",
                "bucket_lo_f": 70.0,
                "bucket_hi_f": float("inf"),
            },
        ]
    )

    cache = build_pmf_forecast_cache(
        evaluation=evaluation,
        source_snapshots=source,
        markets=markets,
        station_id="KLGA",
        split_role="model_test",
        tft_checkpoint_sha256="a" * 64,
        pmf_artifact_sha256="b" * 64,
        feature_schema_version=1,
    )

    assert cache.snapshots.iloc[0]["distribution_source"] == "daily_high_pmf_pilot"
    assert cache.lineages.iloc[0]["distribution_artifact_sha256"] == "b" * 64
    assert cache.probabilities.set_index("condition_id")["yes_probability"].to_dict() == {
        "at-least-70": 0.75,
        "below-70": 0.25,
    }


def test_build_pmf_forecast_cache_clips_only_float_sum_boundary_noise() -> None:
    evaluation = DailyHighPmfEvaluation(
        snapshots=pd.DataFrame(
            [{
                "snapshot_id": "pmf", "source_snapshot_id": "mc",
                "init_time": "2026-05-01T00:00:00Z", "target_date": "2026-05-01",
                "p10_f": 68.0, "p50_f": 69.0, "p90_f": 70.0,
                "coverage_type": "forecast_only", "observed_high_f": 0.0,
                "observed_hour_count": 0, "forecast_hour_count": 24,
                "target_hour_count": 24,
            }]
        ),
        temperature_probabilities=pd.DataFrame(
            {
                "snapshot_id": ["pmf"] * 3,
                "temperature_f": [68, 69, 70],
                "pmf_probability": [0.1, 0.2, 0.7000000000000002],
            }
        ),
    )
    cache = build_pmf_forecast_cache(
        evaluation=evaluation,
        source_snapshots=pd.DataFrame(
            [{
                "snapshot_id": "mc", "available_at": "2026-05-01T01:00:00Z",
                "observed_high_f": float("nan"), "observed_through": pd.NaT,
                "forecast_start_at": "2026-05-01T01:00:00Z",
                "forecast_through": "2026-05-02T00:00:00Z",
            }]
        ),
        markets=pd.DataFrame(
            [{
                "target_date": "2026-05-01", "condition_id": "all",
                "bucket_lo_f": float("-inf"), "bucket_hi_f": float("inf"),
            }]
        ),
        station_id="KLGA", split_role="model_test",
        tft_checkpoint_sha256="a" * 64, pmf_artifact_sha256="b" * 64,
        feature_schema_version=1,
    )

    assert cache.probabilities.iloc[0]["yes_probability"] == 1.0


def test_build_forecast_cache_uses_predictions_even_when_label_weight_is_zero() -> None:
    init = pd.Timestamp("2026-06-05T00:00:00Z")
    valid = pd.date_range(
        "2026-06-05T05:00:00Z",
        periods=24,
        freq="h",
    )
    hourly = pd.DataFrame(
        {
            "init_time": init,
            "valid_time": valid,
            "fxx": range(5, 29),
            "target": 0.0,
            "p10": -0.001,
            "p50": 0.0,
            "p90": 0.001,
            "base_temp_c": 22.0,
            "weight": 0.0,
        }
    )
    availability = pd.DataFrame(
        [
            {
                "init_time": init,
                "complete": True,
                "available_at": "2026-06-05T01:15:00Z",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "target_date": "2026-06-05",
                "event_end_at": "2026-06-06T12:00:00Z",
                "condition_id": "70-71",
                "bucket_lo_f": float("-inf"),
                "bucket_hi_f": 72.0,
            },
            {
                "target_date": "2026-06-05",
                "event_end_at": "2026-06-06T12:00:00Z",
                "condition_id": "at-least-72",
                "bucket_lo_f": 72.0,
                "bucket_hi_f": float("inf"),
            },
        ]
    )
    calibration = McCalibration(
        correlation=np.eye(48),
        sigma_scales=np.ones(5),
        max_bias_f=0.0,
        model_sha256="a" * 64,
        fit_split="val",
        fit_start="2026-04-01",
        fit_end="2026-04-30",
        sample_count=30,
        station_timezone="America/New_York",
    )
    lineages = pd.DataFrame(
        [
            {
                "lineage_id": "fixed",
                "checkpoint_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
                "split_role": "post_test_holdout",
                "mc_seed": 42,
                "mc_samples": 1000,
            }
        ]
    )
    assignments = pd.DataFrame(
        [{"init_time": init, "lineage_id": "fixed"}]
    )

    cache = build_forecast_cache(
        hourly_predictions=hourly,
        observations=pd.DataFrame(
            columns=["valid_time", "available_at", "temperature_c"]
        ),
        availability=availability,
        markets=markets,
        lineages=lineages,
        cycle_lineages=assignments,
        calibrations={"fixed": calibration},
        partial_biases={"fixed": PartialDayBiasCalibration.zero()},
        station_id="KLGA",
        station_timezone="America/New_York",
    )

    assert len(cache.snapshots) == 1
    assert cache.snapshots.iloc[0]["available_at"] == pd.Timestamp(
        "2026-06-05T01:15:00Z"
    )
    assert cache.snapshots.iloc[0]["target_date"] == "2026-06-05"
    assert cache.snapshots.iloc[0]["coverage_type"] == "forecast_only"
    assert cache.snapshots.iloc[0]["forecast_hour_count"] == 24
    assert set(cache.probabilities["condition_id"]) == {
        "70-71",
        "at-least-72",
    }
    assert cache.probabilities["yes_probability"].sum() == 1.0
    assert cache.probabilities.set_index("condition_id").loc[
        "at-least-72", "yes_probability"
    ] > 0.99


def test_build_forecast_cache_ignores_early_event_end_before_local_day_end() -> None:
    init = pd.Timestamp("2026-05-28T10:00:00Z")
    forecast_times = pd.date_range(
        "2026-05-28T11:00:00Z",
        "2026-05-29T00:00:00Z",
        freq="h",
    )
    hourly = pd.DataFrame(
        {
            "init_time": init,
            "valid_time": forecast_times,
            "fxx": range(1, len(forecast_times) + 1),
            "p10": -0.001,
            "p50": 0.0,
            "p90": 0.001,
            "base_temp_c": 22.0,
        }
    )
    observed_times = pd.date_range(
        "2026-05-28T01:00:00Z",
        init,
        freq="h",
    )
    observations = pd.DataFrame(
        {
            "valid_time": observed_times,
            "available_at": observed_times,
            "temperature_c": 22.0,
        }
    )
    availability = pd.DataFrame(
        [
            {
                "init_time": init,
                "complete": True,
                "available_at": "2026-05-28T13:00:00Z",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "target_date": "2026-05-28",
                "event_end_at": "2026-05-28T12:00:00Z",
                "condition_id": "all",
                "bucket_lo_f": float("-inf"),
                "bucket_hi_f": float("inf"),
            }
        ]
    )
    calibration = McCalibration(
        correlation=np.eye(48),
        sigma_scales=np.ones(5),
        max_bias_f=0.0,
        model_sha256="a" * 64,
        fit_split="val",
        fit_start="2026-04-01",
        fit_end="2026-04-30",
        sample_count=30,
        station_timezone="UTC",
    )
    lineages = pd.DataFrame(
        [
            {
                "lineage_id": "fixed",
                "checkpoint_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
                "split_role": "model_test",
                "mc_seed": 42,
                "mc_samples": 100,
            }
        ]
    )

    cache = build_forecast_cache(
        hourly_predictions=hourly,
        observations=observations,
        availability=availability,
        markets=markets,
        lineages=lineages,
        cycle_lineages=pd.DataFrame(
            [{"init_time": init, "lineage_id": "fixed"}]
        ),
        calibrations={"fixed": calibration},
        partial_biases={"fixed": PartialDayBiasCalibration.zero()},
        station_id="KLGA",
        station_timezone="UTC",
    )

    assert len(cache.snapshots) == 1
    assert cache.snapshots.iloc[0]["available_at"] == pd.Timestamp(
        "2026-05-28T13:00:00Z"
    )


def test_contiguous_oof_assignments_hold_out_every_cycle_once() -> None:
    init_times = pd.date_range("2026-04-01", periods=10, freq="h", tz="UTC")

    assignments = contiguous_oof_assignments(init_times, folds=3)

    assert assignments["init_time"].tolist() == list(init_times)
    assert assignments["lineage_id"].tolist() == [
        "oof-fold-0",
        "oof-fold-0",
        "oof-fold-0",
        "oof-fold-0",
        "oof-fold-1",
        "oof-fold-1",
        "oof-fold-1",
        "oof-fold-2",
        "oof-fold-2",
        "oof-fold-2",
    ]


def test_fit_oof_calibrations_never_fits_on_held_out_cycles(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    init_times = pd.date_range("2026-04-01", periods=6, freq="h", tz="UTC")
    hourly = pd.DataFrame({"init_time": init_times})
    fitted_on: list[set[pd.Timestamp]] = []

    def fake_fit(frame, checkpoint_path, *, station_timezone, mc_samples):
        fitted_on.append(set(pd.to_datetime(frame["init_time"], utc=True)))
        return McCalibration(
            correlation=np.eye(48),
            sigma_scales=np.ones(5),
            max_bias_f=0.0,
            model_sha256=McCalibration.sha256(checkpoint_path),
            fit_split="val",
            fit_start="2026-04-01",
            fit_end="2026-04-01",
            sample_count=len(frame),
            station_timezone=station_timezone,
        )

    bias_fitted_on: list[set[pd.Timestamp]] = []

    def fake_bias(*, hourly_predictions, **kwargs):
        bias_fitted_on.append(
            set(pd.to_datetime(hourly_predictions["init_time"], utc=True))
        )
        return PartialDayBiasCalibration.zero()

    lineages, assignments, calibrations, biases = fit_oof_calibrations(
        hourly,
        observations=pd.DataFrame(),
        checkpoint_path=checkpoint,
        station_timezone="America/New_York",
        folds=2,
        mc_samples=100,
        fit_calibration=fake_fit,
        fit_partial_bias=fake_bias,
    )

    for fold in range(2):
        lineage_id = f"oof-fold-{fold}"
        held_out = set(
            assignments.loc[
                assignments["lineage_id"] == lineage_id,
                "init_time",
            ]
        )
        assert held_out.isdisjoint(fitted_on[fold])
        assert held_out.isdisjoint(bias_fitted_on[fold])
    assert set(calibrations) == {"oof-fold-0", "oof-fold-1"}
    assert set(biases) == {"oof-fold-0", "oof-fold-1"}
    assert set(lineages["split_role"]) == {"oof_development"}
