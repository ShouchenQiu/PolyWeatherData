from __future__ import annotations

import numpy as np
import pandas as pd

from training.backtest_forecast_generation import (
    build_forecast_cache,
    contiguous_oof_assignments,
    fit_oof_calibrations,
)
from model.daily_high import PartialDayBiasCalibration
from model.mc_calibration import McCalibration


def test_build_forecast_cache_uses_predictions_even_when_label_weight_is_zero() -> None:
    init = pd.Timestamp("2026-06-05T00:00:00Z")
    valid = pd.date_range(
        "2026-06-05T04:00:00Z",
        periods=24,
        freq="h",
    )
    hourly = pd.DataFrame(
        {
            "init_time": init,
            "valid_time": valid,
            "fxx": range(4, 28),
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
