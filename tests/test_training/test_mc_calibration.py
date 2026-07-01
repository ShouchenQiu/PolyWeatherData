from __future__ import annotations

import numpy as np
import pandas as pd

from training.mc_calibration import (
    cross_validate_mc_calibration,
    evaluate_mc_calibration_oof,
    fit_mc_calibration,
    local_calendar_day_groups,
)


def _prediction_frame(cycles: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    rows = []
    for cycle in range(cycles):
        init = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(
            hours=cycle
        )
        common = rng.normal(scale=1.2)
        for fxx in range(1, 49):
            error = common + rng.normal(scale=0.5)
            rows.append(
                {
                    "init_time": init,
                    "valid_time": init + pd.Timedelta(hours=fxx),
                    "fxx": fxx,
                    "target": error,
                    "p10": -1.2815515655,
                    "p50": 0.0,
                    "p90": 1.2815515655,
                    "base_temp_c": 15.0 + np.sin(fxx / 6),
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_local_calendar_days_exclude_partial_days_and_respect_timezone() -> None:
    frame = _prediction_frame(cycles=1)

    groups = local_calendar_day_groups(frame, "America/New_York")

    assert len(groups) == 1
    group = groups[0]
    local = pd.to_datetime(group["valid_time"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    assert local.dt.hour.tolist() == list(range(24))
    assert local.dt.date.nunique() == 1


def test_fit_mc_calibration_produces_dynamic_sigma_correlation(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")

    calibration = fit_mc_calibration(
        _prediction_frame(),
        checkpoint,
        station_timezone="UTC",
        mc_samples=300,
    )

    assert calibration.correlation.shape == (48, 48)
    assert np.linalg.eigvalsh(calibration.correlation).min() > 0
    assert calibration.sigma_scales.shape == (5,)
    assert (calibration.sigma_scales > 1.0).all()
    assert np.isfinite(calibration.max_bias_f)


def test_cross_validation_holds_out_each_cycle_once(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")

    evidence = cross_validate_mc_calibration(
        _prediction_frame(),
        checkpoint,
        station_timezone="UTC",
        folds=4,
        mc_samples=200,
    )

    assert evidence["folds"] == 4
    assert evidence["cycles"] == len(
        local_calendar_day_groups(_prediction_frame(), "UTC")
    )
    assert 0.0 <= evidence["p10_p90_coverage"] <= 1.0
    assert evidence["mean_interval_width_f"] > 0
    assert evidence["daily_high_mae_f"] >= 0


def test_oof_evidence_includes_one_row_per_complete_cycle_day(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")

    summary, predictions = evaluate_mc_calibration_oof(
        _prediction_frame(),
        checkpoint,
        station_timezone="UTC",
        folds=4,
        mc_samples=100,
    )

    assert len(predictions) == summary["cycles"] == len(
        local_calendar_day_groups(_prediction_frame(), "UTC")
    )
    assert not predictions.duplicated(["init_time", "target_date"]).any()
    assert {
        "actual_high_f",
        "hrrr_high_f",
        "tft_high_f",
        "mc_p10_f",
        "mc_p50_f",
        "mc_p90_f",
        "min_fxx",
        "max_fxx",
    } <= set(predictions)
