from __future__ import annotations

import numpy as np
import pandas as pd

from training.mc_calibration import (
    cross_validate_mc_calibration,
    fit_mc_calibration,
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


def test_fit_mc_calibration_produces_dynamic_sigma_correlation(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")

    calibration = fit_mc_calibration(
        _prediction_frame(),
        checkpoint,
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
        folds=4,
        mc_samples=200,
    )

    assert evidence["folds"] == 4
    assert evidence["cycles"] == 80
    assert 0.0 <= evidence["p10_p90_coverage"] <= 1.0
    assert evidence["mean_interval_width_f"] > 0
    assert evidence["daily_high_mae_f"] >= 0
