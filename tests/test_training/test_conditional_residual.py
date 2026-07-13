from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.daily_high import PartialDayBiasCalibration
from model.mc_calibration import McCalibration
from model.conditional_residual import ConditionalPathSampler, ConditionalResidualPathLibrary
from training.conditional_residual import (
    CONDITIONAL_STATE_FEATURES,
    build_complete_residual_path_library,
    build_conditional_daily_high_distribution,
    conditional_state,
    fit_conditional_partial_day_bias,
)


def _calibration() -> McCalibration:
    return McCalibration(
        correlation=np.eye(48), sigma_scales=np.ones(5), max_bias_f=0.0,
        model_sha256="a" * 64, fit_split="val", fit_start="2025-01-01",
        fit_end="2025-01-31", sample_count=10, station_timezone="UTC",
    )


def _cycle(*, extrapolated: bool = False) -> pd.DataFrame:
    init = pd.Timestamp("2025-01-01T00:00:00Z")
    return pd.DataFrame({
        "init_time": init,
        "valid_time": pd.date_range(init + pd.Timedelta(hours=1), periods=48, freq="h"),
        "fxx": range(1, 49), "p10": np.zeros(48), "p50": np.zeros(48),
        "p90": np.full(48, 2.0), "base_temp_c": np.arange(48, dtype=float),
        "target": np.ones(48), "weight": np.ones(48),
        "is_extrapolated": [extrapolated] * 48,
    })


def test_path_factory_uses_calibrated_standardized_residuals_and_rejects_tail() -> None:
    library = build_complete_residual_path_library(
        _cycle(), _calibration(), station_timezone="UTC", source_fold_id="fold-00"
    )

    assert len(library.paths) == 1
    assert tuple(library.feature_columns) == CONDITIONAL_STATE_FEATURES
    assert library.paths.loc[0, "residual_01"] == 1.2815515655446004

    with pytest.raises(ValueError, match="no complete trusted"):
        build_complete_residual_path_library(
            _cycle(extrapolated=True), _calibration(), station_timezone="UTC", source_fold_id="fold-00"
        )


def test_conditional_state_is_ex_ante_and_includes_observed_gap() -> None:
    state = conditional_state(
        _cycle().iloc[:24], target_date="2025-01-01", station_timezone="UTC",
        observed_hour_count=3, observed_high_f=60.0,
    )

    assert set(state) == set(CONDITIONAL_STATE_FEATURES)
    assert state["observed_hour_fraction"] == 3 / 27
    assert np.isfinite(list(state.values())).all()


def test_conditional_daily_high_preserves_visible_observed_high() -> None:
    library = build_complete_residual_path_library(
        _cycle(), _calibration(), station_timezone="UTC", source_fold_id="fold-00"
    )
    predictions = _cycle().iloc[11:24].copy()
    observations = pd.DataFrame({
        "valid_time": pd.date_range("2025-01-01T00:00:00Z", periods=12, freq="h"),
        "available_at": pd.date_range("2025-01-01T00:00:00Z", periods=12, freq="h"),
        "temperature_c": [30.0] * 12,
    })
    distribution = build_conditional_daily_high_distribution(
        init_time="2025-01-01T11:00:00Z", target_date="2025-01-01", station_timezone="UTC",
        predictions=predictions, observations=observations,
        sampler=ConditionalPathSampler(library, n_samp=20, rng_seed=42, neighbors=1, min_effective_samples=1),
        calibration=_calibration(), bias=PartialDayBiasCalibration.zero(),
    )

    assert distribution.coverage_type == "observed_plus_forecast"
    assert np.all(distribution.daily_high_samples_f >= 86.0)


def test_conditional_partial_bias_uses_the_path_library() -> None:
    cycle = _cycle()
    base_library = build_complete_residual_path_library(cycle, _calibration(), station_timezone="UTC", source_fold_id="fold-00")
    paths = pd.concat([base_library.paths.assign(path_id=lambda frame, index=index: frame["path_id"] + f"-{index}") for index in range(32)], ignore_index=True)
    library = ConditionalResidualPathLibrary.fit(paths, feature_columns=base_library.feature_columns)
    bias = fit_conditional_partial_day_bias(
        hourly_predictions=cycle,
        observations=pd.DataFrame({"valid_time": cycle["valid_time"], "temperature_c": cycle["base_temp_c"] + cycle["target"]}),
        calibration=_calibration(), library=library, station_timezone="UTC", mc_samples=20,
    )

    assert sum(bias.sample_counts) > 0
