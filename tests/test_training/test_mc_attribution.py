from __future__ import annotations

import numpy as np

from model.daily_high import PartialDayBiasCalibration
from model.mc_calibration import McCalibration
from training.mc_attribution import (
    apply_attribution_variant,
    apply_partial_day_bias_variant,
    attribution_variant,
    historical_attribution_folds,
)


def _calibration() -> McCalibration:
    return McCalibration(
        correlation=np.full((48, 48), 0.2) + np.eye(48) * 0.8,
        sigma_scales=np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
        max_bias_f=0.7,
        model_sha256="a" * 64,
        fit_split="val",
        fit_start="2025-01-01",
        fit_end="2025-03-31",
        sample_count=100,
    )


def test_attribution_variants_change_only_the_selected_component() -> None:
    calibration = _calibration()

    no_sigma = apply_attribution_variant(
        calibration, attribution_variant("no_sigma_scales")
    )
    diagonal = apply_attribution_variant(
        calibration, attribution_variant("diagonal_dependence")
    )

    np.testing.assert_allclose(no_sigma.sigma_scales, 1.0)
    np.testing.assert_allclose(no_sigma.correlation, calibration.correlation)
    np.testing.assert_allclose(diagonal.correlation, np.eye(48))
    np.testing.assert_allclose(diagonal.sigma_scales, calibration.sigma_scales)


def test_no_partial_day_bias_is_exactly_zero_without_mutating_calibration() -> None:
    bias = PartialDayBiasCalibration((0.0,) + (0.5,) * 24)

    variant_bias = apply_partial_day_bias_variant(
        bias, attribution_variant("no_partial_day_bias")
    )

    assert variant_bias.bias_by_forecast_hours_f == (0.0,) * 25
    assert bias.for_forecast_hours(12) == 0.5


def test_historical_attribution_never_calibrates_on_its_evaluation_fold(tmp_path) -> None:
    for index in range(3):
        fold = tmp_path / f"fold-{index:02d}"
        fold.mkdir()
        (fold / "predictions.parquet").touch()

    folds = historical_attribution_folds(tmp_path)

    assert [(row.evaluation_fold_id, row.calibration_fold_ids) for row in folds] == [
        ("fold-01", ("fold-00",)),
        ("fold-02", ("fold-00", "fold-01")),
    ]
