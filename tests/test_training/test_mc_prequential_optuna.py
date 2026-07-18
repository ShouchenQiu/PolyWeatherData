from __future__ import annotations

import numpy as np
import pytest
import pandas as pd
import argparse
import json

import optuna
import scripts.run_prequential_mc_optuna as prequential_script

from model.daily_high import PartialDayBiasCalibration
from model.mc_calibration import McCalibration
from training.mc_prequential_optuna import (
    McTuneComponents,
    apply_mc_tune_parameters,
    enqueue_identity_mc_trial,
    event_day_rps,
    fixed_integer_rps,
    identity_mc_tune_parameters,
    mc_tune_component_subsets,
    shapley_rps_attribution,
)
from scripts.run_prequential_mc_optuna import (
    _planned_folds,
    selection_frame_for_mode,
    _write_or_validate_contract,
    filter_evaluation_prediction_window,
)


def _calibration() -> McCalibration:
    return McCalibration(
        correlation=np.full((48, 48), 0.2) + np.eye(48) * 0.8,
        sigma_scales=np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        max_bias_f=0.5,
        model_sha256="a" * 64,
        fit_split="oof",
        fit_start="2025-01-01",
        fit_end="2025-03-31",
        sample_count=100,
    )


def test_tuned_components_change_only_their_declared_mc_contract() -> None:
    calibration = _calibration()
    bias = PartialDayBiasCalibration((0.0,) + (0.4,) * 24)

    tuned_calibration, tuned_bias = apply_mc_tune_parameters(
        calibration,
        bias,
        components=McTuneComponents(sigma=True, correlation=False, partial_bias=False),
        parameters={"sigma_multiplier_0": 1.2, "sigma_multiplier_1": 0.9,
                    "sigma_multiplier_2": 1.1, "sigma_multiplier_3": 1.0,
                    "sigma_multiplier_4": 0.8},
    )

    np.testing.assert_allclose(
        tuned_calibration.sigma_scales,
        calibration.sigma_scales * np.array([1.2, 0.9, 1.1, 1.0, 0.8]),
    )
    np.testing.assert_allclose(tuned_calibration.correlation, calibration.correlation)
    assert tuned_bias == bias


def test_every_parameter_subset_has_an_exact_identity_candidate() -> None:
    calibration = _calibration()
    bias = PartialDayBiasCalibration((0.0,) + (0.4,) * 24)

    for components in mc_tune_component_subsets():
        tuned_calibration, tuned_bias = apply_mc_tune_parameters(
            calibration,
            bias,
            components=components,
            parameters=identity_mc_tune_parameters(components),
        )

        np.testing.assert_allclose(tuned_calibration.sigma_scales, calibration.sigma_scales)
        np.testing.assert_allclose(tuned_calibration.correlation, calibration.correlation)
        assert tuned_bias == bias


def test_tuner_executes_the_exact_identity_candidate_before_random_trials() -> None:
    components = McTuneComponents(sigma=True, correlation=True, partial_bias=True)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))

    enqueue_identity_mc_trial(study, components)
    study.optimize(lambda trial: sum(suggested for suggested in (
        trial.suggest_float(name, 0.0, 2.0)
        for name in identity_mc_tune_parameters(components)
    )), n_trials=1)

    assert study.best_params == identity_mc_tune_parameters(components)


def test_tuned_correlation_is_positive_definite_and_unit_diagonal() -> None:
    calibration, bias = _calibration(), PartialDayBiasCalibration.zero()

    tuned, _ = apply_mc_tune_parameters(
        calibration,
        bias,
        components=McTuneComponents(sigma=False, correlation=True, partial_bias=False),
        parameters={"correlation_ar1_weight": 0.7, "correlation_ar1_rho": 0.8},
    )

    assert np.allclose(np.diag(tuned.correlation), 1.0)
    assert np.linalg.eigvalsh(tuned.correlation).min() > 0
    assert not np.allclose(tuned.correlation, calibration.correlation)


def test_partial_bias_tuning_preserves_the_zero_hour_contract() -> None:
    calibration, bias = _calibration(), PartialDayBiasCalibration((0.0,) + (0.4,) * 24)

    _, tuned = apply_mc_tune_parameters(
        calibration,
        bias,
        components=McTuneComponents(sigma=False, correlation=False, partial_bias=True),
        parameters={"partial_bias_gain": 1.5, "partial_bias_offset_f": -0.2},
    )

    assert tuned.for_forecast_hours(0) == 0.0
    assert tuned.for_forecast_hours(12) == pytest.approx(0.4)


def test_shapley_rps_attribution_recovers_an_additive_game() -> None:
    scores = {
        "none": 0.100,
        "sigma": 0.090,
        "correlation": 0.080,
        "partial_bias": 0.095,
        "sigma+correlation": 0.070,
        "sigma+partial_bias": 0.085,
        "correlation+partial_bias": 0.075,
        "sigma+correlation+partial_bias": 0.065,
    }

    values = shapley_rps_attribution(scores)

    assert values == pytest.approx(
        {"sigma": 0.010, "correlation": 0.020, "partial_bias": 0.005}
    )


def test_factorial_subsets_are_complete_and_event_days_are_equal_weighted() -> None:
    assert [item.subset_id for item in mc_tune_component_subsets()] == [
        "none",
        "sigma",
        "correlation",
        "partial_bias",
        "sigma+correlation",
        "sigma+partial_bias",
        "correlation+partial_bias",
        "sigma+correlation+partial_bias",
    ]

    rows = __import__("pandas").DataFrame(
        {
            "target_date": ["2026-04-01", "2026-04-01", "2026-04-02"],
            "rps": [0.0, 0.2, 0.4],
        }
    )

    assert event_day_rps(rows) == pytest.approx(0.25)


def test_fixed_integer_rps_uses_one_shared_support_for_all_candidates() -> None:
    concentrated = fixed_integer_rps(
        np.array([70, 70, 71, 71]), actual_integer_f=70,
    )
    wide = fixed_integer_rps(
        np.array([-100, 70, 160, 160]), actual_integer_f=70,
    )

    assert concentrated < wide
    with pytest.raises(ValueError, match="outside fixed support"):
        fixed_integer_rps(np.array([70]), actual_integer_f=161)


def test_evaluation_window_drops_crossfit_horizon_tail_outside_declared_dates() -> None:
    frame = pd.DataFrame(
        {
            "valid_time": pd.to_datetime([
                "2026-04-01T04:00:00Z",  # Right-closed target date: March 31.
                "2026-04-01T05:00:00Z",  # Right-closed target date: April 1.
            ]),
            "value": [1, 2],
        }
    )

    filtered = filter_evaluation_prediction_window(
        frame, predict=("2026-04-01", "2026-04-30"),
        station_timezone="America/New_York",
    )

    assert filtered["value"].tolist() == [2]


def test_requested_probe_folds_remain_a_valid_subset_of_the_fixed_plan(tmp_path) -> None:
    for index in range(6):
        fold = tmp_path / f"fold-{index:02d}"
        fold.mkdir()
        (fold / "predictions.parquet").touch()

    selected = _planned_folds(
        tmp_path,
        requested_fold_ids=["fold-04", "fold-02"],
    )

    assert [fold.fold_id for fold in selected] == ["fold-02", "fold-04"]
    with pytest.raises(ValueError, match="unknown requested fold ids"):
        _planned_folds(tmp_path, requested_fold_ids=["fold-99"])


def test_oracle_selection_uses_only_the_declared_evaluation_frame() -> None:
    tuning = pd.DataFrame({"kind": ["tune"]})
    evaluation = pd.DataFrame({"kind": ["evaluation"]})

    assert selection_frame_for_mode("frozen_tune", tuning, evaluation) is tuning
    assert selection_frame_for_mode("oracle_eval", tuning, evaluation) is evaluation
    with pytest.raises(ValueError, match="selection mode"):
        selection_frame_for_mode("unknown", tuning, evaluation)


def test_resume_rejects_a_pre_window_filter_contract(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_contract.json").write_text(
        json.dumps({"schema_version": 2}), encoding="ascii"
    )
    args = argparse.Namespace(
        output=output,
        station="KLGA",
        timezone="America/New_York",
        trials=40,
        calibration_mc_samples=2_000,
        trial_mc_samples=256,
        evaluation_mc_samples=4_000,
        fold_ids=None,
        selection_mode="frozen_tune",
        crossfit_root=tmp_path / "crossfit",
        tune_predictions_root=tmp_path / "tune",
    )
    monkeypatch.setattr(prequential_script, "_source_hashes", lambda *_: {})

    with pytest.raises(ValueError, match="resume output contract does not match"):
        _write_or_validate_contract(args, ("fold-00",))
