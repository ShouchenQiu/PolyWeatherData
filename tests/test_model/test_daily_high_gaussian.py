from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.daily_high_gaussian import GaussianFutureMaxModel, TruncatedNormal, compose_daily_high
from training.daily_high_gaussian import fit_gaussian_future_max_model


def test_composed_daily_high_collapses_future_mass_below_observed_high() -> None:
    future = TruncatedNormal(mean_f=70.0, sigma_f=2.0)
    daily = compose_daily_high(future, observed_high_f=72.0)

    assert daily.probability_for_bucket(float("-inf"), 72.0) == 0.0
    assert daily.probability_for_bucket(72.0, 73.0) > 0.5
    assert daily.probability_for_bucket(73.0, float("inf")) > 0.0
    assert daily.p50_f == 72.0
    samples = daily.samples(seed=3, count=500)
    assert np.all(samples >= 72.0)
    assert np.any(samples == 72.0)


def test_truncated_normal_bucket_probabilities_obey_half_up_temperature_bins() -> None:
    distribution = compose_daily_high(TruncatedNormal(mean_f=70.0, sigma_f=0.1))

    assert distribution.probability_for_bucket(70.0, 71.0) > 0.999
    assert distribution.p10_f == 70.0
    assert distribution.p50_f == 70.0
    assert distribution.p90_f == 70.0
    with pytest.raises(ValueError, match="bucket lower"):
        distribution.probability_for_bucket(71.0, 71.0)


def test_gaussian_future_max_artifact_round_trip(tmp_path) -> None:
    features = ("tft_future_peak_p50_f", "forecast_hour_fraction")
    frame = pd.DataFrame(
        {
            "tft_future_peak_p50_f": [65.0, 67.0, 70.0, 72.0],
            "forecast_hour_fraction": [1.0, 0.75, 0.5, 0.25],
            "future_max_f": [66.0, 69.0, 71.0, 74.0],
        }
    )
    model = fit_gaussian_future_max_model(
        frame,
        feature_columns=features,
        tft_checkpoint_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
        epochs=3,
        seed=9,
    )
    before = model.predict_future(frame.iloc[1])
    model.save(tmp_path / "model")
    loaded = GaussianFutureMaxModel.load(
        tmp_path / "model", expected_tft_checkpoint_sha256="a" * 64
    )
    after = loaded.predict_future(frame.iloc[1])

    assert before.mean_f == pytest.approx(after.mean_f)
    assert before.sigma_f == pytest.approx(after.sigma_f)
    with pytest.raises(ValueError, match="TFT checkpoint hash mismatch"):
        GaussianFutureMaxModel.load(
            tmp_path / "model", expected_tft_checkpoint_sha256="c" * 64
        )
