from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.daily_high_pmf import DailyHighPmfModel, TemperaturePmf
from training.daily_high_pmf import fit_daily_high_pmf_model


def test_temperature_pmf_masks_values_below_observed_high_and_maps_buckets() -> None:
    pmf = TemperaturePmf.from_logits(
        np.array([0.0, 0.0, 0.0, 0.0]),
        support_min_f=68,
        observed_high_f=69.2,
    )

    np.testing.assert_allclose(pmf.probabilities, [0.0, 1 / 3, 1 / 3, 1 / 3])
    assert pmf.probability_for_bucket(float("-inf"), 71.0) == pytest.approx(2 / 3)
    assert pmf.probability_for_bucket(71.0, float("inf")) == pytest.approx(1 / 3)
    assert pmf.quantile(0.1) == 69.0
    assert pmf.quantile(0.9) == 71.0
    assert pmf.p10_f == 69.0
    assert pmf.p50_f == 70.0
    assert pmf.p90_f == 71.0
    assert set(pmf.samples(seed=7, count=100)) <= {69.0, 70.0, 71.0}


def test_daily_high_pmf_artifact_round_trip_and_rejects_tft_mismatch(
    tmp_path,
) -> None:
    features = ("observed_high_f", "forecast_hour_fraction")
    frame = pd.DataFrame(
        {
            "observed_high_f": [0.0, 65.0, 70.0, 72.0],
            "forecast_hour_fraction": [1.0, 0.75, 0.5, 0.25],
            "observed_high_mask": [0.0, 1.0, 1.0, 1.0],
            "label_temperature_f": [68, 69, 72, 74],
        }
    )
    model = fit_daily_high_pmf_model(
        frame,
        feature_columns=features,
        tft_checkpoint_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
        epochs=2,
        seed=7,
    )
    before = model.predict(frame.iloc[1])
    model.save(tmp_path / "candidate")
    loaded = DailyHighPmfModel.load(
        tmp_path / "candidate", expected_tft_checkpoint_sha256="a" * 64
    )
    after = loaded.predict(frame.iloc[1])

    np.testing.assert_allclose(before.probabilities, after.probabilities)
    assert after.probabilities.sum() == pytest.approx(1.0)
    assert after.probability_for_bucket(69.0, 70.0) >= 0.0
    with pytest.raises(ValueError, match="TFT checkpoint hash mismatch"):
        DailyHighPmfModel.load(
            tmp_path / "candidate",
            expected_tft_checkpoint_sha256="c" * 64,
        )
