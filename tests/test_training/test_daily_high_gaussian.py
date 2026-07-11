from __future__ import annotations

import numpy as np
import pandas as pd

from training.daily_high_gaussian import (
    FEATURE_COLUMNS,
    FutureMaxExamples,
    build_future_max_examples,
    run_expanding_gaussian_oof,
)


def test_future_max_examples_keep_future_observations_out_of_features() -> None:
    init = pd.Timestamp("2026-01-01T10:30:00Z")
    target_hours = pd.date_range("2026-01-01T01:00:00Z", periods=24, freq="h")
    predictions = pd.DataFrame(
        {
            "init_time": init,
            "valid_time": target_hours[10:],
            "fxx": range(1, 15),
            "p10": -1.0,
            "p50": 0.0,
            "p90": 1.0,
            "base_temp_c": 20.0,
            "is_extrapolated": False,
        }
    )
    temperatures = [20.0] * 10 + [22.0] * 13 + [25.0]
    observations = pd.DataFrame(
        {
            "valid_time": target_hours,
            "available_at": list(target_hours[:10]) + list(target_hours[10:] + pd.Timedelta(hours=2)),
            "temperature_c": temperatures,
        }
    )
    examples = build_future_max_examples(
        hourly_predictions=predictions,
        observations=observations,
        snapshots=pd.DataFrame(
            [{"init_time": init, "target_date": "2026-01-01"}]
        ),
        station_timezone="UTC",
    )

    row = examples.frame.iloc[0]
    assert row["observed_high_f"] == 68.0
    assert row["future_max_f"] == 77.0
    assert row["forecast_hour_count"] == 14
    assert row["tft_future_peak_p50_f"] == 68.0


def test_expanding_gaussian_oof_never_trains_on_validation_event_days() -> None:
    dates = pd.date_range("2026-04-01", periods=12, freq="D")
    frame = pd.DataFrame({
        "init_time": dates.tz_localize("UTC"),
        "target_date": dates.strftime("%Y-%m-%d"),
        "coverage_type": ["forecast_only"] * len(dates),
        "observed_high_f": [0.0] * len(dates),
        "observed_high_mask": [0.0] * len(dates),
        "observed_hour_count": [0] * len(dates),
        "forecast_hour_count": [24] * len(dates),
        "target_hour_count": [24] * len(dates),
        "future_max_f": np.linspace(60.0, 71.0, len(dates)),
    })
    for index, name in enumerate(FEATURE_COLUMNS):
        frame[name] = 65.0 if name == "tft_future_peak_p50_f" else float(index)

    evaluation = run_expanding_gaussian_oof(
        FutureMaxExamples(frame), tft_checkpoint_sha256="a" * 64,
        training_manifest_sha256="b" * 64, folds=4, epochs=2, seed=4,
    )

    assert len(evaluation.snapshots) == 9
    assert evaluation.snapshots["target_date"].min() > "2026-04-03"
    assert np.isfinite(evaluation.snapshots[["mean_f", "sigma_f", "future_nll"]]).all().all()
    assert (evaluation.snapshots["sigma_f"] > 0).all()
