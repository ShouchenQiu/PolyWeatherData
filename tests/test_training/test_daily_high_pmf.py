from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.daily_high_pmf import (
    DailyHighPmfExamples,
    build_point_in_time_examples,
    expanding_target_day_folds,
    run_expanding_pmf_oof,
)


def _forecast(init: str, valid_times: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "init_time": pd.Timestamp(init),
            "valid_time": valid_times,
            "fxx": range(1, len(valid_times) + 1),
            "p10": -1.0,
            "p50": 0.0,
            "p90": 1.0,
            "base_temp_c": 20.0,
            "is_extrapolated": False,
        }
    )


def test_point_in_time_examples_use_only_observations_visible_at_init() -> None:
    init = "2026-01-01T10:30:00Z"
    target_hours = pd.date_range(
        "2026-01-01T01:00:00Z", periods=24, freq="h"
    )
    observations = pd.DataFrame(
        {
            "valid_time": target_hours[:11],
            "available_at": list(target_hours[:10]) + [target_hours[11]],
            "temperature_c": [10.0] * 10 + [40.0],
        }
    )
    snapshots = pd.DataFrame(
        [{"init_time": init, "target_date": "2026-01-01"}]
    )
    actuals = pd.DataFrame(
        [{"target_date": "2026-01-01", "actual_high_f": 77.0}]
    )

    examples = build_point_in_time_examples(
        hourly_predictions=_forecast(init, target_hours[10:]),
        observations=observations,
        snapshots=snapshots,
        actuals=actuals,
        station_timezone="UTC",
    )

    assert isinstance(examples, DailyHighPmfExamples)
    assert len(examples.frame) == 1
    row = examples.frame.iloc[0]
    assert row["observed_hour_count"] == 10
    assert row["forecast_hour_count"] == 14
    assert row["observed_high_f"] == pytest.approx(50.0)
    assert row["slot_10_observation_mask"] == 0.0
    assert row["slot_10_forecast_mask"] == 1.0
    assert row["label_temperature_f"] == 77
    assert len(examples.feature_columns) == len(set(examples.feature_columns))
    assert np.isfinite(
        examples.frame[list(examples.feature_columns)].to_numpy(dtype=float)
    ).all()


def test_examples_artifact_round_trip_preserves_feature_contract(tmp_path) -> None:
    init = "2026-01-01T00:00:00Z"
    target_hours = pd.date_range("2026-01-01T01:00:00Z", periods=24, freq="h")
    examples = build_point_in_time_examples(
        hourly_predictions=_forecast(init, target_hours),
        observations=pd.DataFrame(
            columns=["valid_time", "available_at", "temperature_c"]
        ),
        snapshots=pd.DataFrame(
            [{"init_time": init, "target_date": "2026-01-01"}]
        ),
        actuals=pd.DataFrame(
            [{"target_date": "2026-01-01", "actual_high_f": 71.5}]
        ),
        station_timezone="UTC",
    )
    manifest = examples.save(
        tmp_path / "examples", source_sha256={"predictions": "a" * 64}
    )
    loaded = DailyHighPmfExamples.load(tmp_path / "examples")

    assert manifest["row_count"] == 1
    assert manifest["target_date_range"] == ["2026-01-01", "2026-01-01"]
    assert loaded.feature_columns == examples.feature_columns
    pd.testing.assert_frame_equal(loaded.frame, examples.frame)


def test_expanding_folds_train_only_on_earlier_event_days() -> None:
    frame = pd.DataFrame(
        {
            "target_date": np.repeat(
                pd.date_range("2026-04-01", periods=10).date.astype(str), 2
            ),
            "value": range(20),
        }
    )

    folds = expanding_target_day_folds(frame, folds=4)

    assert len(folds) == 4
    validation_days: list[str] = []
    for train, validation in folds:
        assert set(train.index).isdisjoint(validation.index)
        assert train["target_date"].max() < validation["target_date"].min()
        assert set(train["target_date"]).isdisjoint(validation["target_date"])
        validation_days.extend(validation["target_date"].unique())
    assert validation_days == sorted(validation_days)


def test_expanding_oof_emits_normalized_candidate_and_climatology_pmfs() -> None:
    rows = []
    for day_index, target_date in enumerate(
        pd.date_range("2026-04-01", periods=10).date.astype(str)
    ):
        target_date = str(target_date)
        rows.append(
            {
                "init_time": pd.Timestamp(target_date, tz="UTC"),
                "target_date": target_date,
                "coverage_type": "forecast_only",
                "observed_hour_count": 0,
                "forecast_hour_count": 24,
                "observed_high_f": 0.0,
                "observed_high_mask": 0.0,
                "feature": float(day_index),
                "label_temperature_f": 60 + day_index,
            }
        )
    examples = DailyHighPmfExamples(
        pd.DataFrame(rows), feature_columns=("feature",)
    )

    evidence = run_expanding_pmf_oof(
        examples,
        tft_checkpoint_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
        folds=4,
        epochs=1,
        seed=3,
    )

    assert evidence.snapshots["target_date"].nunique() == 8
    assert set(evidence.snapshots["fold"]) == {0, 1, 2, 3}
    grouped = evidence.temperature_probabilities.groupby("snapshot_id")
    assert np.allclose(grouped["pmf_probability"].sum(), 1.0)
    assert np.allclose(grouped["climatology_probability"].sum(), 1.0)
    assert {
        "pmf_log_loss",
        "climatology_log_loss",
        "pmf_rps",
        "climatology_rps",
    } <= set(evidence.snapshots)
