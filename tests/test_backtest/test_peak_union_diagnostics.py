from __future__ import annotations

import pandas as pd
import pytest

from backtest.peak_union_diagnostics import (
    build_contiguous_union_panel,
    build_peak_snapshot_diagnostics,
    score_union_panel,
)


def test_contiguous_union_panel_sums_probabilities_and_excludes_full_event() -> None:
    panel = pd.DataFrame(
        {
            "event_id": ["event-1"] * 3,
            "target_date": ["2026-04-01"] * 3,
            "timestamp": pd.to_datetime(["2026-04-01T12:00:00Z"] * 3),
            "bucket_order": [0, 1, 2],
            "model_probability": [0.2, 0.5, 0.3],
            "midpoint_probability": [0.3, 0.4, 0.3],
            "yes_outcome": [0.0, 1.0, 0.0],
            "lead_band": ["12-24h"] * 3,
        }
    )

    unions = build_contiguous_union_panel(panel)

    assert len(unions) == 5  # 3 singleton unions + two width-two unions.
    middle = unions.query("union_start == 1 and union_end == 1").iloc[0]
    assert middle["model_probability"] == pytest.approx(0.5)
    assert middle["midpoint_probability"] == pytest.approx(0.4)
    assert middle["yes_outcome"] == 1.0
    assert not ((unions["union_start"] == 0) & (unions["union_end"] == 2)).any()


def test_peak_diagnostics_excludes_future_observations_and_uses_target_day_slots() -> None:
    snapshots = pd.DataFrame(
        {
            "snapshot_id": ["s1"],
            "init_time": pd.to_datetime(["2026-04-01T12:00:00Z"]),
            "available_at": pd.to_datetime(["2026-04-01T12:00:00Z"]),
            "target_date": ["2026-04-01"],
            "coverage_type": ["observed_plus_forecast"],
            "observed_hour_count": [1],
        }
    )
    predictions = pd.DataFrame(
        {
            "init_time": pd.to_datetime(["2026-04-01T12:00:00Z"] * 2),
            "valid_time": pd.to_datetime(["2026-04-01T13:00:00Z", "2026-04-01T14:00:00Z"]),
            "p10": [-1.0, -1.0], "p50": [1.0, 3.0], "p90": [3.0, 7.0],
            "base_temp_c": [10.0, 10.0],
        }
    )
    observations = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(["2026-04-01T11:00:00Z", "2026-04-01T12:00:00Z"]),
            "available_at": pd.to_datetime(["2026-04-01T11:30:00Z", "2026-04-01T13:00:00Z"]),
            "temperature_c": [12.0, 30.0],
        }
    )

    result = build_peak_snapshot_diagnostics(
        snapshots=snapshots, predictions=predictions, observations=observations,
        station_timezone="America/New_York",
    )

    row = result.iloc[0]
    assert row["observed_high_f"] == pytest.approx(53.6)
    assert row["tft_peak_f"] == pytest.approx(55.4)
    assert row["tft_peak_margin_f"] == pytest.approx(3.6)
    assert row["tft_peak_interval_width_f"] == pytest.approx(14.4)
    assert row["observed_future_peak_gap_f"] == pytest.approx(-1.8)


def test_union_scoring_tolerates_machine_precision_probability_roundoff() -> None:
    unions = pd.DataFrame(
        {
            "event_id": ["event-1"], "target_date": ["2026-04-01"],
            "timestamp": pd.to_datetime(["2026-04-01T12:00:00Z"]),
            "union_width": [1], "union_position": ["middle"], "lead_band": ["12-24h"],
            "model_probability": [1.0000000000000002], "midpoint_probability": [0.5],
            "yes_outcome": [1.0],
        }
    )

    scores, _, _ = score_union_panel(unions, bootstrap_samples=10)

    assert scores.loc[0, "model_probability"] == 1.0
