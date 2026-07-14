from __future__ import annotations

import pandas as pd
import pytest

from training.daily_high_attribution import (
    build_daily_high_layer_attribution,
    summarize_daily_high_layer_attribution,
)


def test_layer_attribution_separates_hourly_max_and_settlement_truth() -> None:
    predictions = pd.DataFrame(
        [
            {
                "init_time": "2026-04-01T00:00:00Z",
                "valid_time": "2026-04-01T01:00:00Z",
                "base_temp_c": 10.0,
                "target": 0.0,
                "p10": 0.0,
                "p50": 1.0,
                "p90": 2.0,
            },
            {
                "init_time": "2026-04-01T00:00:00Z",
                "valid_time": "2026-04-01T02:00:00Z",
                "base_temp_c": 10.0,
                "target": 2.0,
                "p10": 1.0,
                "p50": 3.0,
                "p90": 4.0,
            },
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "forecast",
                "init_time": "2026-04-01T00:00:00Z",
                "target_date": "2026-04-01",
                "coverage_type": "forecast_only",
                "observed_high_f": float("nan"),
                "observed_hour_count": 0,
                "forecast_hour_count": 2,
                "forecast_start_at": "2026-04-01T01:00:00Z",
                "forecast_through": "2026-04-01T02:00:00Z",
                "p10_f": 50.0,
                "p50_f": 55.0,
                "p90_f": 60.0,
                "target_hour_count": 2,
            },
            {
                "snapshot_id": "partial",
                "init_time": "2026-04-01T00:00:00Z",
                "target_date": "2026-04-01",
                "coverage_type": "observed_plus_forecast",
                "observed_high_f": 50.0,
                "observed_hour_count": 1,
                "forecast_hour_count": 1,
                "forecast_start_at": "2026-04-01T02:00:00Z",
                "forecast_through": "2026-04-01T02:00:00Z",
                "p10_f": 51.0,
                "p50_f": 55.0,
                "p90_f": 59.0,
                "target_hour_count": 2,
            },
        ]
    )
    actuals = pd.DataFrame(
        [{"target_date": "2026-04-01", "actual_high_f": 54.0}]
    )

    rows = build_daily_high_layer_attribution(
        predictions=predictions,
        snapshots=snapshots,
        actuals=actuals,
        station_timezone="UTC",
    ).set_index("snapshot_id")

    assert rows.loc["forecast", "future_iem_high_f"] == pytest.approx(53.6)
    assert rows.loc["forecast", "raw_tft_max_p50_f"] == pytest.approx(55.4)
    assert rows.loc["forecast", "hybrid_iem_high_f"] == pytest.approx(53.6)
    assert rows.loc["partial", "future_iem_high_f"] == pytest.approx(53.6)
    assert rows.loc["partial", "hybrid_iem_high_f"] == pytest.approx(53.6)
    assert rows.loc["partial", "weather_actual_high_f"] == pytest.approx(54.0)
    assert rows.loc["partial", "future_tft_max_bias_vs_iem_f"] == pytest.approx(1.8)
    assert rows.loc["partial", "hybrid_tft_max_bias_vs_iem_f"] == pytest.approx(1.8)
    assert rows.loc["partial", "hourly_p10_p90_coverage"] == 1.0


def test_layer_attribution_summary_keeps_coverage_types_separate() -> None:
    rows = pd.DataFrame(
        [
            {
                "target_date": "2026-04-01",
                "coverage_type": "forecast_only",
                "forecast_hour_count": 24,
                "mc_p50_mae_vs_weather_f": 2.0,
                "mc_p10_p90_coverage_vs_weather": 0.0,
            },
            {
                "target_date": "2026-04-01",
                "coverage_type": "observed_plus_forecast",
                "forecast_hour_count": 6,
                "mc_p50_mae_vs_weather_f": 1.0,
                "mc_p10_p90_coverage_vs_weather": 1.0,
            },
        ]
    )

    summary = summarize_daily_high_layer_attribution(rows)

    assert summary["overall"]["snapshot_count"] == 2
    assert summary["by_coverage"][0]["coverage_type"] == "forecast_only"
    assert summary["by_coverage"][1]["coverage_type"] == "observed_plus_forecast"
