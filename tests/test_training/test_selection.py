from __future__ import annotations

import pandas as pd

from training.selection import compute_validation_evidence


def test_validation_evidence_reports_quantiles_strata_and_daily_high() -> None:
    valid_time = pd.date_range(
        "2026-04-01T00:00:00Z",
        periods=24,
        freq="h",
    )
    frame = pd.DataFrame(
        {
            "init_time": [pd.Timestamp("2026-03-31T23:00:00Z")] * 24,
            "valid_time": valid_time,
            "fxx": range(1, 25),
            "target": [0.0] * 24,
            "p10": [-1.0] * 24,
            "p50": [0.0] * 24,
            "p90": [1.0] * 24,
            "base_temp_c": range(10, 34),
            "weight": [1.0] * 24,
        }
    )

    evidence = compute_validation_evidence(
        frame,
        station_timezone="UTC",
        mc_samples=200,
    )

    assert evidence["p50_mae"] == 0.0
    assert evidence["coverage_p10_p90"] == 1.0
    assert evidence["interval_width"] == 2.0
    assert set(evidence["by_fxx"]) == {str(value) for value in range(1, 25)}
    assert set(evidence["by_init_hour"]) == {"23"}
    assert evidence["trusted_cells"] == 24
    assert evidence["daily_high"]["p50_mae_f"] == 0.0
    assert evidence["mc_calibration"]["sample_count"] == 200
    assert evidence["trading_backtest"]["status"] == "not_available"
