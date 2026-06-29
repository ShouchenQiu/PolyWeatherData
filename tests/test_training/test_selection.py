from __future__ import annotations

import pandas as pd

from training.selection import compute_validation_evidence


def test_validation_evidence_reports_quantiles_strata_and_daily_high() -> None:
    frame = pd.DataFrame(
        {
            "init_time": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T06:00:00Z",
                    "2026-04-01T06:00:00Z",
                ]
            ),
            "valid_time": pd.to_datetime(
                [
                    "2026-04-01T01:00:00Z",
                    "2026-04-01T02:00:00Z",
                    "2026-04-01T07:00:00Z",
                    "2026-04-01T08:00:00Z",
                ]
            ),
            "fxx": [1, 2, 1, 2],
            "target": [0.0, 1.0, -1.0, 0.0],
            "p10": [-1.0, 0.0, -2.0, -1.0],
            "p50": [0.0, 1.0, -1.0, 0.0],
            "p90": [1.0, 2.0, 0.0, 1.0],
            "base_temp_c": [10.0, 12.0, 11.0, 13.0],
            "weight": [1.0, 1.0, 1.0, 0.0],
        }
    )

    evidence = compute_validation_evidence(frame, mc_samples=200)

    assert evidence["p50_mae"] == 0.0
    assert evidence["coverage_p10_p90"] == 1.0
    assert evidence["interval_width"] == 2.0
    assert set(evidence["by_fxx"]) == {"1", "2"}
    assert set(evidence["by_init_hour"]) == {"0", "6"}
    assert evidence["trusted_cells"] == 3
    assert evidence["daily_high"]["p50_mae_f"] == 0.0
    assert evidence["mc_calibration"]["sample_count"] == 200
    assert evidence["trading_backtest"]["status"] == "not_available"
