from __future__ import annotations

import pandas as pd

from scripts.run_historical_gaussian_april_market_study import _paired_candidate_vs_mc


def test_paired_candidate_vs_mc_scores_the_same_market_vector() -> None:
    keys = {
        "event_id": ["event", "event"],
        "target_date": ["2026-04-01", "2026-04-01"],
        "timestamp": [pd.Timestamp("2026-04-01T12:00:00Z")] * 2,
        "condition_id": ["low", "high"],
    }
    candidate = pd.DataFrame({
        **keys,
        "bucket_order": [0, 1],
        "model_probability": [0.8, 0.2],
        "yes_outcome": [1.0, 0.0],
        "lead_band": ["12-24h", "12-24h"],
    })
    mc = candidate.assign(model_probability=[0.6, 0.4])

    paired = _paired_candidate_vs_mc(candidate, mc, bootstrap_samples=20)

    assert paired["sources"]["model_raw"]["log_score"] < paired["sources"][
        "synthetic_midpoint"
    ]["log_score"]
