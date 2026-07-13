from __future__ import annotations

import pandas as pd

from backtest.conditional_residual_evaluation import (
    actual_daily_highs,
    score_parallel_bucket_distributions,
    summarize_conditional_diagnostics,
)
from backtest.forecast_cache import ForecastCache


def _cache(probability: float, *, source: str) -> ForecastCache:
    snapshots = pd.DataFrame([{
        "snapshot_id": source, "station_id": "KLGA", "init_time": "2026-04-01T00:00:00Z",
        "target_date": "2026-04-01", "available_at": "2026-04-01T00:00:00Z",
        "p10_f": 50.0, "p50_f": 51.0, "p90_f": 52.0, "lineage_id": source,
        "path_ood": source == "candidate", "path_effective_sample_size": 40.0,
    }])
    probabilities = pd.DataFrame([{"snapshot_id": source, "condition_id": "bucket", "yes_probability": probability}])
    lineages = pd.DataFrame([{"lineage_id": source, "checkpoint_sha256": "a" * 64, "calibration_sha256": "b" * 64, "split_role": "oof_development", "mc_seed": 42, "mc_samples": 100}])
    return ForecastCache(snapshots, probabilities, lineages)


def test_scores_candidate_and_baseline_on_same_bucket_panel() -> None:
    scores = score_parallel_bucket_distributions(
        candidate=_cache(.8, source="candidate"), baseline=_cache(.2, source="baseline"),
        markets=pd.DataFrame([{"target_date": "2026-04-01", "condition_id": "bucket", "bucket_lo_f": 51.0, "bucket_hi_f": 52.0}]),
        actuals=pd.DataFrame([{"target_date": "2026-04-01", "actual_high_f": 51.2}]),
    )

    assert scores.loc[0, "truth_yes"] == 1.0
    assert scores.loc[0, "candidate_brier"] < scores.loc[0, "baseline_brier"]
    assert summarize_conditional_diagnostics(scores, bootstrap_samples=10)["event_days"] == 1


def test_actual_daily_highs_uses_station_local_calendar_day() -> None:
    highs = actual_daily_highs(
        pd.DataFrame({"valid_time": ["2026-04-02T02:00:00Z", "2026-04-02T03:00:00Z"], "temperature_c": [20.0, 25.0]}),
        station_timezone="America/New_York", start_target="2026-04-01", end_target="2026-04-01",
    )
    assert highs.loc[0, "actual_high_f"] == 77.0
