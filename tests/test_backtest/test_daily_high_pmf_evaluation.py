from __future__ import annotations

import pandas as pd

from backtest.daily_high_pmf_evaluation import score_parallel_forecasts
from backtest.forecast_cache import ForecastCache
from training.daily_high_pmf import DailyHighPmfEvaluation


def _cache(snapshot_id: str, probability: float) -> ForecastCache:
    return ForecastCache(
        snapshots=pd.DataFrame(
            [
                {
                    "snapshot_id": snapshot_id,
                    "station_id": "KLGA",
                    "init_time": "2026-05-01T00:00:00Z",
                    "target_date": "2026-05-01",
                    "available_at": "2026-05-01T01:00:00Z",
                    "p10_f": 69.0,
                    "p50_f": 70.0,
                    "p90_f": 71.0,
                    "lineage_id": "lineage",
                    "coverage_type": "forecast_only",
                    "observed_high_f": float("nan"),
                    "observed_hour_count": 0,
                    "forecast_hour_count": 24,
                    "target_hour_count": 24,
                }
            ]
        ),
        probabilities=pd.DataFrame(
            [
                {"snapshot_id": snapshot_id, "condition_id": "low", "yes_probability": probability},
                {"snapshot_id": snapshot_id, "condition_id": "high", "yes_probability": 1 - probability},
            ]
        ),
        lineages=pd.DataFrame(
            [
                {
                    "lineage_id": "lineage",
                    "checkpoint_sha256": "a" * 64,
                    "calibration_sha256": "b" * 64,
                    "split_role": "model_test",
                    "mc_seed": 42,
                    "mc_samples": 100,
                }
            ]
        ),
    )


def test_score_parallel_forecasts_compares_same_snapshot_and_bucket_vector() -> None:
    candidate = _cache("candidate", 0.8)
    mc = _cache("mc", 0.4)
    evaluation = DailyHighPmfEvaluation(
        snapshots=pd.DataFrame(
            [{"snapshot_id": "candidate", "source_snapshot_id": "mc"}]
        ),
        temperature_probabilities=pd.DataFrame(),
    )
    markets = pd.DataFrame(
        [
            {"target_date": "2026-05-01", "condition_id": "low", "bucket_lo_f": float("-inf"), "bucket_hi_f": 70.0},
            {"target_date": "2026-05-01", "condition_id": "high", "bucket_lo_f": 70.0, "bucket_hi_f": float("inf")},
        ]
    )
    scores = score_parallel_forecasts(
        evaluation=evaluation,
        candidate_cache=candidate,
        mc_cache=mc,
        markets=markets,
        actuals=pd.DataFrame(
            [{"target_date": "2026-05-01", "actual_high_f": 69.0}]
        ),
    )

    assert len(scores) == 1
    assert scores.iloc[0]["candidate_rps"] < scores.iloc[0]["mc_rps"]
    assert scores.iloc[0]["candidate_true_bucket_probability"] == 0.8
    assert scores.iloc[0]["mc_true_bucket_probability"] == 0.4
