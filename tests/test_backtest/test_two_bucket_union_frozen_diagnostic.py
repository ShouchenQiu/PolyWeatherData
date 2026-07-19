from __future__ import annotations

import pandas as pd
import pytest

from backtest.two_bucket_union_frozen_diagnostic import (
    add_exact_horizon_band,
    add_fixed_gap_band,
    build_frozen_union_diagnostic_report,
    write_frozen_union_diagnostic_artifact,
)


def test_exact_horizon_bands_separate_the_far_forecast_window() -> None:
    frame = pd.DataFrame(
        {"lead_hours": [5.9, 6.0, 12.0, 24.0, 30.0, 36.0, 42.0, 48.0]}
    )

    result = add_exact_horizon_band(frame)

    assert result["horizon_band"].tolist() == [
        "<6h", "6-12h", "12-24h", "24-30h",
        "30-36h", "36-42h", "42-48h", "out_of_scope",
    ]


def test_gap_bands_are_fixed_before_diagnostic_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "model_union_probability": [0.60, 0.70, 0.80, 0.90],
            "midpoint_union_probability": [0.50, 0.50, 0.50, 0.40],
            "union_outcome": [1.0, 0.0, 1.0, 0.0],
        }
    )

    result = add_fixed_gap_band(frame)

    assert result["model_market_gap_band"].tolist() == [
        "<0.15", "0.15-0.30", "0.30-0.45", ">=0.45",
    ]


def test_frozen_report_closes_month_horizon_and_lcb_questions() -> None:
    settlements = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "target_date": ["2026-04-01", "2026-04-02", "2026-05-01", "2026-05-02"],
            "evaluation_id": ["april_2026", "april_2026", "may_2026", "may_2026"],
            "horizon_band": ["30-36h", "42-48h", "36-42h", "42-48h"],
            "coverage_type": ["forecast_only"] * 4,
            "stage": ["frozen"] * 4,
            "execution_mode": ["taker"] * 4,
            "edge_threshold": [0.0] * 4,
            "union_outcome": [1.0, 0.0, 1.0, 0.0],
            "union_probability_lower_90": [0.6, 0.7, 0.6, 0.7],
            "model_union_probability": [0.7, 0.8, 0.7, 0.8],
            "midpoint_union_probability": [0.5, 0.5, 0.5, 0.5],
            "shares": [10.0] * 4,
            "cost_basis_usd": [5.0, 5.0, 5.0, 5.0],
            "payout_usd": [10.0, 0.0, 10.0, 0.0],
            "realized_pnl_usd": [5.0, -5.0, 5.0, -5.0],
            "roi": [1.0, -1.0, 1.0, -1.0],
            "basket_fill_status": ["full"] * 4,
        }
    )
    candidates = settlements.loc[:, [
        "event_id", "target_date", "evaluation_id", "horizon_band",
        "coverage_type", "union_outcome", "model_union_probability",
        "midpoint_union_probability", "union_probability_lower_90",
    ]].assign(
        timestamp=pd.to_datetime(
            ["2026-03-31T12:00:00Z", "2026-04-01T12:00:00Z",
             "2026-04-30T12:00:00Z", "2026-05-01T12:00:00Z"]
        ),
        modal_neighbor_direction=["lower", "upper", "lower", "upper"],
    )
    candidates = pd.concat(
        [
            candidates,
            candidates.iloc[[0]].assign(
                event_id="no-trade",
                target_date="2026-04-03",
                timestamp=pd.to_datetime(["2026-04-02T12:00:00Z"]),
            ),
        ],
        ignore_index=True,
    )

    report = build_frozen_union_diagnostic_report(
        candidates=candidates,
        decisions=pd.DataFrame(),
        legs=pd.DataFrame(),
        settlements=settlements,
        exclusions=pd.DataFrame(
            {"target_date": ["2026-05-17", "2026-05-18"], "reason": ["market_unavailable"] * 2}
        ),
        bootstrap_samples=50,
        seed=4,
        forecast_lineages={"april_2026": "checkpoint-a", "may_2026": "checkpoint-b"},
    )

    by_evaluation = {row["evaluation_id"]: row for row in report["execution"]["by_evaluation"]}
    assert by_evaluation["april_2026"]["event_days"] == 3
    assert by_evaluation["april_2026"]["traded_event_days"] == 2
    assert by_evaluation["may_2026"]["total_pnl_usd"] == pytest.approx(0.0)
    strata = {
        (row["evaluation_id"], row["horizon_band"])
        for row in report["execution"]["by_evaluation_horizon"]
    }
    assert ("april_2026", "30-36h") in strata
    assert ("may_2026", "36-42h") in strata
    assert "by_evaluation_model_market_gap" in report["execution"]
    calibration = report["calibration_at_entry"]["overall"]
    assert calibration["mean_hit_minus_lcb"] == pytest.approx(-0.15)
    assert "by_evaluation_model_market_gap" in report["calibration_at_entry"]
    assert report["limitations"]["same_forecast_lineage_across_months"] is False
    question_ids = {row["question_id"] for row in report["decision_ledger"]}
    assert question_ids == {
        "frozen_april_taker_pnl",
        "frozen_may_taker_pnl",
        "thirty_to_forty_two_hour_persistence",
        "forty_two_to_forty_eight_hour_risk",
        "frozen_lcb_credibility",
    }


def test_frozen_diagnostic_artifact_is_immutable_and_hash_complete(tmp_path) -> None:
    output = tmp_path / "frozen-diagnostic"
    frames = {
        "selection_panel": pd.DataFrame({"event_id": ["a"]}),
        "calibration_coefficients": pd.DataFrame({"coefficient_kind": ["point"]}),
        "decisions": pd.DataFrame({"event_id": ["a"]}),
        "execution_legs": pd.DataFrame({"event_id": ["a"]}),
        "event_pnl": pd.DataFrame({"event_id": ["a"]}),
        "exclusions": pd.DataFrame({"target_date": ["2026-05-17"]}),
    }

    manifest = write_frozen_union_diagnostic_artifact(
        output=output,
        frames=frames,
        report={"schema_version": 1},
        input_sha256={"historical_a2_manifest": "abc"},
    )

    assert manifest["data_class"] == "frozen_two_bucket_union_april_may_diagnostic"
    assert set(manifest["sha256"]) == {
        "selection_panel.parquet",
        "calibration_coefficients.parquet",
        "decisions.parquet",
        "execution_legs.parquet",
        "event_pnl.parquet",
        "exclusions.parquet",
        "report.json",
    }
    with pytest.raises(FileExistsError):
        write_frozen_union_diagnostic_artifact(
            output=output,
            frames=frames,
            report={},
            input_sha256={},
        )
