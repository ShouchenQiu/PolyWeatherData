from __future__ import annotations

import math
import json

import pandas as pd
import pytest

from backtest.market_calibration import (
    assign_lead_band,
    build_buy_settlement_returns,
    build_question_ledger,
    build_multibucket_decomposition,
    build_transaction_panel,
    select_calibration_candidates,
    summarize_calibration_bins,
    summarize_fold_stability,
    summarize_price_lead_cells,
    write_market_calibration_artifact,
)


def test_transaction_panel_uses_local_day_boundary_and_token_truth() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["spring", "fall"],
            "target_date": ["2026-03-08", "2025-11-02"],
        }
    )
    markets = pd.DataFrame(
        {
            "event_id": ["spring", "fall", "fall"],
            "condition_id": ["c-spring", "c-fall", "c-fall-winner"],
            "yes_token_id": ["yes-spring", "yes-fall", "yes-fall-winner"],
            "no_token_id": ["no-spring", "no-fall", "no-fall-winner"],
            "bucket_lo_f": [50.0, 40.0, 50.0],
            "bucket_hi_f": [60.0, 50.0, float("inf")],
            "resolved_yes": [True, False, True],
            "closed_at": [
                pd.Timestamp("2026-03-09T04:30:00Z"),
                pd.Timestamp("2025-11-03T04:30:00Z"),
                pd.Timestamp("2025-11-03T05:30:00Z"),
            ],
            "fees_enabled": [False, False, False],
            "fee_rate": [None, None, None],
            "fee_exponent": [None, None, None],
        }
    )
    trades = pd.DataFrame(
        {
            "condition_id": ["c-spring", "c-spring", "c-fall"],
            "token_id": ["yes-spring", "no-spring", "no-fall"],
            "timestamp": pd.to_datetime(
                [
                    "2026-03-09T03:00:00Z",
                    "2026-03-09T04:15:00Z",
                    "2025-11-03T04:45:00Z",
                ],
                utc=True,
            ),
            "outcome": ["YES", "NO", "NO"],
            "side": ["BUY", "BUY", "SELL"],
            "price": [0.6, 0.4, 0.8],
            "size": [10.0, 20.0, 30.0],
            "usdc": [6.0, 8.0, 24.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "target_date": ["2026-03-08", "2025-11-02"],
            "actual_high_f": [55.0, 55.0],
        }
    )

    panel = build_transaction_panel(
        events=events,
        markets=markets,
        trades=trades,
        actuals=actuals,
        timezone="America/New_York",
    )

    assert panel["settlement_boundary"].tolist() == [
        pd.Timestamp("2026-03-09T04:00:00Z"),
        pd.Timestamp("2026-03-09T04:00:00Z"),
        pd.Timestamp("2025-11-03T05:00:00Z"),
    ]
    assert panel["time_regime"].tolist() == [
        "forecast_relevant",
        "resolution_latency",
        "invalid_post_close",
    ]
    assert panel["y_token"].tolist() == [1.0, 0.0, 1.0]
    assert panel["p_token"].tolist() == [0.6, 0.4, 0.8]
    assert panel["lead_hours"].tolist() == [1.0, -0.25, 0.25]
    assert panel["fold_id"].tolist() == ["fold-05", "fold-05", "fold-03"]
    mislabeled = trades.copy()
    mislabeled.loc[0, "outcome"] = "NO"
    with pytest.raises(ValueError, match="outcome label disagrees"):
        build_transaction_panel(
            events=events,
            markets=markets,
            trades=mislabeled,
            actuals=actuals,
            timezone="America/New_York",
        )


def test_calibration_bins_separate_fill_event_and_usdc_weighting() -> None:
    panel = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e2"],
            "condition_id": ["c1", "c1", "c2"],
            "p_token": [0.21, 0.23, 0.22],
            "y_token": [1.0, 1.0, 0.0],
            "usdc": [1.0, 1.0, 8.0],
            "time_regime": ["forecast_relevant"] * 3,
        }
    )

    result = summarize_calibration_bins(
        panel,
        bin_widths=(0.1,),
        bootstrap_samples=100,
        seed=7,
        min_event_days=2,
        min_condition_cells=2,
    )

    rows = result.set_index("weighting")
    assert rows.loc["fill", "fills"] == 3
    assert rows.loc["fill", "condition_cells"] == 2
    assert rows.loc["fill", "event_days"] == 2
    assert rows.loc["fill", "empirical_rate"] == 2 / 3
    assert rows.loc["fill", "mean_price"] == (0.21 + 0.23 + 0.22) / 3
    assert rows.loc["event_contract_equal", "empirical_rate"] == 0.5
    assert rows.loc["event_contract_equal", "mean_price"] == 0.22
    assert rows.loc["usdc", "empirical_rate"] == 0.2
    assert rows.loc["usdc", "mean_price"] == pytest.approx(0.22)
    assert set(result["inference_status"]) == {"eligible"}
    assert result["price_bin_start"].tolist() == [0.2, 0.2, 0.2]
    assert result["price_bin_end"].tolist() == [0.3, 0.3, 0.3]
    assert result["ci_low"].notna().all()
    assert result["ci_high"].notna().all()


def test_price_lead_cells_use_predeclared_bands_and_simultaneous_intervals() -> None:
    assert assign_lead_band(pd.Series([0.0, 1.0, 42.0, 48.0, 72.0])).tolist() == [
        "0-1h",
        "1-3h",
        "42-48h",
        "48-72h",
        "72h+",
    ]
    panel = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "condition_id": ["c1", "c2", "c3", "c4"],
            "p_token": [0.205, 0.206, 0.405, 0.406],
            "y_token": [1.0, 0.0, 1.0, 0.0],
            "usdc": [1.0, 1.0, 1.0, 1.0],
            "lead_hours": [45.0, 45.0, 35.0, 35.0],
            "time_regime": ["forecast_relevant"] * 4,
        }
    )

    result = summarize_price_lead_cells(
        panel,
        price_bin_width=0.01,
        bootstrap_samples=100,
        seed=11,
        min_event_days=2,
        min_condition_cells=2,
    )

    primary = result[result["weighting"].eq("event_contract_equal")]
    assert set(primary["lead_band"]) == {"42-48h", "30-36h"}
    assert primary["simultaneous_ci_low"].notna().all()
    assert primary["simultaneous_ci_high"].notna().all()
    assert (
        primary["simultaneous_ci_low"] <= primary["ci_low"]
    ).all()
    assert (
        primary["simultaneous_ci_high"] >= primary["ci_high"]
    ).all()
    secondary = result[~result["weighting"].eq("event_contract_equal")]
    assert secondary["simultaneous_ci_low"].isna().all()


def test_candidate_requires_cross_fold_direction_without_opposite_evidence() -> None:
    panel = pd.DataFrame(
        {
            "event_id": [f"e{index}" for index in range(6)],
            "condition_id": [f"c{index}" for index in range(6)],
            "p_token": [0.205] * 6,
            "y_token": [1.0] * 6,
            "usdc": [1.0] * 6,
            "lead_hours": [45.0] * 6,
            "time_regime": ["forecast_relevant"] * 6,
            "fold_id": [f"fold-0{index}" for index in range(6)],
        }
    )
    cells = summarize_price_lead_cells(
        panel,
        bootstrap_samples=50,
        seed=13,
        min_event_days=1,
        min_condition_cells=1,
    )
    stability = summarize_fold_stability(
        panel,
        bootstrap_samples=50,
        seed=13,
    )

    candidates = select_calibration_candidates(cells, stability)

    assert len(candidates) == 1
    assert candidates.iloc[0]["candidate_direction"] == "underpriced_token"
    assert candidates.iloc[0]["consistent_fold_count"] == 6
    contradicted = stability.copy()
    contradicted.loc[contradicted["fold_id"].eq("fold-05"), [
        "calibration_gap",
        "ci_low",
        "ci_high",
    ]] = [-0.2, -0.3, -0.1]
    assert select_calibration_candidates(cells, contradicted).empty


def test_buy_settlement_returns_use_actual_buy_token_and_fee_only() -> None:
    panel = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e1"],
            "target_date": ["2025-04-01"] * 3,
            "condition_id": ["c1", "c1", "c2"],
            "side": ["BUY", "SELL", "BUY"],
            "p_token": [0.505, 0.505, 0.805],
            "y_token": [1.0, 1.0, 0.0],
            "size": [10.0, 20.0, 30.0],
            "usdc": [5.05, 10.1, 24.15],
            "lead_hours": [45.0, 45.0, 45.0],
            "time_regime": ["forecast_relevant"] * 3,
            "fees_enabled": [True, True, True],
            "fee_rate": [0.05, 0.05, 0.05],
            "fee_exponent": [1.0, 1.0, 1.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "price_bin_width": [0.01, 0.01],
            "price_bin_start": [0.50, 0.80],
            "price_bin_end": [0.51, 0.81],
            "lead_band": ["42-48h", "42-48h"],
            "candidate_direction": ["underpriced_token", "overpriced_token"],
        }
    )

    result = build_buy_settlement_returns(panel, candidates)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["fill_count"] == 1
    assert row["shares"] == 10.0
    assert row["observed_usdc"] == 5.05
    assert row["fee_usdc"] == pytest.approx(0.12499)
    assert row["net_settlement_return_usdc"] == pytest.approx(4.82501)
    assert row["net_settlement_return_per_share"] == pytest.approx(0.482501)


def test_multibucket_decomposition_requires_complete_pre_settlement_vector() -> None:
    events = pd.DataFrame(
        {"event_id": ["e1"], "target_date": ["2025-04-01"]}
    )
    markets = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "condition_id": ["c1", "c2"],
            "bucket_lo_f": [float("-inf"), 50.0],
            "bucket_hi_f": [50.0, float("inf")],
            "closed_at": pd.to_datetime(
                ["2025-04-02T05:00:00Z", "2025-04-02T05:00:00Z"], utc=True
            ),
        }
    )
    quotes = pd.DataFrame(
        {
            "condition_id": ["c1", "c2", "c1"],
            "timestamp": pd.to_datetime(
                [
                    "2025-04-02T03:00:00Z",
                    "2025-04-02T03:00:00Z",
                    "2025-04-02T03:05:00Z",
                ],
                utc=True,
            ),
            "mid_price_yes": [0.4, 0.8, 0.5],
        }
    )
    actuals = pd.DataFrame(
        {"target_date": ["2025-04-01"], "actual_high_f": [55.0]}
    )

    result = build_multibucket_decomposition(
        events=events,
        markets=markets,
        quotes=quotes,
        actuals=actuals,
        timezone="America/New_York",
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["midpoint_sum"] == pytest.approx(1.2)
    assert row["midpoint_overround"] == pytest.approx(0.2)
    assert row["multiclass_brier"] == pytest.approx(2 / 9)
    assert row["log_score"] == pytest.approx(-math.log(2 / 3))
    assert row["normalized_rps"] == pytest.approx(1 / 9)


def test_market_calibration_artifact_is_immutable_and_hash_bound(tmp_path) -> None:
    output = tmp_path / "calibration"
    frames = {
        "transaction_panel": pd.DataFrame({"event_id": ["e1"]}),
        "calibration_bins": pd.DataFrame({"calibration_gap": [0.1]}),
    }

    manifest = write_market_calibration_artifact(
        output=output,
        frames=frames,
        report={"status": "complete"},
        question_ledger={
            "executable_edge": {"status": "not_measured", "evidence": []}
        },
        source_sha256={"clob_manifest": "abc", "actuals_manifest": "def"},
        protocol={"split_role": "oof_development"},
    )

    assert manifest["row_counts"] == {
        "transaction_panel": 1,
        "calibration_bins": 1,
    }
    assert set(manifest["sha256"]) == {
        "transaction_panel.parquet",
        "calibration_bins.parquet",
        "report.json",
        "question_ledger.json",
    }
    assert json.loads((output / "question_ledger.json").read_text())[
        "executable_edge"
    ]["status"] == "not_measured"
    with pytest.raises(FileExistsError):
        write_market_calibration_artifact(
            output=output,
            frames=frames,
            report={},
            question_ledger={},
            source_sha256={},
            protocol={},
        )


def test_question_ledger_never_claims_executable_edge() -> None:
    bins = pd.DataFrame(
        {
            "price_bin_width": [0.1],
            "weighting": ["fill"],
            "time_regime": ["forecast_relevant"],
            "inference_status": ["eligible"],
            "ci_low": [0.01],
            "ci_high": [0.05],
        }
    )
    cells = pd.DataFrame(
        {
            "weighting": ["event_contract_equal"],
            "inference_status": ["eligible"],
            "simultaneous_ci_low": [0.01],
            "simultaneous_ci_high": [0.05],
        }
    )
    candidates = pd.DataFrame({"candidate_direction": ["underpriced_token"]})
    returns = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "net_settlement_return_per_share": [0.1, 0.2],
        }
    )

    ledger = build_question_ledger(
        calibration_bins=bins,
        price_lead_cells=cells,
        candidates=candidates,
        buy_returns=returns,
        resolution_latency=pd.DataFrame(),
        bootstrap_samples=100,
        seed=5,
    )

    assert ledger["paper_style_weather_miscalibration"]["status"] == "supported"
    assert ledger["fine_price_time_miscalibration"]["status"] == "supported"
    assert ledger["cross_fold_stability"]["status"] == "supported"
    assert ledger["observed_buy_settlement_return"]["status"] == "supported"
    assert ledger["executable_edge"]["status"] == "not_measured"
