from __future__ import annotations

import pandas as pd
import pytest

from backtest.two_bucket_union_study import (
    build_two_bucket_report,
    build_two_bucket_candidate_panel,
    fit_expanding_union_calibration,
    fit_frozen_union_calibration,
    price_equal_share_taker_union,
    replay_maker_unions,
    replay_taker_unions,
    select_model_two_bucket_unions,
    simulate_strict_maker_fills,
    write_two_bucket_artifact,
)


def test_selects_highest_mass_adjacent_pair_without_using_outcome() -> None:
    panel = pd.DataFrame(
        {
            "event_id": ["event-1"] * 4,
            "target_date": ["2026-02-01"] * 4,
            "timestamp": pd.to_datetime(["2026-01-31T18:00:00Z"] * 4),
            "snapshot_id": ["snapshot-1"] * 4,
            "bucket_order": [0, 1, 2, 3],
            "condition_id": ["c0", "c1", "c2", "c3"],
            "bucket_lo_f": [float("-inf"), 35.0, 37.0, 39.0],
            "bucket_hi_f": [35.0, 37.0, 39.0, float("inf")],
            "model_probability": [0.10, 0.30, 0.40, 0.20],
            "midpoint_probability": [0.20, 0.20, 0.30, 0.30],
            "yes_outcome": [0.0, 0.0, 0.0, 1.0],
            "lead_hours": [30.0] * 4,
            "lead_band": [">=24h"] * 4,
        }
    )

    selected = select_model_two_bucket_unions(panel.drop(columns="yes_outcome"))

    assert len(selected) == 1
    row = selected.iloc[0]
    assert row["lower_condition_id"] == "c1"
    assert row["upper_condition_id"] == "c2"
    assert row["model_union_probability"] == 0.70
    assert row["modal_neighbor_direction"] == "lower"


def test_taker_union_uses_equal_shares_with_total_risk_cap() -> None:
    lower_book = {
        "bids": [{"price": 0.38, "size": 200.0}],
        "asks": [{"price": 0.40, "size": 200.0}],
    }
    upper_book = {
        "bids": [{"price": 0.48, "size": 200.0}],
        "asks": [{"price": 0.50, "size": 200.0}],
    }

    execution = price_equal_share_taker_union(
        (lower_book, upper_book),
        fee_schedules=(
            {"fees_enabled": False, "fee_rate": None, "fee_exponent": 1.0},
            {"fees_enabled": False, "fee_rate": None, "fee_exponent": 1.0},
        ),
        risk_usd=50.0,
    )

    assert execution.complete_fill is True
    assert execution.shares == pytest.approx(50.0 / 0.90)
    assert execution.leg_quotes[0].filled_shares == pytest.approx(execution.shares)
    assert execution.leg_quotes[1].filled_shares == pytest.approx(execution.shares)
    assert execution.total_cost_usd == pytest.approx(50.0)


def test_maker_fill_requires_future_strict_trade_through_and_tracks_orphan() -> None:
    orders = pd.DataFrame(
        {
            "leg_id": ["lower", "upper"],
            "condition_id": ["c1", "c2"],
            "placed_at": pd.to_datetime(["2026-02-01T12:00:00Z"] * 2),
            "expires_at": pd.to_datetime(["2026-02-01T13:00:00Z"] * 2),
            "limit_price": [0.40, 0.50],
            "target_shares": [10.0, 10.0],
        }
    )
    trades = pd.DataFrame(
        {
            "condition_id": ["c1", "c1", "c2", "c2"],
            "timestamp": pd.to_datetime(
                [
                    "2026-02-01T11:59:00Z",  # Before placement.
                    "2026-02-01T12:10:00Z",  # Strictly through c1 bid.
                    "2026-02-01T12:15:00Z",  # Same-price touch only.
                    "2026-02-01T13:01:00Z",  # Strict, but after expiry.
                ]
            ),
            "outcome": ["YES", "YES", "YES", "NO"],
            "side": ["SELL", "SELL", "SELL", "BUY"],
            "price": [0.30, 0.39, 0.50, 0.51],
            "size": [20.0, 10.0, 10.0, 20.0],
        }
    )

    fills = simulate_strict_maker_fills(orders, trades).set_index("leg_id")

    assert fills.loc["lower", "filled_shares"] == pytest.approx(10.0)
    assert fills.loc["lower", "fill_status"] == "full"
    assert fills.loc["upper", "filled_shares"] == 0.0
    assert fills.loc["upper", "touch_unknown_shares"] == pytest.approx(10.0)
    assert fills.loc["upper", "fill_status"] == "none"


def test_expanding_calibration_never_reads_current_fold_outcome() -> None:
    rows = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "target_date": ["2025-04-01", "2025-04-02", "2025-06-01", "2025-06-02"],
            "fold_id": ["fold-00", "fold-00", "fold-01", "fold-01"],
            "model_union_probability": [0.8, 0.6, 0.7, 0.9],
            "union_outcome": [1.0, 0.0, 1.0, 0.0],
        }
    )

    calibrated, contracts = fit_expanding_union_calibration(
        rows, bootstrap_samples=20, seed=7
    )
    changed = rows.copy()
    changed.loc[changed["fold_id"] == "fold-01", "union_outcome"] = [0.0, 1.0]
    changed_calibrated, _ = fit_expanding_union_calibration(
        changed, bootstrap_samples=20, seed=7
    )

    assert set(calibrated["fold_id"]) == {"fold-01"}
    assert contracts.loc[0, "fit_fold_ids"] == ["fold-00"]
    assert calibrated["calibrated_union_probability"].tolist() == pytest.approx(
        changed_calibrated["calibrated_union_probability"].tolist()
    )
    assert calibrated["union_probability_lower_90"].tolist() == pytest.approx(
        changed_calibrated["union_probability_lower_90"].tolist()
    )


def test_frozen_calibration_scores_diagnostics_without_reading_their_outcomes() -> None:
    training = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "target_date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
            "model_union_probability": [0.55, 0.65, 0.75, 0.85],
            "union_outcome": [0.0, 1.0, 0.0, 1.0],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "event_id": ["may-a", "may-b"],
            "model_union_probability": [0.60, 0.90],
            "union_outcome": [0.0, 1.0],
        }
    )

    calibration = fit_frozen_union_calibration(
        training,
        bootstrap_samples=20,
        seed=9,
    )
    scored = calibration.apply(diagnostics)
    changed = diagnostics.copy()
    changed["union_outcome"] = 1.0 - changed["union_outcome"]
    changed_scored = calibration.apply(changed)

    assert calibration.fit_target_end == "2026-03-04"
    assert calibration.fit_event_days == 4
    assert scored["calibrated_union_probability"].tolist() == pytest.approx(
        changed_scored["calibrated_union_probability"].tolist()
    )
    assert scored["union_probability_lower_90"].tolist() == pytest.approx(
        changed_scored["union_probability_lower_90"].tolist()
    )


def test_frozen_taker_replay_enters_first_positive_lcb_edge_and_settles_union() -> None:
    candidates = pd.DataFrame(
        {
            "event_id": ["event-1", "event-1"],
            "target_date": ["2026-02-01"] * 2,
            "timestamp": pd.to_datetime(["2026-01-31T12:00:00Z", "2026-01-31T13:00:00Z"]),
            "snapshot_id": ["s1", "s2"],
            "lower_condition_id": ["c1", "c1"],
            "upper_condition_id": ["c2", "c2"],
            "lower_yes_outcome": [0.0, 0.0],
            "upper_yes_outcome": [1.0, 1.0],
            "union_outcome": [1.0, 1.0],
            "model_union_probability": [0.60, 0.75],
            "union_probability_lower_90": [0.45, 0.70],
            "calibrated_union_probability": [0.50, 0.74],
            "midpoint_union_probability": [0.48, 0.49],
            "lead_hours": [36.0, 35.0],
            "lead_band": [">=24h", ">=24h"],
            "horizon_band": ["36-42h", "30-36h"],
            "evaluation_id": ["may_2026", "may_2026"],
            "probability_source": ["model_plus_market", "model_plus_market"],
            "fold_id": ["fold-05", "fold-05"],
            "coverage_type": ["forecast_only", "forecast_only"],
            "lower_bid_price_yes": [0.19, 0.19],
            "lower_ask_price_yes": [0.20, 0.20],
            "upper_bid_price_yes": [0.29, 0.29],
            "upper_ask_price_yes": [0.30, 0.30],
            "lower_capacity_usdc": [100.0, 100.0],
            "upper_capacity_usdc": [100.0, 100.0],
            "lower_impact_per_usdc": [0.0, 0.0],
            "upper_impact_per_usdc": [0.0, 0.0],
            "lower_fees_enabled": [False, False],
            "upper_fees_enabled": [False, False],
            "lower_fee_rate": [None, None],
            "upper_fee_rate": [None, None],
            "lower_fee_exponent": [1.0, 1.0],
            "upper_fee_exponent": [1.0, 1.0],
            "lower_yes_token_id": ["t1", "t1"],
            "upper_yes_token_id": ["t2", "t2"],
        }
    )

    replay = replay_taker_unions(
        candidates, stage="frozen", edge_threshold=0.0, risk_usd=50.0
    )

    assert len(replay.settlements) == 1
    assert replay.settlements.loc[0, "timestamp"] == candidates.loc[1, "timestamp"]
    assert replay.settlements.loc[0, "payout_usd"] == pytest.approx(100.0)
    assert replay.settlements.loc[0, "realized_pnl_usd"] == pytest.approx(50.0)
    assert replay.settlements.loc[0, "evaluation_id"] == "may_2026"
    assert replay.settlements.loc[0, "horizon_band"] == "30-36h"
    assert replay.settlements.loc[0, "union_probability_lower_90"] == pytest.approx(0.70)
    assert replay.settlements.loc[0, "midpoint_union_probability"] == pytest.approx(0.49)
    assert replay.settlements.loc[0, "probability_source"] == "model_plus_market"
    assert replay.legs["shares"].nunique() == 1


def test_maker_replay_keeps_single_leg_fill_loss_in_pnl() -> None:
    candidates = pd.DataFrame(
        {
            "event_id": ["event-1"],
            "target_date": ["2026-02-01"],
            "timestamp": pd.to_datetime(["2026-01-31T12:00:00Z"]),
            "snapshot_id": ["s1"],
            "lower_condition_id": ["c1"],
            "upper_condition_id": ["c2"],
            "lower_yes_outcome": [0.0],
            "upper_yes_outcome": [1.0],
            "union_outcome": [1.0],
            "model_union_probability": [0.75],
            "lead_hours": [36.0],
            "lead_band": [">=24h"],
            "fold_id": ["fold-05"],
            "coverage_type": ["forecast_only"],
            "lower_bid_price_yes": [0.20],
            "lower_ask_price_yes": [0.22],
            "upper_bid_price_yes": [0.30],
            "upper_ask_price_yes": [0.32],
            "lower_yes_token_id": ["t1"],
            "upper_yes_token_id": ["t2"],
        }
    )
    trades = pd.DataFrame(
        {
            "condition_id": ["c1", "c2"],
            "timestamp": pd.to_datetime(["2026-01-31T12:10:00Z", "2026-01-31T12:10:00Z"]),
            "outcome": ["YES", "YES"],
            "side": ["SELL", "SELL"],
            "price": [0.19, 0.30],
            "size": [100.0, 100.0],
        }
    )

    replay = replay_maker_unions(
        candidates,
        trades=trades,
        stage="unconditional",
        edge_threshold=0.0,
        risk_usd=50.0,
    )

    settlement = replay.settlements.iloc[0]
    assert settlement["basket_fill_status"] == "orphan"
    assert settlement["payout_usd"] == 0.0
    assert settlement["cost_basis_usd"] == pytest.approx(20.0)
    assert settlement["realized_pnl_usd"] == pytest.approx(-20.0)


def test_candidate_panel_binds_snapshot_outcome_quotes_and_market_contract() -> None:
    base = {
        "event_id": ["event-1"] * 3,
        "target_date": ["2026-02-01"] * 3,
        "timestamp": pd.to_datetime(["2026-01-31T12:00:00Z"] * 3),
        "snapshot_id": ["s1"] * 3,
        "bucket_order": [0, 1, 2],
        "condition_id": ["c0", "c1", "c2"],
        "bucket_lo_f": [float("-inf"), 35.0, 37.0],
        "bucket_hi_f": [35.0, 37.0, float("inf")],
        "model_probability": [0.1, 0.5, 0.4],
        "midpoint_probability": [0.2, 0.4, 0.4],
        "yes_outcome": [0.0, 1.0, 0.0],
        "lead_hours": [36.0] * 3,
        "lead_band": [">=24h"] * 3,
    }
    panel = pd.DataFrame(base)
    snapshots = pd.DataFrame(
        {
            "snapshot_id": ["s1"],
            "lineage_id": ["checkpoint-aligned-static-mc-fold-05"],
            "coverage_type": ["forecast_only"],
            "forecast_hour_count": [24],
        }
    )
    quotes = pd.DataFrame(
        {
            "condition_id": ["c0", "c1", "c2"],
            "timestamp": pd.to_datetime(["2026-01-31T12:00:00Z"] * 3),
            "bid_price_yes": [0.1, 0.4, 0.3],
            "ask_price_yes": [0.12, 0.42, 0.32],
            "capacity_usdc": [100.0] * 3,
            "impact_per_usdc": [0.0] * 3,
        }
    )
    markets = pd.DataFrame(
        {
            "condition_id": ["c0", "c1", "c2"],
            "bucket_lo_f": [float("-inf"), 35.0, 37.0],
            "bucket_hi_f": [35.0, 37.0, float("inf")],
            "yes_token_id": ["t0", "t1", "t2"],
            "fees_enabled": [False] * 3,
            "fee_rate": [None] * 3,
            "fee_exponent": [1.0] * 3,
        }
    )

    result = build_two_bucket_candidate_panel(
        panel=panel, snapshots=snapshots, quotes=quotes, markets=markets
    )

    row = result.iloc[0]
    assert row["fold_id"] == "fold-05"
    assert row["coverage_type"] == "forecast_only"
    assert row["lower_condition_id"] == "c1"
    assert row["upper_condition_id"] == "c2"
    assert row["union_outcome"] == 1.0
    assert row["lower_ask_price_yes"] == pytest.approx(0.42)
    assert row["upper_yes_token_id"] == "t2"


def test_candidate_panel_accepts_an_explicit_frozen_diagnostic_period() -> None:
    panel = pd.DataFrame(
        {
            "event_id": ["event-1"] * 2,
            "target_date": ["2026-04-01"] * 2,
            "timestamp": pd.to_datetime(["2026-03-31T12:00:00Z"] * 2),
            "snapshot_id": ["s1"] * 2,
            "bucket_order": [0, 1],
            "condition_id": ["c0", "c1"],
            "bucket_lo_f": [float("-inf"), 35.0],
            "bucket_hi_f": [35.0, float("inf")],
            "model_probability": [0.4, 0.6],
            "midpoint_probability": [0.5, 0.5],
            "yes_outcome": [0.0, 1.0],
            "lead_hours": [40.0, 40.0],
            "lead_band": [">=24h", ">=24h"],
        }
    )
    snapshots = pd.DataFrame(
        {
            "snapshot_id": ["s1"],
            "lineage_id": ["strict-preapril-fixed-epoch-mc"],
            "coverage_type": ["forecast_only"],
            "forecast_hour_count": [24],
        }
    )
    quotes = pd.DataFrame(
        {
            "condition_id": ["c0", "c1"],
            "timestamp": pd.to_datetime(["2026-03-31T12:00:00Z"] * 2),
            "bid_price_yes": [0.39, 0.49],
            "ask_price_yes": [0.41, 0.51],
            "capacity_usdc": [100.0, 100.0],
            "impact_per_usdc": [0.0, 0.0],
        }
    )
    markets = pd.DataFrame(
        {
            "condition_id": ["c0", "c1"],
            "bucket_lo_f": [float("-inf"), 35.0],
            "bucket_hi_f": [35.0, float("inf")],
            "yes_token_id": ["t0", "t1"],
            "fees_enabled": [False, False],
            "fee_rate": [None, None],
            "fee_exponent": [1.0, 1.0],
        }
    )

    result = build_two_bucket_candidate_panel(
        panel=panel,
        snapshots=snapshots,
        quotes=quotes,
        markets=markets,
        evaluation_id="april_2026",
        evaluation_target_range=("2026-04-01", "2026-04-30"),
    )

    assert result["evaluation_id"].tolist() == ["april_2026"]
    assert "fold_id" not in result

    with pytest.raises(ValueError, match="outside explicit evaluation range"):
        build_two_bucket_candidate_panel(
            panel=panel.assign(target_date="2026-05-01"),
            snapshots=snapshots,
            quotes=quotes,
            markets=markets,
            evaluation_id="april_2026",
            evaluation_target_range=("2026-04-01", "2026-04-30"),
        )


def test_report_answers_original_profit_question_with_event_day_bootstrap() -> None:
    candidates = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "target_date": ["2026-02-01", "2026-02-02"],
            "timestamp": pd.to_datetime(["2026-02-01T01:00:00Z", "2026-02-02T01:00:00Z"]),
            "modal_neighbor_direction": ["lower", "upper"],
            "model_union_probability": [0.7, 0.8],
            "midpoint_union_probability": [0.5, 0.6],
            "union_outcome": [1.0, 1.0],
        }
    )
    settlements = pd.DataFrame(
        {
            "stage": ["unconditional", "unconditional"],
            "execution_mode": ["taker", "taker"],
            "edge_threshold": [0.0, 0.0],
            "event_id": ["a", "b"],
            "target_date": ["2026-02-01", "2026-02-02"],
            "realized_pnl_usd": [5.0, 10.0],
            "cost_basis_usd": [50.0, 50.0],
            "payout_usd": [55.0, 60.0],
            "union_outcome": [1.0, 1.0],
            "basket_fill_status": ["full", "full"],
            "fold_id": ["fold-05", "fold-05"],
            "coverage_type": ["forecast_only", "forecast_only"],
            "lead_band": [">=24h", ">=24h"],
        }
    )

    report = build_two_bucket_report(
        candidates=candidates,
        decisions=pd.DataFrame(),
        legs=pd.DataFrame(),
        settlements=settlements,
        bootstrap_samples=20,
        seed=3,
    )

    question = next(
        item for item in report["decision_ledger"]
        if item["question_id"] == "unconditional_taker_absolute_pnl"
    )
    assert question["status"] == "supported"
    assert question["evidence"]["mean_event_day_pnl_usd"] == pytest.approx(7.5)


def test_artifact_is_immutable_and_hashes_every_decision_payload(tmp_path) -> None:
    output = tmp_path / "study"
    frames = {
        "selection_panel": pd.DataFrame({"event_id": ["a"]}),
        "calibration_folds": pd.DataFrame({"fold_id": ["fold-01"]}),
        "decisions": pd.DataFrame({"event_id": ["a"]}),
        "execution_legs": pd.DataFrame({"event_id": ["a"]}),
        "event_pnl": pd.DataFrame({"event_id": ["a"]}),
    }

    manifest = write_two_bucket_artifact(
        output=output,
        frames=frames,
        report={"schema_version": 1},
        input_sha256={"signal_manifest": "abc"},
    )

    assert set(manifest["sha256"]) == {
        "selection_panel.parquet", "calibration_folds.parquet",
        "decisions.parquet", "execution_legs.parquet", "event_pnl.parquet",
        "report.json",
    }
    with pytest.raises(FileExistsError):
        write_two_bucket_artifact(
            output=output, frames=frames, report={}, input_sha256={}
        )
