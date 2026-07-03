from __future__ import annotations

import pandas as pd

from backtest.replay import ReplayResult
from backtest.selection import event_day_pnl, select_strategy_candidates


def test_candidate_selection_bootstraps_by_event_day_and_keeps_at_most_two() -> None:
    days = pd.date_range("2026-04-01", periods=12, freq="D")
    results = {
        "steady": pd.DataFrame(
            {"event_day": days, "realized_pnl_usd": [1.0] * 12}
        ),
        "also-positive": pd.DataFrame(
            {"event_day": days, "realized_pnl_usd": [0.5] * 12}
        ),
        "losing": pd.DataFrame(
            {"event_day": days, "realized_pnl_usd": [-1.0] * 12}
        ),
    }

    table = select_strategy_candidates(
        results,
        min_event_days=10,
        confidence=0.90,
        bootstrap_samples=500,
        seed=42,
        max_candidates=2,
    )

    assert table.loc[table["selected"], "strategy"].tolist() == [
        "steady",
        "also-positive",
    ]
    assert not table.loc[table["strategy"] == "losing", "selected"].item()
    assert table.loc[
        table["strategy"] == "steady", "pnl_mean_lcb_90"
    ].item() == 1.0


def test_event_day_pnl_includes_trade_cashflows_sales_fees_and_zero_days() -> None:
    result = ReplayResult(
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "condition-1",
                    "side": "BUY",
                    "size_usd": 50.0,
                    "fee_usd": 1.0,
                },
                {
                    "condition_id": "condition-1",
                    "side": "SELL",
                    "size_usd": 60.0,
                    "fee_usd": 0.5,
                },
            ]
        ),
        settlements=pd.DataFrame(
            columns=["event_id", "payout_usd"]
        ),
        starting_cash_usd=500.0,
        ending_cash_usd=508.5,
    )
    events = pd.DataFrame(
        [
            {"event_id": "event-1", "target_date": "2026-04-01"},
            {"event_id": "event-2", "target_date": "2026-04-02"},
        ]
    )
    markets = pd.DataFrame(
        [{"condition_id": "condition-1", "event_id": "event-1"}]
    )

    daily = event_day_pnl(result, events=events, markets=markets)

    assert daily.to_dict("records") == [
        {"event_day": "2026-04-01", "realized_pnl_usd": 8.5},
        {"event_day": "2026-04-02", "realized_pnl_usd": 0.0},
    ]
