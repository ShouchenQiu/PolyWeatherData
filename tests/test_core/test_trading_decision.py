from __future__ import annotations

from core.trading.decision import (
    FeeSchedule,
    PortfolioState,
    TradingDecisionKernel,
)
from core.types import DirectionalSignal


def _signal(direction: str) -> DirectionalSignal:
    return DirectionalSignal(
        yes_token_id="yes",
        no_token_id="no",
        direction=direction,
        model_prob=0.70,
        reference_price=0.50,
        edge=0.20,
        condition_id="condition",
    )


def _books() -> dict[str, dict]:
    return {
        "yes": {
            "bids": [{"price": 0.49, "size": 1000.0}],
            "asks": [{"price": 0.50, "size": 1000.0}],
        },
        "no": {
            "bids": [{"price": 0.29, "size": 1000.0}],
            "asks": [{"price": 0.30, "size": 1000.0}],
        },
    }


def test_bullish_signal_sells_held_no_before_buying_yes() -> None:
    kernel = TradingDecisionKernel()
    portfolio = PortfolioState(
        cash_usd=400.0,
        shares_by_token={"no": 10.0},
        notional_by_token={"no": 3.0},
        event_by_token={"no": "event", "yes": "event"},
    )

    decision = kernel.decide(_signal("bullish"), _books(), portfolio)

    assert decision.approved
    assert decision.token_id == "no"
    assert decision.side == "SELL"
    assert decision.size_usd == 2.9
    assert decision.reason is None


def test_bullish_signal_buys_yes_for_fixed_size_without_opposing_inventory() -> None:
    kernel = TradingDecisionKernel()
    portfolio = PortfolioState(
        cash_usd=500.0,
        event_by_token={"yes": "event", "no": "event"},
    )

    decision = kernel.decide(_signal("bullish"), _books(), portfolio)

    assert decision.approved
    assert decision.token_id == "yes"
    assert decision.side == "BUY"
    assert decision.size_usd == 50.0
    assert decision.executable_price == 0.50


def test_buy_is_rejected_instead_of_shrunk_when_gross_cap_has_only_partial_room() -> None:
    kernel = TradingDecisionKernel()
    portfolio = PortfolioState(
        cash_usd=500.0,
        notional_by_token={"other": 470.0},
        event_by_token={"yes": "event", "no": "event", "other": "other-event"},
    )

    decision = kernel.decide(_signal("bullish"), _books(), portfolio)

    assert not decision.approved
    assert decision.reason == "gross_cap"


def test_buy_is_rejected_when_event_cap_would_be_exceeded() -> None:
    kernel = TradingDecisionKernel()
    portfolio = PortfolioState(
        cash_usd=500.0,
        notional_by_token={"other-bucket": 120.0},
        event_by_token={
            "yes": "event",
            "no": "event",
            "other-bucket": "event",
        },
    )

    decision = kernel.decide(_signal("bullish"), _books(), portfolio)

    assert not decision.approved
    assert decision.reason == "event_cap"


def test_taker_fee_is_included_in_cash_and_executable_edge() -> None:
    kernel = TradingDecisionKernel()
    portfolio = PortfolioState(
        cash_usd=500.0,
        event_by_token={"yes": "event", "no": "event"},
    )

    decision = kernel.decide(
        _signal("bullish"),
        _books(),
        portfolio,
        fee_schedule=FeeSchedule(enabled=True, rate=0.05, exponent=1.0),
    )

    assert decision.approved
    assert decision.fee_usd == 1.25
    assert decision.cash_delta_usd == -51.25
    assert decision.edge == 0.1875
