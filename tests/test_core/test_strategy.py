"""Tests for core/strategy/ -- EdgeStrategy and MultiStrategy."""
from __future__ import annotations

import pytest

from core.strategy.base import BaseStrategy
from core.strategy.edge import EdgeStrategy
from core.types import DirectionalSignal


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _market(lo_f: float, hi_f: float, token_id: str,
            outcome: str = "Yes", condition_id: str = "c1") -> dict:
    return {
        "token_id":     token_id,
        "bucket_lo_f":  lo_f,
        "bucket_hi_f":  hi_f,
        "outcome":      outcome,
        "condition_id": condition_id,
    }


def _l2_book(token_id: str, ask_price: float,
             bid_price: float | None = None) -> dict[str, dict]:
    """L2 book with 1000-share depth so quote_size_usd always fills."""
    if bid_price is None:
        bid_price = max(0.01, ask_price - 0.02)
    return {
        token_id: {
            "bids": [{"price": bid_price, "size": 1000.0}],
            "asks": [{"price": ask_price, "size": 1000.0}],
        }
    }


# -----------------------------------------------------------------------
# Basic signal generation
# -----------------------------------------------------------------------

def test_bullish_signal_when_edge_exceeds_threshold():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.55}   # model says 55%
    books   = _l2_book("tok1", 0.48) # YES ask = 0.48  => edge = 0.07
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].direction == "bullish"
    assert abs(sigs[0].edge - 0.07) < 1e-9


def test_bearish_signal_when_yes_overpriced():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.30}   # model says 30%
    books   = _l2_book("tok1", 0.38) # YES ask = 0.38  => edge = -0.08
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].direction == "bearish"
    assert abs(sigs[0].edge - 0.08) < 1e-9


def test_no_signal_when_edge_below_threshold():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.50}
    books   = _l2_book("tok1", 0.49)  # edge = 0.01  < min_edge
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


def test_no_signal_below_threshold():
    """Edge below min_edge produces no signal (min_edge=0.05, edge~=0.04)."""
    strat   = EdgeStrategy(min_edge=0.05)
    probs   = {(60.0, 70.0): 0.54}   # edge ~0.04 < 0.05
    books   = _l2_book("tok1", 0.50)
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


# -----------------------------------------------------------------------
# NO-outcome market alone produces no signal (YES drives signal generation)
# -----------------------------------------------------------------------

def test_no_outcome_market_produces_no_signal():
    """A bucket with only a NO-outcome market does not produce a signal.

    Signal generation is driven by YES market; NO token_id is carried
    in the signal for arbitration but does not trigger its own signal.
    """
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.70}
    books   = _l2_book("tok-no", 0.30)
    markets = [_market(60.0, 70.0, "tok-no", outcome="No")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


def test_bullish_signal_includes_no_token_id():
    """When YES and NO markets exist for the same bucket, no_token_id is populated."""
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.70}
    books   = _l2_book("tok-yes", 0.50)
    markets = [
        _market(60.0, 70.0, "tok-yes", outcome="Yes"),
        _market(60.0, 70.0, "tok-no",  outcome="No"),
    ]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].yes_token_id == "tok-yes"
    assert sigs[0].no_token_id  == "tok-no"


def test_bearish_signal_includes_no_token_id():
    """Bearish signal also carries no_token_id for SELL YES vs BUY NO arbitration."""
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.30}
    books   = _l2_book("tok-yes", 0.45)
    markets = [
        _market(60.0, 70.0, "tok-yes", outcome="Yes"),
        _market(60.0, 70.0, "tok-no",  outcome="No"),
    ]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].direction   == "bearish"
    assert sigs[0].no_token_id == "tok-no"


# -----------------------------------------------------------------------
# Incomplete fill: skip signal
# -----------------------------------------------------------------------

def test_incomplete_fill_produces_no_signal():
    """When the YES book lacks depth for quote_size_usd, no signal is emitted."""
    strat   = EdgeStrategy(min_edge=0.04, quote_size_usd=50.0)
    probs   = {(60.0, 70.0): 0.70}
    books   = {
        "tok1": {
            "bids": [{"price": 0.40, "size": 0.1}],  # only 0.1 share = ~$0.04
            "asks": [{"price": 0.50, "size": 0.1}],  # not enough to fill $50
        }
    }
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


# -----------------------------------------------------------------------
# Missing prob / missing book
# -----------------------------------------------------------------------

def test_missing_prob_produces_no_signal():
    strat   = EdgeStrategy()
    probs   = {}   # no prob for (60,70)
    books   = _l2_book("tok1", 0.40)
    markets = [_market(60.0, 70.0, "tok1")]
    assert strat.compute_signals(probs, books, markets) == []


def test_missing_book_produces_no_signal():
    strat   = EdgeStrategy()
    probs   = {(60.0, 70.0): 0.70}
    books   = {}   # no book entry
    markets = [_market(60.0, 70.0, "tok1")]
    assert strat.compute_signals(probs, books, markets) == []


# -----------------------------------------------------------------------
# Multiple markets -- sorting by edge
# -----------------------------------------------------------------------

def test_multiple_signals_sorted_by_edge():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {
        (60.0, 70.0): 0.60,   # edge 0.60 - 0.40 = 0.20
        (70.0, 80.0): 0.80,   # edge 0.80 - 0.45 = 0.35
    }
    books   = {
        "tok1": {
            "bids": [{"price": 0.38, "size": 1000.0}],
            "asks": [{"price": 0.40, "size": 1000.0}],
        },
        "tok2": {
            "bids": [{"price": 0.43, "size": 1000.0}],
            "asks": [{"price": 0.45, "size": 1000.0}],
        },
    }
    markets = [
        _market(60.0, 70.0, "tok1"),
        _market(70.0, 80.0, "tok2"),
    ]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 2
    # Strongest edge first
    assert sigs[0].yes_token_id == "tok2"
    assert sigs[1].yes_token_id == "tok1"


# -----------------------------------------------------------------------
# Signal fields
# -----------------------------------------------------------------------

def test_directional_signal_fields_populated():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.60}
    books   = _l2_book("tok1", 0.40)
    markets = [_market(60.0, 70.0, "tok1", condition_id="cond-99")]

    sig = strat.compute_signals(probs, books, markets)[0]
    assert sig.yes_token_id    == "tok1"
    assert sig.model_prob      == 0.60
    assert sig.reference_price == pytest.approx(0.40)
    assert sig.bucket_lo_f     == 60.0
    assert sig.bucket_hi_f     == 70.0
    assert sig.condition_id    == "cond-99"
    assert sig.direction       == "bullish"


def test_empty_markets_returns_empty():
    strat = EdgeStrategy()
    assert strat.compute_signals({}, {}, []) == []


# -----------------------------------------------------------------------
# EdgeStrategy.name
# -----------------------------------------------------------------------

def test_edge_strategy_name():
    assert EdgeStrategy().name == "edge"


# -----------------------------------------------------------------------
# MultiStrategy
# -----------------------------------------------------------------------

def test_multi_strategy_import():
    from core.strategy.multi import MultiStrategy  # noqa: F401


def test_multi_strategy_name():
    from core.strategy.multi import MultiStrategy
    ms = MultiStrategy([EdgeStrategy()])
    assert ms.name == "multi"


def test_multi_strategy_empty_list_raises():
    from core.strategy.multi import MultiStrategy
    with pytest.raises(ValueError, match="at least one"):
        MultiStrategy([])


def test_multi_strategy_merges_signals_from_two_strategies():
    """Signals from both sub-strategies are collected."""
    from core.strategy.multi import MultiStrategy

    class ConstantStrategy(BaseStrategy):
        def __init__(self, signals):
            self._signals = signals

        @property
        def name(self):
            return "const"

        def compute_signals(self, probs, books, markets, open_positions=None):
            return list(self._signals)

    sig_a = DirectionalSignal(yes_token_id="tok-a", no_token_id="", direction="bullish",
                               model_prob=0.6, reference_price=0.4, edge=0.2)
    sig_b = DirectionalSignal(yes_token_id="tok-b", no_token_id="", direction="bullish",
                               model_prob=0.7, reference_price=0.5, edge=0.2)

    ms = MultiStrategy([ConstantStrategy([sig_a]), ConstantStrategy([sig_b])])
    result = ms.compute_signals({}, {}, [])

    token_ids = {s.yes_token_id for s in result}
    assert "tok-a" in token_ids
    assert "tok-b" in token_ids


def test_multi_strategy_deduplicates_by_token_id_first_wins():
    """When two strategies emit signals for the same token_id, first wins."""
    from core.strategy.multi import MultiStrategy

    class ConstantStrategy(BaseStrategy):
        def __init__(self, signals):
            self._signals = signals

        @property
        def name(self):
            return "const"

        def compute_signals(self, probs, books, markets, open_positions=None):
            return list(self._signals)

    sig_first  = DirectionalSignal(yes_token_id="tok1", no_token_id="", direction="bullish",
                                    model_prob=0.7, reference_price=0.4, edge=0.3)
    sig_second = DirectionalSignal(yes_token_id="tok1", no_token_id="", direction="bearish",
                                    model_prob=0.2, reference_price=0.4, edge=0.2)

    ms = MultiStrategy([ConstantStrategy([sig_first]),
                        ConstantStrategy([sig_second])])
    result = ms.compute_signals({}, {}, [])

    assert len(result) == 1
    assert result[0].direction == "bullish"   # first strategy wins


def test_multi_passes_open_positions_to_sub_strategies():
    """open_positions kwarg is forwarded to each sub-strategy."""
    from core.strategy.multi import MultiStrategy

    received: list[dict | None] = []

    class SpyStrategy(BaseStrategy):
        @property
        def name(self):
            return "spy"

        def compute_signals(self, probs, books, markets, open_positions=None):
            received.append(open_positions)
            return []

    ms = MultiStrategy([SpyStrategy()])
    ms.compute_signals({}, {}, [], open_positions={"tok1": 2})
    assert received == [{"tok1": 2}]
