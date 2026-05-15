"""Tests for core/strategy/ -- EdgeStrategy and MultiStrategy."""
from __future__ import annotations

import pytest

from core.strategy.base import BaseStrategy
from core.strategy.edge import EdgeStrategy
from core.types import Signal


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


def _book(token_id: str, mid: float) -> dict[str, dict]:
    return {token_id: {"mid_price": mid}}


# -----------------------------------------------------------------------
# Basic signal generation
# -----------------------------------------------------------------------

def test_buy_signal_when_edge_exceeds_threshold():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.55}   # model says 55%
    books   = _book("tok1", 0.48)     # CLOB says 48%  => edge = 0.07
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].side == "BUY"
    assert abs(sigs[0].edge - 0.07) < 1e-9


def test_sell_signal_when_edge_below_negative_threshold():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.30}   # model says 30%
    books   = _book("tok1", 0.38)     # CLOB says 38%  => edge = -0.08
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].side == "SELL"


def test_no_signal_when_edge_below_threshold():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.50}
    books   = _book("tok1", 0.49)     # edge = 0.01  < min_edge
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


def test_no_signal_below_threshold():
    """Edge below min_edge produces no signal (min_edge=0.05, edge~=0.04)."""
    strat   = EdgeStrategy(min_edge=0.05)
    probs   = {(60.0, 70.0): 0.54}   # edge ~0.04 < 0.05
    books   = _book("tok1", 0.50)
    markets = [_market(60.0, 70.0, "tok1")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


# -----------------------------------------------------------------------
# NO-outcome tokens are skipped
# -----------------------------------------------------------------------

def test_no_outcome_tokens_are_skipped():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.70}
    books   = _book("tok-no", 0.30)
    markets = [_market(60.0, 70.0, "tok-no", outcome="No")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


# -----------------------------------------------------------------------
# Missing prob / missing book
# -----------------------------------------------------------------------

def test_missing_prob_produces_no_signal():
    strat   = EdgeStrategy()
    probs   = {}   # no prob for (60,70)
    books   = _book("tok1", 0.40)
    markets = [_market(60.0, 70.0, "tok1")]
    assert strat.compute_signals(probs, books, markets) == []


def test_missing_book_produces_no_signal():
    strat   = EdgeStrategy()
    probs   = {(60.0, 70.0): 0.70}
    books   = {}   # no book entry
    markets = [_market(60.0, 70.0, "tok1")]
    assert strat.compute_signals(probs, books, markets) == []


# -----------------------------------------------------------------------
# Multiple markets -- sorting by absolute edge
# -----------------------------------------------------------------------

def test_multiple_signals_sorted_by_abs_edge():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {
        (60.0, 70.0): 0.60,   # edge 0.60 - 0.40 = 0.20
        (70.0, 80.0): 0.80,   # edge 0.80 - 0.45 = 0.35
    }
    books   = {"tok1": {"mid_price": 0.40}, "tok2": {"mid_price": 0.45}}
    markets = [
        _market(60.0, 70.0, "tok1"),
        _market(70.0, 80.0, "tok2"),
    ]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 2
    # Strongest edge first
    assert sigs[0].token_id == "tok2"
    assert sigs[1].token_id == "tok1"


# -----------------------------------------------------------------------
# Signal fields
# -----------------------------------------------------------------------

def test_signal_fields_populated():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.60}
    books   = _book("tok1", 0.40)
    markets = [_market(60.0, 70.0, "tok1", condition_id="cond-99")]

    sig = strat.compute_signals(probs, books, markets)[0]
    assert sig.token_id     == "tok1"
    assert sig.mc_prob      == 0.60
    assert sig.clob_price   == 0.40
    assert sig.bucket_lo_f  == 60.0
    assert sig.bucket_hi_f  == 70.0
    assert sig.condition_id == "cond-99"


def test_empty_markets_returns_empty():
    strat = EdgeStrategy()
    assert strat.compute_signals({}, {}, []) == []


# -----------------------------------------------------------------------
# EdgeStrategy.name
# -----------------------------------------------------------------------

def test_edge_strategy_name():
    assert EdgeStrategy().name == "edge"


# -----------------------------------------------------------------------
# EdgeStrategy -- NO token path
# -----------------------------------------------------------------------

def test_no_token_buy_signal_when_no_edge_exceeds_threshold():
    """NO edge = (1 - mc_prob) - clob_price_no; BUY when NO edge > min_edge."""
    strat   = EdgeStrategy(min_edge=0.04)
    # mc_prob for YES = 0.30 -> NO fair value = 0.70
    # clob_price for NO token = 0.60  => NO edge = 0.70 - 0.60 = 0.10 > 0.04
    probs   = {(60.0, 70.0): 0.30}
    books   = {"tok-no": {"mid_price": 0.60}}
    markets = [_market(60.0, 70.0, "tok-no", outcome="No")]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side == "BUY"
    assert abs(sig.edge - 0.10) < 1e-9
    assert sig.mc_prob == pytest.approx(0.30)
    assert sig.clob_price == pytest.approx(0.60)


def test_no_token_no_signal_when_no_edge_below_threshold():
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.50}   # NO fair = 0.50
    books   = {"tok-no": {"mid_price": 0.48}}  # NO edge = 0.02 < 0.04
    markets = [_market(60.0, 70.0, "tok-no", outcome="No")]

    sigs = strat.compute_signals(probs, books, markets)
    assert sigs == []


def test_no_token_sell_signal_when_no_edge_negative():
    """NO edge < -min_edge => SELL signal."""
    strat   = EdgeStrategy(min_edge=0.04)
    probs   = {(60.0, 70.0): 0.80}   # NO fair = 0.20
    books   = {"tok-no": {"mid_price": 0.30}}  # NO edge = 0.20 - 0.30 = -0.10
    markets = [_market(60.0, 70.0, "tok-no", outcome="No")]

    sigs = strat.compute_signals(probs, books, markets)
    assert len(sigs) == 1
    assert sigs[0].side == "SELL"
    assert abs(sigs[0].edge - (-0.10)) < 1e-9


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

    sig_a = Signal(token_id="tok-a", side="BUY", mc_prob=0.6,
                   clob_price=0.4, edge=0.2)
    sig_b = Signal(token_id="tok-b", side="BUY", mc_prob=0.7,
                   clob_price=0.5, edge=0.2)

    ms = MultiStrategy([ConstantStrategy([sig_a]), ConstantStrategy([sig_b])])
    result = ms.compute_signals({}, {}, [])

    token_ids = {s.token_id for s in result}
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

    sig_first  = Signal(token_id="tok1", side="BUY",  mc_prob=0.7,
                        clob_price=0.4, edge=0.3)
    sig_second = Signal(token_id="tok1", side="SELL", mc_prob=0.2,
                        clob_price=0.4, edge=-0.2)

    ms = MultiStrategy([ConstantStrategy([sig_first]),
                        ConstantStrategy([sig_second])])
    result = ms.compute_signals({}, {}, [])

    assert len(result) == 1
    assert result[0].side == "BUY"   # first strategy wins


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
