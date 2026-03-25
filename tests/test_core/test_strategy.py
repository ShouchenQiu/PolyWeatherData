"""Tests for core/strategy/ -- EdgeStrategy."""
from __future__ import annotations

import pytest

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
