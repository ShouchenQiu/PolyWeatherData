"""Tests for core/exchange/ -- PaperExchange and Settler."""
from __future__ import annotations

import pytest

from core.exchange.paper import PaperExchange
from core.exchange.settlement import Settler


# -----------------------------------------------------------------------
# PaperExchange.place_order
# -----------------------------------------------------------------------

def test_paper_place_order_returns_string():
    exch = PaperExchange()
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert isinstance(oid, str)
    assert len(oid) > 0


def test_paper_place_order_status_filled():
    exch = PaperExchange()
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    order = exch.get_order(oid)
    assert order is not None
    assert order["status"] == "filled"


def test_paper_place_order_uses_submitted_price_when_no_fetcher():
    exch = PaperExchange(book_fetcher=None)
    exch.place_order("tok1", "BUY", 20.0, 0.55)
    pos = exch.get_positions()
    assert len(pos) == 1
    assert abs(pos[0]["avg_price"] - 0.55) < 1e-9


def test_paper_place_order_uses_mid_price_from_fetcher():
    class FakeFetcher:
        def get_book(self, token_id):
            return {"mid_price": 0.48}

    exch = PaperExchange(book_fetcher=FakeFetcher())
    exch.place_order("tok1", "BUY", 20.0, 0.55)
    pos = exch.get_positions()
    assert abs(pos[0]["avg_price"] - 0.48) < 1e-9


def test_paper_get_positions_after_buy():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 20.0, 0.50)
    positions = exch.get_positions()
    assert len(positions) == 1
    assert positions[0]["token_id"] == "tok1"
    assert positions[0]["size_usd"] == 20.0


def test_paper_sell_reduces_position():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY",  30.0, 0.50)
    exch.place_order("tok1", "SELL", 10.0, 0.55)
    pos = exch.get_positions()
    assert abs(pos[0]["size_usd"] - 20.0) < 1e-9


def test_paper_sell_closes_full_position():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY",  20.0, 0.50)
    exch.place_order("tok1", "SELL", 20.0, 0.55)
    pos = exch.get_positions()
    assert pos == []


def test_paper_cancel_order_pending_returns_true():
    exch = PaperExchange()
    # We must trick the order into pending status
    exch._orders["test-id"] = {"status": "pending", "token_id": "tok1",
                                "size_usd": 10.0, "price": 0.5}
    assert exch.cancel_order("test-id") is True
    assert exch._orders["test-id"]["status"] == "cancelled"


def test_paper_cancel_filled_order_returns_false():
    exch = PaperExchange()
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert exch.cancel_order(oid) is False


def test_paper_multiple_tokens_independent():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 10.0, 0.40)
    exch.place_order("tok2", "BUY", 20.0, 0.60)
    pos = {p["token_id"]: p for p in exch.get_positions()}
    assert abs(pos["tok1"]["size_usd"] - 10.0) < 1e-9
    assert abs(pos["tok2"]["size_usd"] - 20.0) < 1e-9


# -----------------------------------------------------------------------
# Settler
# -----------------------------------------------------------------------

def test_settler_fallback_sells_open_positions():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 25.0, 0.50)

    settler = Settler(relayer_client=None)
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    assert result == "fallback"
    # After settlement the position should be closed
    pos = exch.get_positions()
    assert pos == []


def test_settler_only_closes_relevant_tokens():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 25.0, 0.50)
    exch.place_order("tok2", "BUY", 25.0, 0.50)

    settler = Settler(relayer_client=None)
    settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    # tok2 should still be open
    pos = {p["token_id"]: p for p in exch.get_positions()}
    assert "tok2" in pos
    assert "tok1" not in pos


def test_settler_relayer_success_returns_relayer():
    class FakeRelayer:
        def settle(self, condition_id):
            return True

    exch    = PaperExchange()
    settler = Settler(relayer_client=FakeRelayer())
    result  = settler.settle(
        event = {"condition_id": "c1", "token_ids": []},
        exchange = exch,
    )
    assert result == "relayer"


def test_settler_relayer_failure_falls_back():
    class BrokenRelayer:
        def settle(self, condition_id):
            raise RuntimeError("network error")

    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 10.0, 0.50)
    settler = Settler(relayer_client=BrokenRelayer())
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    assert result == "fallback"
    assert exch.get_positions() == []
