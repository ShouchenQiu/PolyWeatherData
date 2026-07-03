"""Tests for data/clob/orderbook.py -- V2F02 PublicClient migration."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from data.clob.orderbook import ClobRestClient, OrderbookFetcher


# ---------------------------------------------------------------------------
# Fake PublicClient for tests (simulates new SDK: bids ascending, keyword arg)
# ---------------------------------------------------------------------------

def _make_fake_book(bids_asc, asks_asc):
    """Return a fake orderbook object as the new SDK would deliver it.

    Bids arrive ascending (lowest first), asks ascending (lowest first).
    Entries are plain dicts with 'price' and 'size'.
    """
    return SimpleNamespace(
        bids=[{"price": str(p), "size": "100"} for p in bids_asc],
        asks=[{"price": str(p), "size": "100"} for p in asks_asc],
    )


class FakePublicClient:
    """Records calls and returns a preset book; enforces keyword-only arg."""

    def __init__(self, book):
        self._book = book
        self.calls: list[str] = []

    def get_order_book(self, *, token_id: str):
        self.calls.append(token_id)
        return self._book


_DEFAULT_BOOK = _make_fake_book(
    bids_asc=[0.54, 0.56, 0.58],   # ascending from SDK, best = 0.58
    asks_asc=[0.60, 0.62, 0.64],   # ascending,          best = 0.60
)


# ---------------------------------------------------------------------------
# AC: keyword-only argument is used for get_order_book
# ---------------------------------------------------------------------------

def test_get_book_calls_client_with_keyword_token_id():
    client = FakePublicClient(_DEFAULT_BOOK)
    OrderbookFetcher(client).get_book("tok-abc")
    assert client.calls == ["tok-abc"]


# ---------------------------------------------------------------------------
# AC: bids sorted descending (best bid first), asks sorted ascending
# ---------------------------------------------------------------------------

def test_get_book_bids_descending():
    client = FakePublicClient(_DEFAULT_BOOK)
    book = OrderbookFetcher(client).get_book("tok1")
    prices = [lvl["price"] for lvl in book["bids"]]
    assert prices == sorted(prices, reverse=True), "bids must be descending"


def test_get_book_asks_ascending():
    client = FakePublicClient(_DEFAULT_BOOK)
    book = OrderbookFetcher(client).get_book("tok1")
    prices = [lvl["price"] for lvl in book["asks"]]
    assert prices == sorted(prices), "asks must be ascending"


# ---------------------------------------------------------------------------
# AC: best_bid and best_ask keys present and correct
# ---------------------------------------------------------------------------

def test_get_book_best_bid():
    client = FakePublicClient(_DEFAULT_BOOK)
    book = OrderbookFetcher(client).get_book("tok1")
    assert "best_bid" in book
    assert book["best_bid"] == pytest.approx(0.58)


def test_get_book_best_ask():
    client = FakePublicClient(_DEFAULT_BOOK)
    book = OrderbookFetcher(client).get_book("tok1")
    assert "best_ask" in book
    assert book["best_ask"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# AC: mid_price = (best_bid + best_ask) / 2
# ---------------------------------------------------------------------------

def test_get_book_mid_price():
    client = FakePublicClient(_DEFAULT_BOOK)
    book = OrderbookFetcher(client).get_book("tok1")
    assert book["mid_price"] == pytest.approx((0.58 + 0.60) / 2.0)


def test_get_book_mid_price_none_when_empty_bids():
    empty_book = SimpleNamespace(bids=[], asks=[{"price": "0.60", "size": "50"}])
    client = FakePublicClient(empty_book)
    book = OrderbookFetcher(client).get_book("tok1")
    assert book["best_bid"] is None
    assert book["mid_price"] is None


def test_get_book_mid_price_none_when_empty_asks():
    empty_book = SimpleNamespace(bids=[{"price": "0.58", "size": "50"}], asks=[])
    client = FakePublicClient(empty_book)
    book = OrderbookFetcher(client).get_book("tok1")
    assert book["best_ask"] is None
    assert book["mid_price"] is None


# ---------------------------------------------------------------------------
# AC: no py_clob_client import anywhere in the module
# ---------------------------------------------------------------------------

def test_orderbook_module_has_no_py_clob_client_import():
    import inspect
    import data.clob.orderbook as mod
    src = inspect.getsource(mod)
    assert "py_clob_client" not in src


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "asset_id": "tok-rest",
            "bids": [{"price": "0.40", "size": "10"}],
            "asks": [{"price": "0.42", "size": "12"}],
        }


class _Session:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return _Response()


def test_rest_client_reads_public_clob_book_without_optional_sdk() -> None:
    session = _Session()
    client = ClobRestClient(session=session)

    book = OrderbookFetcher(client).get_book("tok-rest")

    assert session.calls == [
        (
            "https://clob.polymarket.com/book",
            {"token_id": "tok-rest"},
            30,
        )
    ]
    assert book["best_bid"] == 0.40
    assert book["best_ask"] == 0.42
