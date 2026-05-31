import pytest

from core.pricing.quote import BookLevel, NormalizedBook, Quote, compute_quote
from config.settings import Settings


# ---------------------------------------------------------------------------
# Quote dataclass fields
# ---------------------------------------------------------------------------


def test_quote_has_required_fields() -> None:
    q = Quote(
        vwap_price=0.61,
        max_fillable_usd=50.0,
        filled_shares=81.97,
        slippage_bps=0.0,
        best_bid=0.59,
        best_ask=0.61,
        mid_price=0.60,
        complete_fill=True,
    )
    assert q.vwap_price == pytest.approx(0.61)
    assert q.max_fillable_usd == pytest.approx(50.0)
    assert q.filled_shares == pytest.approx(81.97)
    assert q.slippage_bps == pytest.approx(0.0)
    assert q.best_bid == pytest.approx(0.59)
    assert q.best_ask == pytest.approx(0.61)
    assert q.mid_price == pytest.approx(0.60)
    assert q.complete_fill is True


def test_settings_has_quote_size_usd() -> None:
    s = Settings()
    assert hasattr(s, "quote_size_usd")
    assert s.quote_size_usd == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# BUY compute_quote
# ---------------------------------------------------------------------------


def test_compute_quote_buy_full_fill_single_level() -> None:
    book = {
        "bids": [{"price": 0.59, "size": 200.0}],
        "asks": [{"price": 0.61, "size": 200.0}],
    }
    q = compute_quote(book, "BUY", 50.0)
    assert q.complete_fill is True
    assert q.vwap_price == pytest.approx(0.61)
    assert q.filled_shares == pytest.approx(50.0 / 0.61, rel=1e-4)
    assert q.max_fillable_usd == pytest.approx(50.0, rel=1e-4)
    assert q.best_ask == pytest.approx(0.61)
    assert q.best_bid == pytest.approx(0.59)
    assert q.mid_price == pytest.approx(0.60)
    assert q.slippage_bps == pytest.approx(0.0, abs=0.01)


def test_compute_quote_buy_multi_level_slippage() -> None:
    book = {
        "bids": [],
        "asks": [
            {"price": 0.60, "size": 50.0},  # 50 shares * 0.60 = 30 USD
            {"price": 0.70, "size": 50.0},  # need 20 more USD -> 20/0.70 = 28.57 shares
        ],
    }
    q = compute_quote(book, "BUY", 50.0)
    assert q.complete_fill is True
    expected_shares = 50.0 + 20.0 / 0.70
    assert q.filled_shares == pytest.approx(expected_shares, rel=1e-4)
    assert q.vwap_price == pytest.approx(50.0 / expected_shares, rel=1e-4)
    assert q.slippage_bps == pytest.approx(
        (q.vwap_price - 0.60) / 0.60 * 10_000, rel=1e-3
    )


def test_compute_quote_buy_insufficient_depth() -> None:
    book = {
        "bids": [],
        "asks": [{"price": 0.65, "size": 30.0}],  # 30 * 0.65 = 19.5 USD available
    }
    q = compute_quote(book, "BUY", 50.0)
    assert q.complete_fill is False
    assert q.max_fillable_usd == pytest.approx(30.0 * 0.65, rel=1e-4)
    assert q.filled_shares == pytest.approx(30.0, rel=1e-4)


def test_compute_quote_buy_empty_book() -> None:
    book = {"bids": [], "asks": []}
    q = compute_quote(book, "BUY", 50.0)
    assert q.complete_fill is False
    assert q.filled_shares == pytest.approx(0.0)
    assert q.max_fillable_usd == pytest.approx(0.0)
    assert q.best_bid is None
    assert q.best_ask is None
    assert q.mid_price is None
    assert q.slippage_bps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SELL compute_quote
# ---------------------------------------------------------------------------


def test_compute_quote_sell_full_fill_single_level() -> None:
    book = {
        "bids": [{"price": 0.59, "size": 200.0}],
        "asks": [{"price": 0.61, "size": 200.0}],
    }
    q = compute_quote(book, "SELL", 50.0)
    assert q.complete_fill is True
    assert q.vwap_price == pytest.approx(0.59)
    assert q.filled_shares == pytest.approx(50.0 / 0.59, rel=1e-4)
    assert q.slippage_bps == pytest.approx(0.0, abs=0.01)


def test_compute_quote_sell_multi_level_slippage() -> None:
    book = {
        "bids": [
            {"price": 0.62, "size": 40.0},  # 40 * 0.62 = 24.8 USD
            {"price": 0.58, "size": 100.0},  # need 25.2 more -> 25.2/0.58 = 43.45 shares
        ],
        "asks": [],
    }
    q = compute_quote(book, "SELL", 50.0)
    assert q.complete_fill is True
    expected_shares = 40.0 + 25.2 / 0.58
    assert q.filled_shares == pytest.approx(expected_shares, rel=1e-4)
    assert q.vwap_price == pytest.approx(50.0 / expected_shares, rel=1e-4)
    # slippage: (best_bid - vwap) / best_bid * 10000
    assert q.slippage_bps == pytest.approx(
        (0.62 - q.vwap_price) / 0.62 * 10_000, rel=1e-3
    )


def test_compute_quote_sell_insufficient_depth() -> None:
    book = {
        "bids": [{"price": 0.58, "size": 20.0}],  # 20 * 0.58 = 11.6 USD
        "asks": [],
    }
    q = compute_quote(book, "SELL", 50.0)
    assert q.complete_fill is False
    assert q.max_fillable_usd == pytest.approx(20.0 * 0.58, rel=1e-4)
    assert q.filled_shares == pytest.approx(20.0, rel=1e-4)


def test_compute_quote_book_levels_unsorted_input() -> None:
    """compute_quote must sort book levels internally."""
    book = {
        "bids": [{"price": 0.55, "size": 50.0}, {"price": 0.59, "size": 200.0}],
        "asks": [{"price": 0.65, "size": 50.0}, {"price": 0.61, "size": 200.0}],
    }
    # best ask after sorting ascending is 0.61
    q = compute_quote(book, "BUY", 50.0)
    assert q.best_ask == pytest.approx(0.61)
    assert q.vwap_price == pytest.approx(0.61)
    # best bid after sorting descending is 0.59
    assert q.best_bid == pytest.approx(0.59)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compute_quote_invalid_side_raises() -> None:
    book = {"bids": [], "asks": [{"price": 0.60, "size": 100.0}]}
    with pytest.raises(ValueError, match="side"):
        compute_quote(book, "HOLD", 50.0)


def test_compute_quote_lowercase_side_accepted() -> None:
    book = {"bids": [], "asks": [{"price": 0.60, "size": 100.0}]}
    q = compute_quote(book, "buy", 50.0)
    assert q.complete_fill is True
    assert q.vwap_price == pytest.approx(0.60)


def test_compute_quote_zero_size_raises() -> None:
    book = {"bids": [], "asks": [{"price": 0.60, "size": 100.0}]}
    with pytest.raises(ValueError, match="size_usd"):
        compute_quote(book, "BUY", 0.0)


def test_compute_quote_negative_size_raises() -> None:
    book = {"bids": [], "asks": [{"price": 0.60, "size": 100.0}]}
    with pytest.raises(ValueError, match="size_usd"):
        compute_quote(book, "BUY", -10.0)


# ---------------------------------------------------------------------------
# mid_price requires both sides (ADR-0011: mark price = bid/ask midpoint)
# ---------------------------------------------------------------------------


def test_compute_quote_mid_price_none_when_asks_only() -> None:
    book = {"bids": [], "asks": [{"price": 0.62, "size": 100.0}]}
    q = compute_quote(book, "BUY", 50.0)
    assert q.mid_price is None
    assert q.best_ask == pytest.approx(0.62)
    assert q.best_bid is None


def test_compute_quote_mid_price_none_when_bids_only() -> None:
    book = {"bids": [{"price": 0.58, "size": 100.0}], "asks": []}
    q = compute_quote(book, "SELL", 50.0)
    assert q.mid_price is None
    assert q.best_bid == pytest.approx(0.58)
    assert q.best_ask is None


# ---------------------------------------------------------------------------
# Leaf module: must not import from other project packages
# ---------------------------------------------------------------------------


def test_quote_module_is_leaf() -> None:
    import ast
    import inspect
    import core.pricing.quote as mod

    src = inspect.getsource(mod)
    tree = ast.parse(src)
    project_packages = {
        "data", "model", "loops", "core.exchange", "core.strategy",
        "core.risk", "core.positions", "core.alerts", "core.bot",
        "core.probability_store", "core.settlement_resolver", "core.types",
        "events", "snipers", "db", "ui", "backtest",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            assert top not in project_packages, (
                f"core/pricing/quote.py imports from project package '{node.module}'"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in project_packages, (
                    f"core/pricing/quote.py imports from project package '{alias.name}'"
                )


# ---------------------------------------------------------------------------
# BookLevel / NormalizedBook TypedDict contract
# ---------------------------------------------------------------------------


def test_book_level_typeddict_usable_as_dict() -> None:
    level: BookLevel = {"price": 0.60, "size": 100.0}
    assert level["price"] == pytest.approx(0.60)
    assert level["size"] == pytest.approx(100.0)


def test_normalized_book_typeddict_accepted_by_compute_quote() -> None:
    book: NormalizedBook = {
        "bids": [{"price": 0.58, "size": 50.0}],
        "asks": [{"price": 0.62, "size": 50.0}],
    }
    q = compute_quote(book, "BUY", 20.0)
    assert q.complete_fill is True
    assert q.vwap_price == pytest.approx(0.62)
    assert q.mid_price == pytest.approx(0.60)

