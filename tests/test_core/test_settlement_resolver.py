"""Tests for core/settlement_resolver.py."""
from __future__ import annotations

from db.schema import DbClient
from core.settlement_resolver import OrderSettlement, SettlementResolver


def _make_db_with_market(
    condition_id: str,
    token_id: str,
    lo: float,
    hi: float,
    outcome: str,
    order_id: str,
    size_usd: float,
) -> DbClient:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.77, -73.87, "America/New_York")
    db.upsert_event(
        condition_id, "Max Temp", "slug", "2026-07-04", "KLGA", "pending_settle"
    )
    db.upsert_market(token_id, condition_id, "market-slug", lo, hi, outcome)
    db._conn.execute(
        """
        INSERT INTO orders
            (order_id, token_id, condition_id, side, size_usd, price, status)
        VALUES (?, ?, ?, 'BUY', ?, 0.50, 'filled')
        """,
        (order_id, token_id, condition_id, size_usd),
    )
    db._conn.commit()
    return db


# ---------------------------------------------------------------------------
# Single-order YES/NO payout rules
# ---------------------------------------------------------------------------


def test_yes_token_in_bucket_pays_full() -> None:
    db = _make_db_with_market("c1", "tok1", 90.0, 95.0, "YES", "ord1", 25.0)
    resolver = SettlementResolver(db)
    results = resolver.resolve_from_obs("c1", 92.0)
    assert results == [OrderSettlement(order_id="ord1", payout=25.0)]


def test_yes_token_outside_bucket_pays_zero() -> None:
    db = _make_db_with_market("c1", "tok1", 90.0, 95.0, "YES", "ord1", 25.0)
    resolver = SettlementResolver(db)
    results = resolver.resolve_from_obs("c1", 88.0)
    assert results == [OrderSettlement(order_id="ord1", payout=0.0)]


def test_no_token_outside_bucket_pays_full() -> None:
    db = _make_db_with_market("c1", "tok1", 90.0, 95.0, "NO", "ord1", 10.0)
    resolver = SettlementResolver(db)
    results = resolver.resolve_from_obs("c1", 80.0)
    assert results == [OrderSettlement(order_id="ord1", payout=10.0)]


def test_no_token_in_bucket_pays_zero() -> None:
    db = _make_db_with_market("c1", "tok1", 90.0, 95.0, "NO", "ord1", 10.0)
    resolver = SettlementResolver(db)
    results = resolver.resolve_from_obs("c1", 91.0)
    assert results == [OrderSettlement(order_id="ord1", payout=0.0)]


def test_bucket_boundary_hi_is_exclusive() -> None:
    db = _make_db_with_market("c1", "tok1", 90.0, 95.0, "YES", "ord1", 5.0)
    resolver = SettlementResolver(db)
    # exactly at hi -> outside bucket
    results = resolver.resolve_from_obs("c1", 95.0)
    assert results == [OrderSettlement(order_id="ord1", payout=0.0)]


def test_bucket_boundary_lo_is_inclusive() -> None:
    db = _make_db_with_market("c1", "tok1", 90.0, 95.0, "YES", "ord1", 5.0)
    resolver = SettlementResolver(db)
    results = resolver.resolve_from_obs("c1", 90.0)
    assert results == [OrderSettlement(order_id="ord1", payout=5.0)]


# ---------------------------------------------------------------------------
# Missing market entry
# ---------------------------------------------------------------------------


def test_order_with_no_market_entry_pays_zero() -> None:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.77, -73.87, "America/New_York")
    db.upsert_event("c1", "Q", "s", "2026-07-04", "KLGA", "pending_settle")
    # Market row exists but has no bucket range -> resolver excludes it from token_market
    db.upsert_market("tok-orphan", "c1", "mkt-slug", None, None, "YES")
    db._conn.execute(
        """
        INSERT INTO orders
            (order_id, token_id, condition_id, side, size_usd, price, status)
        VALUES ('ord1', 'tok-orphan', 'c1', 'BUY', 20.0, 0.50, 'filled')
        """
    )
    db._conn.commit()
    resolver = SettlementResolver(db)
    results = resolver.resolve_from_obs("c1", 92.0)
    assert results == [OrderSettlement(order_id="ord1", payout=0.0)]


# ---------------------------------------------------------------------------
# Multiple orders for the same condition
# ---------------------------------------------------------------------------


def test_multiple_orders_resolved_independently() -> None:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.77, -73.87, "America/New_York")
    db.upsert_event("c1", "Q", "s", "2026-07-04", "KLGA", "pending_settle")
    db.upsert_market("tok-yes", "c1", "mkt-slug", 90.0, 95.0, "YES")
    db.upsert_market("tok-no", "c1", "mkt-slug2", 90.0, 95.0, "NO")
    db._conn.executemany(
        """
        INSERT INTO orders
            (order_id, token_id, condition_id, side, size_usd, price, status)
        VALUES (?, ?, 'c1', 'BUY', ?, 0.50, 'filled')
        """,
        [("ord-yes", "tok-yes", 30.0), ("ord-no", "tok-no", 20.0)],
    )
    db._conn.commit()

    resolver = SettlementResolver(db)
    results = {s.order_id: s.payout for s in resolver.resolve_from_obs("c1", 92.0)}
    # max_temp=92.0 is in [90, 95): YES wins, NO loses
    assert results["ord-yes"] == 30.0
    assert results["ord-no"] == 0.0


def test_no_orders_returns_empty_list() -> None:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.77, -73.87, "America/New_York")
    db.upsert_event("c1", "Q", "s", "2026-07-04", "KLGA", "pending_settle")
    resolver = SettlementResolver(db)
    assert resolver.resolve_from_obs("c1", 92.0) == []
