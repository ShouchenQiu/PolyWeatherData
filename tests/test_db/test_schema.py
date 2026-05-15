import sqlite3
from datetime import datetime, timezone

import pytest

from db.schema import DbClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATION = ("KLGA", 40.7769, -73.874, "America/New_York")
EVENT = ("cid-001", "Will the high exceed 90F?", "will-the-high-lga", "2026-07-04", "KLGA")
MARKET = ("tok-001", "cid-001", "will-the-high-lga-yes", 88.0, 92.0, "Yes")


def _seed(db: DbClient) -> None:
    db.upsert_station(*STATION)
    db.upsert_event(*EVENT)
    db.upsert_market(*MARKET)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


def test_tables_created(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {row[0] for row in cur.fetchall()}
    expected = {"stations", "events", "markets", "signals", "orders", "model_versions"}
    assert expected <= names  # sqlite_sequence may also be present


# ---------------------------------------------------------------------------
# upsert_event -- idempotency
# ---------------------------------------------------------------------------


def test_upsert_event_insert(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event(*EVENT)
    rows = db._conn.execute("SELECT * FROM events").fetchall()
    assert len(rows) == 1
    assert rows[0]["condition_id"] == "cid-001"


def test_upsert_event_update_idempotent(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event(*EVENT)
    # Update status via second upsert
    db.upsert_event("cid-001", "Will the high exceed 90F?", "will-the-high-lga", "2026-07-04", "KLGA", "active")
    rows = db._conn.execute("SELECT status FROM events WHERE condition_id='cid-001'").fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "active"


# ---------------------------------------------------------------------------
# Foreign key enforcement
# ---------------------------------------------------------------------------


def test_market_fk_violation_raises(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    # No parent event inserted -- FK should fail
    with pytest.raises(sqlite3.IntegrityError):
        db.upsert_market("tok-999", "nonexistent-cid", "slug", 0.0, 1.0, "Yes")


# ---------------------------------------------------------------------------
# get_active_events filtering
# ---------------------------------------------------------------------------


def test_get_active_events_excludes_resolved(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event("cid-001", "Q1", "slug1", "2026-07-04", "KLGA", "active")
    db.upsert_event("cid-002", "Q2", "slug2", "2026-07-05", "KLGA", "resolved")
    db.upsert_event("cid-003", "Q3", "slug3", "2026-07-06", "KLGA", "cancelled")
    active = db.get_active_events()
    ids = [r["condition_id"] for r in active]
    assert "cid-001" in ids
    assert "cid-002" not in ids
    assert "cid-003" not in ids


# ---------------------------------------------------------------------------
# event_exists
# ---------------------------------------------------------------------------


def test_event_exists_returns_false_when_absent() -> None:
    db = DbClient(":memory:")
    assert db.event_exists("nonexistent-cid") is False


def test_event_exists_returns_true_after_upsert() -> None:
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event(*EVENT)
    assert db.event_exists("cid-001") is True


def test_event_exists_ignores_status() -> None:
    """event_exists returns True regardless of event status (resolved/cancelled)."""
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event("cid-resolved", "Q", "slug", "2026-01-01", "KLGA", "resolved")
    assert db.event_exists("cid-resolved") is True


# ---------------------------------------------------------------------------
# Signal and order round-trip
# ---------------------------------------------------------------------------


def test_signal_insert_returns_id(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    _seed(db)
    sid = db.insert_signal("2026-07-01T12:00:00Z", "tok-001", 0.72, 0.65, 0.07, "BUY")
    assert isinstance(sid, int)
    assert sid >= 1


# ---------------------------------------------------------------------------
# orders table -- new lifecycle schema (AC4 + AC5)
# ---------------------------------------------------------------------------


def test_orders_table_has_lifecycle_columns() -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute("PRAGMA table_info(orders)")
    cols = {row["name"] for row in cur.fetchall()}
    required = {
        "order_id", "token_id", "condition_id", "side",
        "size_usd", "price", "fill_price", "status",
        "created_at", "filled_at", "closed_at", "cancelled_at", "settled_payout",
    }
    assert required <= cols, f"Missing columns: {required - cols}"


def test_orders_status_default_pending() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db._conn.execute(
        "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price)"
        " VALUES ('ord-1', 'tok-001', 'cid-001', 'BUY', 10.0, 0.5)"
    )
    db._conn.commit()
    row = db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["status"] == "pending"


def test_orders_optional_columns_nullable() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db._conn.execute(
        "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price)"
        " VALUES ('ord-2', 'tok-001', 'cid-001', 'BUY', 10.0, 0.5)"
    )
    db._conn.commit()
    row = db._conn.execute(
        "SELECT fill_price, filled_at, closed_at, cancelled_at, settled_payout"
        " FROM orders WHERE order_id='ord-2'"
    ).fetchone()
    for col in ("fill_price", "filled_at", "closed_at", "cancelled_at", "settled_payout"):
        assert row[col] is None, f"Expected {col} to be NULL"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes(tmp_dir) -> None:
    db_path = tmp_dir / "test.db"
    with DbClient(db_path) as db:
        db.upsert_station(*STATION)
    # Connection should be closed; further use raises
    with pytest.raises(Exception):
        db._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# bot_state KV table (private helpers)
# ---------------------------------------------------------------------------


def test_set_and_get_bot_state() -> None:
    db = DbClient(":memory:")
    db._set_bot_state("data_stale", '{"station": "KLGA"}')
    assert db._get_bot_state("data_stale") == '{"station": "KLGA"}'


def test_get_bot_state_absent_returns_none() -> None:
    db = DbClient(":memory:")
    assert db._get_bot_state("nonexistent") is None


def test_set_bot_state_upsert_overwrites() -> None:
    db = DbClient(":memory:")
    db._set_bot_state("key", "first")
    db._set_bot_state("key", "second")
    assert db._get_bot_state("key") == "second"


def test_set_bot_state_none_clears() -> None:
    db = DbClient(":memory:")
    db._set_bot_state("key", "something")
    db._set_bot_state("key", None)
    assert db._get_bot_state("key") is None


# ---------------------------------------------------------------------------
# Typed staleness accessors
# ---------------------------------------------------------------------------


def test_set_and_get_data_stale() -> None:
    db = DbClient(":memory:")
    t = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
    db.set_data_stale("KLGA", t)
    result = db.get_data_stale()
    assert result is not None
    station, occurred_at = result
    assert station == "KLGA"
    assert occurred_at == t


def test_set_data_stale_none_station() -> None:
    db = DbClient(":memory:")
    t = datetime(2026, 5, 15, tzinfo=timezone.utc)
    db.set_data_stale(None, t)
    result = db.get_data_stale()
    assert result is not None
    station, occurred_at = result
    assert station is None
    assert occurred_at == t


def test_get_data_stale_absent_returns_none() -> None:
    db = DbClient(":memory:")
    assert db.get_data_stale() is None


def test_clear_data_stale() -> None:
    db = DbClient(":memory:")
    db.set_data_stale(None, datetime(2026, 5, 15, tzinfo=timezone.utc))
    db.clear_data_stale()
    assert db.get_data_stale() is None
