import sqlite3

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
# Signal and order round-trip
# ---------------------------------------------------------------------------


def test_signal_insert_returns_id(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    _seed(db)
    sid = db.insert_signal("2026-07-01T12:00:00Z", "tok-001", 0.72, 0.65, 0.07, "BUY")
    assert isinstance(sid, int)
    assert sid >= 1


def test_order_status_update(tmp_db: sqlite3.Connection) -> None:
    db = DbClient(":memory:")
    _seed(db)
    sid = db.insert_signal("2026-07-01T12:00:00Z", "tok-001", 0.72, 0.65, 0.07, "BUY")
    db.insert_order("ord-001", sid, 25.0, 0.65)
    open_orders = db.get_open_orders()
    assert len(open_orders) == 1

    db.update_order_status("ord-001", "filled", "2026-07-01T12:05:00Z")
    open_orders_after = db.get_open_orders()
    assert len(open_orders_after) == 0


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
