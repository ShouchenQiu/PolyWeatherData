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
    expected = {"stations", "events", "markets", "signals", "orders", "model_versions", "redeem_tasks", "bot_state"}
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
    """V2V01: get_active_events returns only active events with a parsed market."""
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event("cid-001", "Q1", "slug1", "2026-07-04", "KLGA", "active")
    db.upsert_event("cid-002", "Q2", "slug2", "2026-07-05", "KLGA", "resolved")
    db.upsert_event("cid-003", "Q3", "slug3", "2026-07-06", "KLGA", "cancelled")
    # Only cid-001 gets a parsed market -- cid-002/003 are filtered by status
    db.upsert_market("tok-001", "cid-001", "klga-high-60-70f", 60.0, 70.0, "YES")
    active = db.get_active_events()
    ids = [r["condition_id"] for r in active]
    assert "cid-001" in ids
    assert "cid-002" not in ids
    assert "cid-003" not in ids


def test_get_active_events_excludes_event_without_parsed_market(tmp_db: sqlite3.Connection) -> None:
    """V2V01 hard gate: active event with only unrecognized_slug market not returned."""
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event("cid-good", "Q", "slug", "2026-07-04", "KLGA", "active")
    db.upsert_event("cid-bad",  "Q", "slug", "2026-07-05", "KLGA", "active")
    db.upsert_market("tok-good", "cid-good", "klga-60-70f", 60.0, 70.0, "YES")
    db._conn.execute(
        "INSERT INTO markets (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome, discovery_status)"
        " VALUES ('tok-bad', 'cid-bad', 'unrecog', NULL, NULL, 'YES', 'unrecognized_slug')"
    )
    db._conn.commit()
    active = db.get_active_events()
    ids = [r["condition_id"] for r in active]
    assert "cid-good" in ids
    assert "cid-bad" not in ids


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


def test_orders_has_v2_accounting_columns() -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute("PRAGMA table_info(orders)")
    cols = {row["name"] for row in cur.fetchall()}
    v2_cols = {"is_paper", "executable_price", "filled_shares", "signed_shares", "cash_flow_usd"}
    assert v2_cols <= cols, f"Missing V2 accounting columns: {v2_cols - cols}"


def test_orders_is_paper_defaults_to_zero() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db._conn.execute(
        "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price)"
        " VALUES ('ord-paper', 'tok-001', 'cid-001', 'BUY', 10.0, 0.5)"
    )
    db._conn.commit()
    row = db._conn.execute(
        "SELECT is_paper FROM orders WHERE order_id='ord-paper'"
    ).fetchone()
    assert row["is_paper"] == 0


def test_orders_is_paper_check_constraint_rejects_invalid() -> None:
    import sqlite3 as _sqlite3
    db = DbClient(":memory:")
    _seed(db)
    with pytest.raises(_sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price, is_paper)"
            " VALUES ('ord-bad', 'tok-001', 'cid-001', 'BUY', 10.0, 0.5, 2)"
        )
        db._conn.commit()


# ---------------------------------------------------------------------------
# redeem_tasks table (V2F01)
# ---------------------------------------------------------------------------


def test_redeem_tasks_table_has_required_columns() -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute("PRAGMA table_info(redeem_tasks)")
    cols = {row["name"] for row in cur.fetchall()}
    required = {
        "condition_id", "status", "attempt_count",
        "last_error", "created_at", "updated_at", "redeemed_at",
    }
    assert required <= cols, f"Missing redeem_tasks columns: {required - cols}"


def test_redeem_tasks_status_check_constraint() -> None:
    import sqlite3 as _sqlite3
    db = DbClient(":memory:")
    now = "2025-01-01T00:00:00Z"
    valid_statuses = ["pending", "redeemed", "failed", "skipped_paper"]
    for status in valid_statuses:
        db._conn.execute(
            "INSERT OR REPLACE INTO redeem_tasks (condition_id, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (f"cid-{status}", status, now, now),
        )
    db._conn.commit()
    with pytest.raises(_sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO redeem_tasks (condition_id, status, created_at, updated_at)"
            " VALUES ('cid-bad', 'invalid_status', ?, ?)",
            (now, now),
        )
        db._conn.commit()


# ---------------------------------------------------------------------------
# markets.discovery_status (V2F01)
# ---------------------------------------------------------------------------


def test_markets_has_discovery_status_column() -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute("PRAGMA table_info(markets)")
    cols = {row["name"] for row in cur.fetchall()}
    assert "discovery_status" in cols, "markets table missing discovery_status column"


def test_markets_discovery_status_defaults_to_parsed() -> None:
    db = DbClient(":memory:")
    _seed(db)
    row = db._conn.execute(
        "SELECT discovery_status FROM markets WHERE token_id='tok-001'"
    ).fetchone()
    assert row["discovery_status"] == "parsed"


# ---------------------------------------------------------------------------
# signals V2 shape (V2F01)
# ---------------------------------------------------------------------------


def test_signals_has_v2_columns() -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute("PRAGMA table_info(signals)")
    cols = {row["name"] for row in cur.fetchall()}
    v2_cols = {
        "signal_id", "yes_token_id", "no_token_id", "direction",
        "mc_prob", "reference_price", "edge",
        "final_token_id", "final_side", "executable_price",
        "tick_utc", "approved", "reject_reason",
        "computed_size", "order_id",
    }
    assert v2_cols <= cols, f"Missing V2 signals columns: {v2_cols - cols}"


def test_signals_signal_id_is_text_primary_key() -> None:
    db = DbClient(":memory:")
    cur = db._conn.execute("PRAGMA table_info(signals)")
    rows = {row["name"]: row for row in cur.fetchall()}
    assert rows["signal_id"]["type"].upper() == "TEXT", "signal_id must be TEXT"
    assert rows["signal_id"]["pk"] == 1, "signal_id must be the primary key"


def test_migration_upgrades_v1_db_to_v2(tmp_dir) -> None:
    """Simulate an existing V1 database being opened by V2 DbClient."""
    import sqlite3 as _sqlite3
    db_path = tmp_dir / "v1.db"
    # Build a V1-style DB file without going through DbClient
    conn = _sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE stations (station_id TEXT PRIMARY KEY, lat REAL, lon REAL, timezone TEXT);
        CREATE TABLE events (condition_id TEXT PRIMARY KEY, question TEXT,
            event_slug TEXT, event_end_date TEXT, station_id TEXT, status TEXT DEFAULT 'discovered');
        CREATE TABLE markets (token_id TEXT PRIMARY KEY, condition_id TEXT,
            market_slug TEXT, bucket_lo_f REAL, bucket_hi_f REAL, outcome TEXT);
        CREATE TABLE signals (signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, token_id TEXT, mc_prob REAL, clob_price REAL, edge REAL, side TEXT);
        CREATE TABLE orders (order_id TEXT PRIMARY KEY, token_id TEXT, condition_id TEXT,
            side TEXT, size_usd REAL, price REAL, fill_price REAL, status TEXT DEFAULT 'pending',
            created_at TEXT, filled_at TEXT, closed_at TEXT, cancelled_at TEXT, settled_payout REAL);
        CREATE TABLE model_versions (version_id TEXT PRIMARY KEY, created_at TEXT,
            val_loss REAL, notes TEXT);
        CREATE TABLE bot_state (key TEXT PRIMARY KEY, value_text TEXT, updated_at TEXT NOT NULL);
        """
    )
    conn.commit()
    conn.close()
    # Open with V2 DbClient -- migration should run automatically
    db = DbClient(db_path)
    cur = db._conn.execute("PRAGMA table_info(orders)")
    cols = {row["name"] for row in cur.fetchall()}
    assert {"is_paper", "executable_price", "filled_shares", "signed_shares", "cash_flow_usd"} <= cols
    cur = db._conn.execute("PRAGMA table_info(signals)")
    cols = {row["name"] for row in cur.fetchall()}
    assert "tick_utc" in cols and "timestamp" not in cols
    cur = db._conn.execute("PRAGMA table_info(markets)")
    cols = {row["name"] for row in cur.fetchall()}
    assert "discovery_status" in cols
    # Verify migrated DB enforces CHECK constraints (not just column presence)
    db._conn.execute("PRAGMA foreign_keys=OFF")
    with pytest.raises(_sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO orders (order_id, is_paper) VALUES ('bad-ord', 2)"
        )
    db._conn.rollback()
    with pytest.raises(_sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO orders (order_id, status) VALUES ('bad-status', 'bogus')"
        )
    db._conn.rollback()
    with pytest.raises(_sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO markets (token_id, discovery_status) VALUES ('bad-tok', 'invalid_status')"
        )
    db._conn.rollback()
    db._conn.execute("PRAGMA foreign_keys=ON")


# ---------------------------------------------------------------------------
# Representative inserts -- paper/live orders, signal decision row,
# discovery_status variants (V2F01 acceptance criterion)
# ---------------------------------------------------------------------------


def test_orders_representative_live_and_paper_inserts() -> None:
    db = DbClient(":memory:")
    _seed(db)
    now = "2026-05-31T12:00:00Z"
    # Live order: is_paper=0, BUY fills with positive signed_shares and negative cash flow
    db._conn.execute(
        "INSERT INTO orders"
        " (order_id, token_id, condition_id, side, size_usd, price,"
        "  is_paper, executable_price, filled_shares, signed_shares, cash_flow_usd,"
        "  status, created_at)"
        " VALUES ('live-001', 'tok-001', 'cid-001', 'BUY', 50.0, 0.55,"
        "  0, 0.553, 90.0, 90.0, -49.77, 'filled', ?)",
        (now,),
    )
    # Paper order: is_paper=1, SELL fills with negative signed_shares, positive cash flow
    db._conn.execute(
        "INSERT INTO orders"
        " (order_id, token_id, condition_id, side, size_usd, price,"
        "  is_paper, executable_price, filled_shares, signed_shares, cash_flow_usd,"
        "  status, created_at)"
        " VALUES ('paper-001', 'tok-001', 'cid-001', 'SELL', 50.0, 0.62,"
        "  1, 0.618, 80.0, -80.0, 49.44, 'filled', ?)",
        (now,),
    )
    db._conn.commit()
    rows = {
        r["order_id"]: r
        for r in db._conn.execute("SELECT * FROM orders").fetchall()
    }
    assert rows["live-001"]["is_paper"] == 0
    assert rows["live-001"]["signed_shares"] == pytest.approx(90.0)
    assert rows["live-001"]["cash_flow_usd"] == pytest.approx(-49.77)
    assert rows["paper-001"]["is_paper"] == 1
    assert rows["paper-001"]["signed_shares"] == pytest.approx(-80.0)
    assert rows["paper-001"]["cash_flow_usd"] == pytest.approx(49.44)


def test_signals_representative_decision_row() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db.upsert_market("tok-001-no", "cid-001", "will-the-high-lga-no", 88.0, 92.0, "No")
    now = "2026-05-31T12:00:00Z"
    db._conn.execute(
        "INSERT INTO signals"
        " (signal_id, yes_token_id, no_token_id, direction,"
        "  mc_prob, reference_price, edge,"
        "  final_token_id, final_side, executable_price,"
        "  tick_utc, approved, computed_size)"
        " VALUES ('sig-001', 'tok-001', 'tok-001-no', 'YES_CHEAP',"
        "  0.72, 0.65, 0.07,"
        "  'tok-001', 'BUY', 0.652,"
        "  ?, 1, 45.0)",
        (now,),
    )
    db._conn.commit()
    row = db._conn.execute("SELECT * FROM signals WHERE signal_id='sig-001'").fetchone()
    assert row["direction"] == "YES_CHEAP"
    assert row["mc_prob"] == pytest.approx(0.72)
    assert row["approved"] == 1
    assert row["reject_reason"] is None
    assert row["order_id"] is None


def test_markets_discovery_status_valid_non_parsed_values() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db._conn.execute(
        "INSERT INTO markets"
        " (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome, discovery_status)"
        " VALUES ('tok-unrecog', 'cid-001', 'some-slug', NULL, NULL, NULL, 'unrecognized_slug')"
    )
    db._conn.execute(
        "INSERT INTO markets"
        " (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome, discovery_status)"
        " VALUES ('tok-invalid', 'cid-001', 'other-slug', NULL, NULL, NULL, 'invalid_market')"
    )
    db._conn.commit()
    statuses = {
        r["token_id"]: r["discovery_status"]
        for r in db._conn.execute(
            "SELECT token_id, discovery_status FROM markets WHERE token_id IN"
            " ('tok-unrecog', 'tok-invalid')"
        ).fetchall()
    }
    assert statuses["tok-unrecog"] == "unrecognized_slug"
    assert statuses["tok-invalid"] == "invalid_market"


def test_orders_status_check_constraint_rejects_invalid() -> None:
    import sqlite3 as _sqlite3
    db = DbClient(":memory:")
    _seed(db)
    with pytest.raises(_sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price, status)"
            " VALUES ('ord-bad-status', 'tok-001', 'cid-001', 'BUY', 10.0, 0.5, 'bogus')"
        )
        db._conn.commit()


def test_orders_status_check_allows_valid_values() -> None:
    db = DbClient(":memory:")
    _seed(db)
    for i, status in enumerate(("pending", "filled", "cancelled", "closed", "settled")):
        db._conn.execute(
            "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price, status)"
            " VALUES (?, 'tok-001', 'cid-001', 'BUY', 10.0, 0.5, ?)",
            (f"ord-{i}", status),
        )
    db._conn.commit()


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


# ---------------------------------------------------------------------------
# upsert_market with discovery_status (V2 domain method)
# ---------------------------------------------------------------------------


def test_upsert_market_default_discovery_status_parsed() -> None:
    db = DbClient(":memory:")
    _seed(db)
    row = db._conn.execute(
        "SELECT discovery_status FROM markets WHERE token_id='tok-001'"
    ).fetchone()
    assert row["discovery_status"] == "parsed"


def test_upsert_market_explicit_discovery_status() -> None:
    db = DbClient(":memory:")
    db.upsert_station(*STATION)
    db.upsert_event(*EVENT)
    db.upsert_market("tok-unk", "cid-001", "some-slug", None, None, "Yes",
                     discovery_status="unrecognized_slug")
    row = db._conn.execute(
        "SELECT discovery_status FROM markets WHERE token_id='tok-unk'"
    ).fetchone()
    assert row["discovery_status"] == "unrecognized_slug"


def test_upsert_market_updates_discovery_status_on_conflict() -> None:
    db = DbClient(":memory:")
    _seed(db)
    # Initially parsed; reclassified to invalid_market
    db.upsert_market("tok-001", "cid-001", "will-the-high-lga-yes", 88.0, 92.0, "Yes",
                     discovery_status="invalid_market")
    row = db._conn.execute(
        "SELECT discovery_status FROM markets WHERE token_id='tok-001'"
    ).fetchone()
    assert row["discovery_status"] == "invalid_market"


# ---------------------------------------------------------------------------
# insert_signal_decision (V2 domain method)
# ---------------------------------------------------------------------------


def test_insert_signal_decision_minimal() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db.upsert_market("tok-001-no", "cid-001", "will-the-high-lga-no", 88.0, 92.0, "No")
    now = "2026-05-31T12:00:00Z"
    db.insert_signal_decision(
        signal_id="sig-dm-001",
        yes_token_id="tok-001",
        no_token_id="tok-001-no",
        direction="YES_CHEAP",
        mc_prob=0.72,
        reference_price=0.65,
        edge=0.07,
        final_token_id="tok-001",
        final_side="BUY",
        executable_price=0.652,
        tick_utc=now,
        approved=1,
    )
    row = db._conn.execute(
        "SELECT * FROM signals WHERE signal_id='sig-dm-001'"
    ).fetchone()
    assert row["direction"] == "YES_CHEAP"
    assert row["mc_prob"] == pytest.approx(0.72)
    assert row["approved"] == 1
    assert row["reject_reason"] is None
    assert row["order_id"] is None


def test_insert_signal_decision_rejected_with_reason() -> None:
    db = DbClient(":memory:")
    _seed(db)
    db.upsert_market("tok-001-no", "cid-001", "will-the-high-lga-no", 88.0, 92.0, "No")
    now = "2026-05-31T12:00:00Z"
    db.insert_signal_decision(
        signal_id="sig-dm-002",
        yes_token_id="tok-001",
        no_token_id="tok-001-no",
        direction="YES_CHEAP",
        mc_prob=0.55,
        reference_price=0.51,
        edge=0.04,
        final_token_id="tok-001",
        final_side="BUY",
        executable_price=0.512,
        tick_utc=now,
        approved=0,
        reject_reason="below_min_edge",
        computed_size=45.0,
    )
    row = db._conn.execute(
        "SELECT * FROM signals WHERE signal_id='sig-dm-002'"
    ).fetchone()
    assert row["approved"] == 0
    assert row["reject_reason"] == "below_min_edge"
    assert row["computed_size"] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# enqueue_redeem_task / mark_redeem_failed / mark_redeemed
# ---------------------------------------------------------------------------


def test_enqueue_redeem_task_creates_row() -> None:
    db = DbClient(":memory:")
    db.enqueue_redeem_task("cid-001")
    row = db._conn.execute(
        "SELECT * FROM redeem_tasks WHERE condition_id='cid-001'"
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempt_count"] == 0
    assert row["last_error"] is None


def test_enqueue_redeem_task_reenqueue_does_not_increment_attempt() -> None:
    """Re-enqueuing a task (e.g. pending -> pending) must not bump attempt_count."""
    db = DbClient(":memory:")
    db.enqueue_redeem_task("cid-001")
    db.enqueue_redeem_task("cid-001", status="pending")
    row = db._conn.execute(
        "SELECT * FROM redeem_tasks WHERE condition_id='cid-001'"
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempt_count"] == 0  # still 0


def test_mark_redeem_failed_increments_attempt_and_sets_error() -> None:
    db = DbClient(":memory:")
    db.enqueue_redeem_task("cid-001")
    db.mark_redeem_failed("cid-001", "rpc timeout")
    row = db._conn.execute(
        "SELECT * FROM redeem_tasks WHERE condition_id='cid-001'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempt_count"] == 1
    assert row["last_error"] == "rpc timeout"


def test_enqueue_redeem_task_skipped_paper() -> None:
    db = DbClient(":memory:")
    db.enqueue_redeem_task("cid-paper", status="skipped_paper")
    row = db._conn.execute(
        "SELECT status FROM redeem_tasks WHERE condition_id='cid-paper'"
    ).fetchone()
    assert row["status"] == "skipped_paper"


def test_mark_redeemed_sets_status_and_timestamp() -> None:
    db = DbClient(":memory:")
    db.enqueue_redeem_task("cid-001")
    db.mark_redeemed("cid-001")
    row = db._conn.execute(
        "SELECT * FROM redeem_tasks WHERE condition_id='cid-001'"
    ).fetchone()
    assert row["status"] == "redeemed"
    assert row["redeemed_at"] is not None
