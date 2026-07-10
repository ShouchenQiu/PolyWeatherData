from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DbClient:
    """SQLite client for PolyWeatherBot state storage.

    Uses WAL journal mode for concurrent read performance.
    Pass ':memory:' as db_path for in-memory usage in tests.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).expanduser().parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._migrate_v2_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stations (
                station_id  TEXT PRIMARY KEY,
                lat         REAL,
                lon         REAL,
                timezone    TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                condition_id   TEXT PRIMARY KEY,
                question       TEXT,
                event_slug     TEXT,
                event_end_date TEXT,
                station_id     TEXT REFERENCES stations(station_id),
                status         TEXT DEFAULT 'discovered'
            );

            CREATE TABLE IF NOT EXISTS markets (
                token_id          TEXT PRIMARY KEY,
                condition_id      TEXT REFERENCES events(condition_id),
                market_id         TEXT,
                market_slug       TEXT,
                bucket_lo_f       REAL,
                bucket_hi_f       REAL,
                outcome           TEXT,
                fees_enabled      INTEGER NOT NULL DEFAULT 0,
                fee_rate          REAL,
                fee_exponent      REAL,
                fee_taker_only    INTEGER NOT NULL DEFAULT 1,
                discovery_status  TEXT NOT NULL DEFAULT 'parsed'
                                  CHECK (discovery_status IN ('parsed', 'unrecognized_slug', 'invalid_market'))
            );

            CREATE TABLE IF NOT EXISTS signals (
                signal_id        TEXT PRIMARY KEY,
                yes_token_id     TEXT REFERENCES markets(token_id),
                no_token_id      TEXT REFERENCES markets(token_id),
                direction        TEXT,
                mc_prob          REAL,
                reference_price  REAL,
                edge             REAL,
                final_token_id   TEXT REFERENCES markets(token_id),
                final_side       TEXT,
                executable_price REAL,
                tick_utc         TEXT,
                approved         INTEGER NOT NULL DEFAULT 0 CHECK (approved IN (0, 1)),
                reject_reason    TEXT,
                computed_size    REAL,
                order_id         TEXT REFERENCES orders(order_id)
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id         TEXT PRIMARY KEY,
                token_id         TEXT REFERENCES markets(token_id),
                condition_id     TEXT REFERENCES events(condition_id),
                side             TEXT,
                size_usd         REAL,
                price            REAL,
                fill_price       REAL,
                status           TEXT DEFAULT 'pending'
                                 CHECK (status IN ('pending', 'filled', 'cancelled', 'closed', 'settled')),
                created_at       TEXT,
                filled_at        TEXT,
                closed_at        TEXT,
                cancelled_at     TEXT,
                settled_payout   REAL,
                is_paper         INTEGER NOT NULL DEFAULT 0 CHECK (is_paper IN (0, 1)),
                executable_price REAL,
                filled_shares    REAL,
                signed_shares    REAL,
                cash_flow_usd    REAL,
                fee_usd         REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                version         TEXT PRIMARY KEY,
                checkpoint_path TEXT,
                train_mae       REAL,
                test_mae        REAL,
                created_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS station_max_bias (
                station_id  TEXT PRIMARY KEY,
                max_bias    REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bot_state (
                key        TEXT PRIMARY KEY,
                value_text TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS redeem_tasks (
                condition_id  TEXT PRIMARY KEY,
                market_id     TEXT,
                status        TEXT NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending', 'redeemed', 'failed', 'skipped_paper')),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error    TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                redeemed_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS loop_ticks (
                tick_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                loop_name         TEXT NOT NULL,
                tick_utc          TEXT NOT NULL,
                status            TEXT NOT NULL
                                  CHECK (status IN ('success', 'partial', 'failed', 'skipped')),
                duration_ms       REAL,
                events_processed  INTEGER NOT NULL DEFAULT 0,
                signals_processed INTEGER NOT NULL DEFAULT 0,
                orders_submitted  INTEGER NOT NULL DEFAULT 0,
                message           TEXT,
                error             TEXT
            );

            CREATE TABLE IF NOT EXISTS forecast_snapshots (
                snapshot_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id   TEXT NOT NULL REFERENCES events(condition_id),
                station_id     TEXT REFERENCES stations(station_id),
                init_time      TEXT NOT NULL,
                event_end_date TEXT,
                day_offset     INTEGER,
                tick_utc       TEXT NOT NULL,
                status         TEXT NOT NULL
                               CHECK (status IN ('success', 'failed', 'skipped')),
                p50_high_f     REAL,
                error          TEXT
            );

            CREATE TABLE IF NOT EXISTS forecast_bucket_probs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id   INTEGER NOT NULL REFERENCES forecast_snapshots(snapshot_id) ON DELETE CASCADE,
                token_id      TEXT REFERENCES markets(token_id),
                bucket_lo_f   REAL,
                bucket_hi_f   REAL,
                mc_prob       REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS forecast_quantiles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL REFERENCES forecast_snapshots(snapshot_id) ON DELETE CASCADE,
                fxx         INTEGER NOT NULL,
                p10         REAL NOT NULL,
                p50         REAL NOT NULL,
                p90         REAL NOT NULL,
                UNIQUE(snapshot_id, fxx)
            );

            CREATE TABLE IF NOT EXISTS market_price_snapshots (
                price_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id          TEXT NOT NULL REFERENCES markets(token_id),
                condition_id      TEXT REFERENCES events(condition_id),
                tick_utc          TEXT NOT NULL,
                best_bid          REAL,
                best_ask          REAL,
                mid_price         REAL,
                vwap_price        REAL,
                spread            REAL,
                depth_bid         REAL,
                depth_ask         REAL
            );

            CREATE TABLE IF NOT EXISTS pnl_snapshots (
                pnl_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick_utc        TEXT NOT NULL,
                is_paper        INTEGER CHECK (is_paper IN (0, 1)),
                realized_pnl    REAL NOT NULL,
                unrealized_pnl  REAL NOT NULL,
                total_pnl       REAL NOT NULL,
                gross_exposure  REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def _migrate_v2_schema(self) -> None:
        """Apply additive V2 schema changes to existing databases.

        Idempotent: safe to run on fresh V2 databases (all paths are no-ops).

        ALTER TABLE ADD COLUMN does not carry CHECK constraints on SQLite < 3.37,
        so after adding any missing columns we follow up with a full table recreation
        for orders and markets when the expected CHECK is absent from the DDL.
        For the signals table a full DROP + recreate is safe (observability data only).
        """
        # ------------------------------------------------------------------
        # Step 1: Add missing columns (idempotent; swallow duplicate-column errors)
        # ------------------------------------------------------------------
        _additive = [
            "ALTER TABLE orders ADD COLUMN is_paper         INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE orders ADD COLUMN executable_price REAL",
            "ALTER TABLE orders ADD COLUMN filled_shares    REAL",
            "ALTER TABLE orders ADD COLUMN signed_shares    REAL",
            "ALTER TABLE orders ADD COLUMN cash_flow_usd    REAL",
            "ALTER TABLE orders ADD COLUMN fee_usd REAL NOT NULL DEFAULT 0",
            "ALTER TABLE markets ADD COLUMN discovery_status TEXT NOT NULL DEFAULT 'parsed'",
            "ALTER TABLE markets ADD COLUMN market_id TEXT",
            "ALTER TABLE markets ADD COLUMN fees_enabled INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE markets ADD COLUMN fee_rate REAL",
            "ALTER TABLE markets ADD COLUMN fee_exponent REAL",
            "ALTER TABLE markets ADD COLUMN fee_taker_only INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE redeem_tasks ADD COLUMN market_id TEXT",
        ]
        for stmt in _additive:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise

        # ------------------------------------------------------------------
        # Step 2: Enforce CHECK constraints via full table recreation when absent.
        # Detection: inspect sqlite_master for the CHECK clause text.
        # ------------------------------------------------------------------
        def _ddl(table: str) -> str:
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            return row["sql"] if row else ""

        _orders_ddl = _ddl("orders")
        if (
            "CHECK (is_paper IN (0, 1))" not in _orders_ddl
            or "CHECK (status IN (" not in _orders_ddl
        ):
            self._conn.executescript(
                """
                CREATE TABLE orders_v2 (
                    order_id         TEXT PRIMARY KEY,
                    token_id         TEXT REFERENCES markets(token_id),
                    condition_id     TEXT REFERENCES events(condition_id),
                    side             TEXT,
                    size_usd         REAL,
                    price            REAL,
                    fill_price       REAL,
                    status           TEXT DEFAULT 'pending'
                                     CHECK (status IN ('pending', 'filled', 'cancelled', 'closed', 'settled')),
                    created_at       TEXT,
                    filled_at        TEXT,
                    closed_at        TEXT,
                    cancelled_at     TEXT,
                    settled_payout   REAL,
                    is_paper         INTEGER NOT NULL DEFAULT 0 CHECK (is_paper IN (0, 1)),
                    executable_price REAL,
                    filled_shares    REAL,
                    signed_shares    REAL,
                    cash_flow_usd    REAL,
                    fee_usd         REAL NOT NULL DEFAULT 0
                );
                INSERT INTO orders_v2 SELECT
                    order_id, token_id, condition_id, side, size_usd, price, fill_price,
                    status, created_at, filled_at, closed_at, cancelled_at, settled_payout,
                    COALESCE(is_paper, 0), executable_price, filled_shares,
                    signed_shares, cash_flow_usd, COALESCE(fee_usd, 0)
                FROM orders;
                DROP TABLE orders;
                ALTER TABLE orders_v2 RENAME TO orders;
                """
            )

        if "CHECK (discovery_status IN (" not in _ddl("markets"):
            self._conn.executescript(
                """
                CREATE TABLE markets_v2 (
                    token_id          TEXT PRIMARY KEY,
                    condition_id      TEXT REFERENCES events(condition_id),
                    market_id         TEXT,
                    market_slug       TEXT,
                    bucket_lo_f       REAL,
                    bucket_hi_f       REAL,
                    outcome           TEXT,
                    fees_enabled      INTEGER NOT NULL DEFAULT 0,
                    fee_rate          REAL,
                    fee_exponent      REAL,
                    fee_taker_only    INTEGER NOT NULL DEFAULT 1,
                    discovery_status  TEXT NOT NULL DEFAULT 'parsed'
                                      CHECK (discovery_status IN ('parsed', 'unrecognized_slug', 'invalid_market'))
                );
                INSERT INTO markets_v2 SELECT
                    token_id, condition_id, market_id, market_slug, bucket_lo_f, bucket_hi_f, outcome,
                    COALESCE(fees_enabled, 0), fee_rate, fee_exponent,
                    COALESCE(fee_taker_only, 1),
                    COALESCE(discovery_status, 'parsed')
                FROM markets;
                DROP TABLE markets;
                ALTER TABLE markets_v2 RENAME TO markets;
                """
            )

        # ------------------------------------------------------------------
        # Step 3: Replace V1 signals (integer PK / old columns) with V2 shape.
        # ------------------------------------------------------------------
        cur = self._conn.execute("PRAGMA table_info(signals)")
        if "tick_utc" not in {row["name"] for row in cur.fetchall()}:
            self._conn.executescript(
                """
                DROP TABLE IF EXISTS signals;
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id        TEXT PRIMARY KEY,
                    yes_token_id     TEXT REFERENCES markets(token_id),
                    no_token_id      TEXT REFERENCES markets(token_id),
                    direction        TEXT,
                    mc_prob          REAL,
                    reference_price  REAL,
                    edge             REAL,
                    final_token_id   TEXT REFERENCES markets(token_id),
                    final_side       TEXT,
                    executable_price REAL,
                    tick_utc         TEXT,
                    approved         INTEGER NOT NULL DEFAULT 0 CHECK (approved IN (0, 1)),
                    reject_reason    TEXT,
                    computed_size    REAL,
                    order_id         TEXT REFERENCES orders(order_id)
                );
                """
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------

    def upsert_station(
        self,
        station_id: str,
        lat: float,
        lon: float,
        timezone: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO stations (station_id, lat, lon, timezone)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(station_id) DO UPDATE SET
                lat=excluded.lat, lon=excluded.lon, timezone=excluded.timezone
            """,
            (station_id, lat, lon, timezone),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def upsert_event(
        self,
        condition_id: str,
        question: str,
        event_slug: str,
        event_end_date: str,
        station_id: str,
        status: str = "discovered",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO events
                (condition_id, question, event_slug, event_end_date, station_id, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
                question=excluded.question,
                event_slug=excluded.event_slug,
                event_end_date=excluded.event_end_date,
                station_id=excluded.station_id,
                status=excluded.status
            """,
            (condition_id, question, event_slug, event_end_date, station_id, status),
        )
        self._conn.commit()

    def get_active_events(self) -> list[dict]:
        """Return events in 'active' status that have at least one parsed market.

        Applies the V2V01 hard gate: only events with discovery_status='parsed'
        AND non-null bucket_lo_f / bucket_hi_f are returned.  This prevents
        callers from receiving events whose markets have never been successfully
        parsed.
        """
        cur = self._conn.execute(
            """
            SELECT DISTINCT e.* FROM events e
            JOIN markets m ON m.condition_id = e.condition_id
            WHERE e.status = 'active'
              AND m.discovery_status = 'parsed'
              AND m.bucket_lo_f IS NOT NULL
              AND m.bucket_hi_f IS NOT NULL
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def get_forecast_candidate_events(self, now_date: str, horizon_days: int = 2) -> list[dict]:
        """Return active events whose end date is within [today, today+horizon_days].

        Only 'active' events are returned; 'discovered' events are excluded
        because they have not yet been promoted by DiscoveryLoop and their
        markets may be incomplete.  The date window is the only other filter
        -- callers do not need to re-implement day_offset arithmetic.

        Parameters
        ----------
        now_date:
            ISO-8601 date string for today (e.g. ``"2026-07-04"``).
        horizon_days:
            Maximum number of calendar days ahead (inclusive). Default 2.
        """
        cur = self._conn.execute(
            """
            SELECT * FROM events
            WHERE status = 'active'
              AND event_end_date >= ?
              AND event_end_date <= date(?, '+' || ? || ' days')
            """,
            (now_date, now_date, horizon_days),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_tradeable_events(self, now_date: str) -> list[dict]:
        """Return events that are in 'active' status and within the 0..2-day window.

        Only events that have at least one market with discovery_status='parsed'
        AND non-null bucket_lo_f / bucket_hi_f are returned (V2V01 hard gate).
        """
        cur = self._conn.execute(
            """
            SELECT DISTINCT e.* FROM events e
            JOIN markets m ON m.condition_id = e.condition_id
            WHERE e.status = 'active'
              AND e.event_end_date >= ?
              AND e.event_end_date <= date(?, '+2 days')
              AND m.discovery_status = 'parsed'
              AND m.bucket_lo_f IS NOT NULL
              AND m.bucket_hi_f IS NOT NULL
            """,
            (now_date, now_date),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_pending_settlement_events(self) -> list[dict]:
        """Return events awaiting settlement (status = 'pending_settle')."""
        cur = self._conn.execute(
            "SELECT * FROM events WHERE status = 'pending_settle'"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_non_terminal_events(self) -> list[dict]:
        """Return all events that have not reached a terminal state.

        Terminal states are ``settled`` and ``cancelled``.  Used by
        DiscoveryLoop to drive the lifecycle advance pass.
        """
        cur = self._conn.execute(
            "SELECT * FROM events WHERE status NOT IN ('settled', 'cancelled')"
        )
        return [dict(r) for r in cur.fetchall()]

    def event_exists(self, condition_id: str) -> bool:
        """Return True if the event is already persisted, regardless of status."""
        cur = self._conn.execute(
            "SELECT 1 FROM events WHERE condition_id=? LIMIT 1",
            (condition_id,),
        )
        return cur.fetchone() is not None

    def update_event_status(self, condition_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE events SET status=? WHERE condition_id=?",
            (status, condition_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def upsert_market(
        self,
        token_id: str,
        condition_id: str,
        market_slug: str,
        bucket_lo_f: float | None,
        bucket_hi_f: float | None,
        outcome: str,
        discovery_status: str = "parsed",
        market_id: str | None = None,
        fees_enabled: bool = False,
        fee_rate: float | None = None,
        fee_exponent: float | None = None,
        fee_taker_only: bool = True,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO markets
                (token_id, condition_id, market_id, market_slug, bucket_lo_f,
                 bucket_hi_f, outcome, fees_enabled, fee_rate, fee_exponent,
                 fee_taker_only, discovery_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(token_id) DO UPDATE SET
                condition_id=excluded.condition_id,
                market_id=excluded.market_id,
                market_slug=excluded.market_slug,
                bucket_lo_f=excluded.bucket_lo_f,
                bucket_hi_f=excluded.bucket_hi_f,
                outcome=excluded.outcome,
                fees_enabled=excluded.fees_enabled,
                fee_rate=excluded.fee_rate,
                fee_exponent=excluded.fee_exponent,
                fee_taker_only=excluded.fee_taker_only,
                discovery_status=excluded.discovery_status
            """,
            (
                token_id, condition_id, market_id, market_slug, bucket_lo_f,
                bucket_hi_f, outcome, int(fees_enabled), fee_rate,
                fee_exponent, int(fee_taker_only), discovery_status,
            ),
        )
        self._conn.commit()

    def get_markets_for_event(self, condition_id: str) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM markets
            WHERE condition_id=?
              AND discovery_status='parsed'
              AND bucket_lo_f IS NOT NULL
              AND bucket_hi_f IS NOT NULL
            """,
            (condition_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def market_exists(self, token_id: str) -> bool:
        """Return True if this token_id is already persisted in markets."""
        cur = self._conn.execute(
            "SELECT 1 FROM markets WHERE token_id=? LIMIT 1",
            (token_id,),
        )
        return cur.fetchone() is not None

    def get_orders_for_settlement(self, condition_id: str) -> list[dict]:
        """Return all filled or closed orders for *condition_id*.

        Used by SettlementLoop.confirm_settlement to determine payouts.
        """
        cur = self._conn.execute(
            "SELECT * FROM orders"
            " WHERE condition_id=? AND status IN ('filled', 'closed')",
            (condition_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal_decision(
        self,
        signal_id: str,
        yes_token_id: str,
        no_token_id: str,
        direction: str,
        mc_prob: float,
        reference_price: float,
        edge: float,
        final_token_id: str,
        final_side: str,
        executable_price: float,
        tick_utc: str,
        approved: int,
        reject_reason: str | None = None,
        computed_size: float | None = None,
        order_id: str | None = None,
    ) -> None:
        """Insert one signal decision row into the signals table.

        Parameters
        ----------
        signal_id:
            UUID string; must be unique.
        approved:
            1 if approved for execution, 0 if rejected by risk/filter.
        reject_reason:
            Human-readable reason for rejection; None when approved=1.
        order_id:
            Set after order submission; None at signal creation time.
        """
        self._conn.execute(
            """
            INSERT INTO signals
                (signal_id, yes_token_id, no_token_id, direction,
                 mc_prob, reference_price, edge,
                 final_token_id, final_side, executable_price,
                 tick_utc, approved, reject_reason, computed_size, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id, yes_token_id, no_token_id, direction,
                mc_prob, reference_price, edge,
                final_token_id, final_side, executable_price,
                tick_utc, approved, reject_reason, computed_size, order_id,
            ),
        )
    def record_signal(
        self,
        signal_id:       str,
        yes_token_id:    str,
        no_token_id:     str,
        direction:       str,
        mc_prob:         float,
        reference_price: float,
        edge:            float,
        tick_utc:        str,
    ) -> None:
        """Insert a pending signal row before arbitration begins.

        All decision fields (final_token_id, final_side, executable_price,
        approved, reject_reason, computed_size, order_id) default to NULL / 0
        and are filled by subsequent update calls.

        no_token_id is stored as NULL when empty string (no paired NO market).
        """
        self._conn.execute(
            """
            INSERT INTO signals
                (signal_id, yes_token_id, no_token_id, direction,
                 mc_prob, reference_price, edge, tick_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                yes_token_id,
                no_token_id or None,
                direction,
                mc_prob,
                reference_price,
                edge,
                tick_utc,
            ),
        )
        self._conn.commit()

    def update_signal_decision(
        self,
        signal_id:        str,
        final_token_id:   str,
        final_side:       str,
        executable_price: float,
        approved:         int,
        reject_reason:    str | None,
        computed_size:    float | None,
    ) -> None:
        """Update signal row with arbitration decision and risk outcome.

        Called after directional arbitration and risk pipeline evaluation.
        approved=1 for approved signals, 0 for rejected.
        """
        self._conn.execute(
            """
            UPDATE signals
               SET final_token_id   = ?,
                   final_side       = ?,
                   executable_price = ?,
                   approved         = ?,
                   reject_reason    = ?,
                   computed_size    = ?
             WHERE signal_id = ?
            """,
            (
                final_token_id,
                final_side,
                executable_price,
                approved,
                reject_reason,
                computed_size,
                signal_id,
            ),
        )
        self._conn.commit()

    def update_signal_order_id(
        self,
        signal_id: str,
        order_id:  str,
    ) -> None:
        """Link a placed order_id to its originating signal row."""
        self._conn.execute(
            "UPDATE signals SET order_id = ? WHERE signal_id = ?",
            (order_id, signal_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Redeem tasks
    # ------------------------------------------------------------------

    def enqueue_redeem_task(
        self,
        condition_id: str,
        status: str = "pending",
        market_id: str | None = None,
    ) -> None:
        """Create a redeem_task row; on conflict only update status.

        attempt_count is NEVER incremented by this method -- it is only
        bumped in ``mark_redeem_failed``.
        """
        now = _utcnow_iso()
        self._conn.execute(
            """
            INSERT INTO redeem_tasks
                (condition_id, market_id, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
                market_id=COALESCE(excluded.market_id, redeem_tasks.market_id),
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (condition_id, market_id, status, now, now),
        )
        self._conn.commit()

    def mark_redeem_failed(
        self,
        condition_id: str,
        last_error: str,
    ) -> None:
        """Set redeem_task status to 'failed' and increment attempt_count."""
        now = _utcnow_iso()
        self._conn.execute(
            """
            UPDATE redeem_tasks
            SET status='failed',
                attempt_count=attempt_count + 1,
                last_error=?,
                updated_at=?
            WHERE condition_id=?
            """,
            (last_error[:500], now, condition_id),
        )
        self._conn.commit()

    def mark_redeemed(self, condition_id: str) -> None:
        """Set redeem_task status to 'redeemed' and record redeemed_at timestamp."""
        now = _utcnow_iso()
        self._conn.execute(
            """
            UPDATE redeem_tasks
            SET status='redeemed', redeemed_at=?, updated_at=?
            WHERE condition_id=?
            """,
            (now, now, condition_id),
        )
        self._conn.commit()

    def get_pending_redeem_tasks(self) -> list[dict]:
        """Return all redeem_tasks with status 'pending' or 'failed'."""
        cur = self._conn.execute(
            """
            SELECT * FROM redeem_tasks
            WHERE status IN ('pending', 'failed')
            ORDER BY created_at ASC
            """,
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def record_loop_tick(
        self,
        loop_name: str,
        tick_utc: str,
        status: str,
        duration_ms: float | None = None,
        events_processed: int = 0,
        signals_processed: int = 0,
        orders_submitted: int = 0,
        message: str | None = None,
        error: str | None = None,
    ) -> int:
        """Persist one loop execution summary for UI observability."""
        cur = self._conn.execute(
            """
            INSERT INTO loop_ticks
                (loop_name, tick_utc, status, duration_ms,
                 events_processed, signals_processed, orders_submitted,
                 message, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                loop_name, tick_utc, status, duration_ms,
                events_processed, signals_processed, orders_submitted,
                message, error,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_forecast_snapshot(
        self,
        condition_id: str,
        station_id: str | None,
        init_time: str,
        event_end_date: str | None,
        day_offset: int | None,
        tick_utc: str,
        status: str,
        p50_high_f: float | None = None,
        error: str | None = None,
    ) -> int:
        """Persist one ForecastLoop model-output snapshot."""
        cur = self._conn.execute(
            """
            INSERT INTO forecast_snapshots
                (condition_id, station_id, init_time, event_end_date,
                 day_offset, tick_utc, status, p50_high_f, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                condition_id, station_id, init_time, event_end_date,
                day_offset, tick_utc, status, p50_high_f, error,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_forecast_bucket_prob(
        self,
        snapshot_id: int,
        token_id: str | None,
        bucket_lo_f: float | None,
        bucket_hi_f: float | None,
        mc_prob: float,
    ) -> int:
        """Persist one bucket probability row for a forecast snapshot."""
        cur = self._conn.execute(
            """
            INSERT INTO forecast_bucket_probs
                (snapshot_id, token_id, bucket_lo_f, bucket_hi_f, mc_prob)
            VALUES (?, ?, ?, ?, ?)
            """,
            (snapshot_id, token_id, bucket_lo_f, bucket_hi_f, mc_prob),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_forecast_quantile(
        self,
        snapshot_id: int,
        fxx: int,
        p10: float,
        p50: float,
        p90: float,
    ) -> int:
        """Persist one fxx quantile row for a forecast snapshot."""
        cur = self._conn.execute(
            """
            INSERT INTO forecast_quantiles
                (snapshot_id, fxx, p10, p50, p90)
            VALUES (?, ?, ?, ?, ?)
            """,
            (snapshot_id, fxx, p10, p50, p90),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_market_price_snapshot(
        self,
        token_id: str,
        condition_id: str | None,
        tick_utc: str,
        best_bid: float | None = None,
        best_ask: float | None = None,
        mid_price: float | None = None,
        vwap_price: float | None = None,
        spread: float | None = None,
        depth_bid: float | None = None,
        depth_ask: float | None = None,
    ) -> int:
        """Persist one CLOB price/depth observation."""
        cur = self._conn.execute(
            """
            INSERT INTO market_price_snapshots
                (token_id, condition_id, tick_utc, best_bid, best_ask,
                 mid_price, vwap_price, spread, depth_bid, depth_ask)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_id, condition_id, tick_utc, best_bid, best_ask,
                mid_price, vwap_price, spread, depth_bid, depth_ask,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_pnl_snapshot(
        self,
        tick_utc: str,
        is_paper: bool | None,
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
        gross_exposure: float,
    ) -> int:
        """Persist one PnL/exposure snapshot for UI time-series charts."""
        cur = self._conn.execute(
            """
            INSERT INTO pnl_snapshots
                (tick_utc, is_paper, realized_pnl, unrealized_pnl,
                 total_pnl, gross_exposure)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tick_utc,
                None if is_paper is None else int(is_paper),
                realized_pnl,
                unrealized_pnl,
                total_pnl,
                gross_exposure,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # Model versions
    # ------------------------------------------------------------------

    def register_model_version(
        self,
        version: str,
        checkpoint_path: str,
        train_mae: float,
        test_mae: float,
        created_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO model_versions
                (version, checkpoint_path, train_mae, test_mae, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (version, checkpoint_path, train_mae, test_mae, created_at),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Low-level escape hatches -- prefer domain methods above
    # ------------------------------------------------------------------
    # Use execute() / query() only when no domain method exists yet.
    # New modules needing DB access should get their own domain methods
    # here rather than accumulating raw SQL across the codebase.
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Low-level escape hatch: execute a write statement and commit.

        Prefer adding a domain-level method to DbClient over calling this
        directly.  Only use when no suitable domain method exists.
        Callers must not access ``_conn`` directly.
        """
        self._conn.execute(sql, params)
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Low-level escape hatch: execute a read query and return all rows.

        Prefer adding a domain-level method to DbClient over calling this
        directly.  Only use when no suitable domain method exists.
        Callers must not access ``_conn`` directly.
        """
        return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bot state (cross-process key-value for UI/bot communication)
    # ------------------------------------------------------------------
    # Typed accessors below are the public interface.  The two private
    # helpers (_set_bot_state / _get_bot_state) are implementation detail
    # and should not be called outside this class.
    # ------------------------------------------------------------------

    def _set_bot_state(self, key: str, value_text: str | None) -> None:
        """Upsert *key* in bot_state.  Pass None to clear the entry."""
        if value_text is None:
            self._conn.execute("DELETE FROM bot_state WHERE key=?", (key,))
        else:
            self._conn.execute(
                """
                INSERT INTO bot_state (key, value_text, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_text=excluded.value_text,
                    updated_at=excluded.updated_at
                """,
                (key, value_text, _utcnow_iso()),
            )
        self._conn.commit()

    def _get_bot_state(self, key: str) -> str | None:
        """Return the stored value for *key*, or None if absent."""
        cur = self._conn.execute(
            "SELECT value_text FROM bot_state WHERE key=?",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    # -- Typed staleness interface --
    # JSON schema for the 'data_stale' key:
    #   {"station": str | null, "occurred_at": ISO-8601 str}
    # Both writer (TradingLoop) and reader (UI) go through these methods;
    # neither side touches the raw JSON.

    def set_data_stale(self, station: str | None, occurred_at: datetime) -> None:
        """Record a data_stale event from TradingLoop."""
        self._set_bot_state(
            "data_stale",
            json.dumps({"station": station, "occurred_at": occurred_at.isoformat()}),
        )

    def clear_data_stale(self) -> None:
        """Remove the data_stale entry (called on recovery, if needed)."""
        self._set_bot_state("data_stale", None)

    def get_data_stale(self) -> tuple[str | None, datetime] | None:
        """Return (station, occurred_at) if a stale alert is recorded, else None."""
        raw = self._get_bot_state("data_stale")
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            occurred_at = datetime.fromisoformat(data["occurred_at"])
            return data.get("station"), occurred_at
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DbClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
