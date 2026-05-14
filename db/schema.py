from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class DbClient:
    """SQLite client for PolyWeatherBot state storage.

    Uses WAL journal mode for concurrent read performance.
    Pass ':memory:' as db_path for in-memory usage in tests.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

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
                token_id      TEXT PRIMARY KEY,
                condition_id  TEXT REFERENCES events(condition_id),
                market_slug   TEXT,
                bucket_lo_f   REAL,
                bucket_hi_f   REAL,
                outcome       TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                signal_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                token_id   TEXT REFERENCES markets(token_id),
                mc_prob    REAL,
                clob_price REAL,
                edge       REAL,
                side       TEXT
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id       TEXT PRIMARY KEY,
                token_id       TEXT REFERENCES markets(token_id),
                condition_id   TEXT REFERENCES events(condition_id),
                side           TEXT,
                size_usd       REAL,
                price          REAL,
                fill_price     REAL,
                status         TEXT DEFAULT 'pending',
                -- status domain: pending | filled | cancelled | closed | settled
                created_at     TEXT,
                filled_at      TEXT,
                closed_at      TEXT,
                cancelled_at   TEXT,
                settled_payout REAL
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
        cur = self._conn.execute(
            "SELECT * FROM events WHERE status NOT IN ('resolved', 'cancelled')"
        )
        return [dict(r) for r in cur.fetchall()]

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
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO markets
                (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(token_id) DO UPDATE SET
                condition_id=excluded.condition_id,
                market_slug=excluded.market_slug,
                bucket_lo_f=excluded.bucket_lo_f,
                bucket_hi_f=excluded.bucket_hi_f,
                outcome=excluded.outcome
            """,
            (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome),
        )
        self._conn.commit()

    def get_markets_for_event(self, condition_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM markets WHERE condition_id=?",
            (condition_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(
        self,
        timestamp: str,
        token_id: str,
        mc_prob: float,
        clob_price: float,
        edge: float,
        side: str,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO signals (timestamp, token_id, mc_prob, clob_price, edge, side)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, token_id, mc_prob, clob_price, edge, side),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

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

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DbClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
