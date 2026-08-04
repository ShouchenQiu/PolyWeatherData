from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DbClient:
    """SQLite client for PolyWeatherData analysis storage.

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
                discovery_status  TEXT NOT NULL DEFAULT 'parsed'
                                  CHECK (discovery_status IN ('parsed', 'unrecognized_slug', 'invalid_market'))
            );

            CREATE TABLE IF NOT EXISTS market_price_snapshots (
                price_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id          TEXT NOT NULL REFERENCES markets(token_id),
                condition_id      TEXT REFERENCES events(condition_id),
                tick_utc          TEXT NOT NULL,
                best_bid          REAL,
                best_ask          REAL,
                mid_price         REAL,
                UNIQUE(token_id, tick_utc)
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

    def get_tradeable_events(self, now_date: str) -> list[dict]:
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

    def get_all_events(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM events")
        return [dict(r) for r in cur.fetchall()]

    def event_exists(self, condition_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM events WHERE condition_id = ?", (condition_id,)
        ).fetchone()
        return row is not None

    def update_event_status(self, condition_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE events SET status = ? WHERE condition_id = ?",
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
        market_id: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO markets (token_id, condition_id, market_id, market_slug,
                                 bucket_lo_f, bucket_hi_f, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(token_id) DO UPDATE SET
                market_id=excluded.market_id,
                market_slug=excluded.market_slug,
                bucket_lo_f=excluded.bucket_lo_f,
                bucket_hi_f=excluded.bucket_hi_f,
                outcome=excluded.outcome
            """,
            (token_id, condition_id, market_id, market_slug, bucket_lo_f, bucket_hi_f, outcome),
        )
        self._conn.commit()

    def get_markets_for_event(self, condition_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM markets WHERE condition_id = ?", (condition_id,)
        )
        return [dict(r) for r in cur.fetchall()]

    def get_market_by_token(self, token_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM markets WHERE token_id = ?", (token_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Market Price Snapshots
    # ------------------------------------------------------------------

    def upsert_price_snapshot(
        self,
        token_id: str,
        condition_id: str | None,
        tick_utc: str,
        best_bid: float | None,
        best_ask: float | None,
        mid_price: float | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO market_price_snapshots
                (token_id, condition_id, tick_utc, best_bid, best_ask, mid_price)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(token_id, tick_utc) DO UPDATE SET
                condition_id=excluded.condition_id,
                best_bid=excluded.best_bid,
                best_ask=excluded.best_ask,
                mid_price=excluded.mid_price
            """,
            (token_id, condition_id, tick_utc, best_bid, best_ask, mid_price),
        )
        self._conn.commit()

    def get_price_snapshots_for_event(
        self, condition_id: str,
    ) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM market_price_snapshots
            WHERE condition_id = ?
            ORDER BY tick_utc
            """,
            (condition_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Low-level escape hatches
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write statement and commit."""
        self._conn.execute(sql, params)
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list:
        """Execute a read query and return all rows."""
        return self._conn.execute(sql, params).fetchall()

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
