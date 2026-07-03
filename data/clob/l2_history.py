"""Forward L2 order-book evidence collection."""
from __future__ import annotations

import calendar
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from data.clob.history import JsonTransport, RequestsJsonTransport


class BookReader(Protocol):
    def get_book(self, token_id: str) -> dict: ...


@dataclass(frozen=True)
class ActiveTokens:
    event_count: int
    token_ids: list[str]


class GammaActiveTokenSource:
    """Discover current weather-market CLOB tokens without a bot database."""

    def __init__(
        self,
        *,
        station: object,
        transport: JsonTransport | None = None,
        gamma_api: str = "https://gamma-api.polymarket.com",
        horizon_days: int = 2,
        location_slug: str = "nyc",
    ) -> None:
        if horizon_days < 0:
            raise ValueError("horizon_days must be non-negative")
        self._station = station
        self._transport = transport or RequestsJsonTransport()
        self._gamma_api = gamma_api.rstrip("/")
        self._horizon_days = int(horizon_days)
        self._location_slug = location_slug

    def discover(self, now: str | pd.Timestamp) -> ActiveTokens:
        local_day = _utc(now).tz_convert(
            str(getattr(self._station, "timezone"))
        ).normalize()
        event_ids: set[str] = set()
        token_ids: list[str] = []
        for offset in range(self._horizon_days + 1):
            day = local_day + pd.Timedelta(days=offset)
            slug = (
                f"highest-temperature-in-{self._location_slug}-on-"
                f"{calendar.month_name[day.month].lower()}-{day.day}-{day.year}"
            )
            payload = self._transport.get_json(
                f"{self._gamma_api}/events",
                {"slug": slug},
            )
            events = payload if isinstance(payload, list) else []
            for event in events:
                if not isinstance(event, dict) or bool(event.get("closed", False)):
                    continue
                open_tokens: list[str] = []
                for market in event.get("markets", []) or []:
                    if not isinstance(market, dict) or bool(
                        market.get("closed", False)
                    ):
                        continue
                    if (
                        "acceptingOrders" in market
                        and not bool(market["acceptingOrders"])
                    ):
                        continue
                    open_tokens.extend(_token_ids(market.get("clobTokenIds")))
                if not open_tokens:
                    continue
                event_ids.add(str(event.get("id") or event.get("slug") or slug))
                token_ids.extend(open_tokens)
        return ActiveTokens(
            event_count=len(event_ids),
            token_ids=list(dict.fromkeys(token_ids)),
        )


@dataclass(frozen=True)
class L2Capture:
    snapshots: pd.DataFrame
    levels: pd.DataFrame


class L2SnapshotCollector:
    """Capture independent token books without hiding partial failures."""

    def __init__(self, reader: BookReader) -> None:
        self._reader = reader

    def capture(
        self,
        token_ids: list[str],
        *,
        captured_at: str | pd.Timestamp | None = None,
    ) -> L2Capture:
        timestamp = _utc(captured_at or pd.Timestamp.now(tz="UTC"))
        snapshot_rows: list[dict] = []
        level_rows: list[dict] = []
        for token_id in dict.fromkeys(str(value) for value in token_ids):
            try:
                book = self._reader.get_book(token_id)
                bids = sorted(
                    book.get("bids", []),
                    key=lambda level: float(level["price"]),
                    reverse=True,
                )
                asks = sorted(
                    book.get("asks", []),
                    key=lambda level: float(level["price"]),
                )
                snapshot_rows.append(
                    {
                        "token_id": token_id,
                        "captured_at": timestamp,
                        "status": "ok",
                        "error": None,
                        "best_bid": (
                            float(bids[0]["price"]) if bids else None
                        ),
                        "best_ask": (
                            float(asks[0]["price"]) if asks else None
                        ),
                        "bid_levels": len(bids),
                        "ask_levels": len(asks),
                    }
                )
                for side, levels in [("BID", bids), ("ASK", asks)]:
                    for index, level in enumerate(levels):
                        level_rows.append(
                            {
                                "token_id": token_id,
                                "captured_at": timestamp,
                                "side": side,
                                "level": index,
                                "price": float(level["price"]),
                                "size": float(level["size"]),
                            }
                        )
            except Exception as exc:
                snapshot_rows.append(
                    {
                        "token_id": token_id,
                        "captured_at": timestamp,
                        "status": "error",
                        "error": str(exc),
                        "best_bid": None,
                        "best_ask": None,
                        "bid_levels": 0,
                        "ask_levels": 0,
                    }
                )
        return L2Capture(
            snapshots=pd.DataFrame(
                snapshot_rows,
                columns=[
                    "token_id",
                    "captured_at",
                    "status",
                    "error",
                    "best_bid",
                    "best_ask",
                    "bid_levels",
                    "ask_levels",
                ],
            ),
            levels=pd.DataFrame(
                level_rows,
                columns=[
                    "token_id",
                    "captured_at",
                    "side",
                    "level",
                    "price",
                    "size",
                ],
            ),
        )


class L2HistoryStore:
    """Low-file-count append store for long-running L2 collection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    token_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    best_bid REAL,
                    best_ask REAL,
                    bid_levels INTEGER NOT NULL,
                    ask_levels INTEGER NOT NULL,
                    PRIMARY KEY (token_id, captured_at)
                );
                CREATE TABLE IF NOT EXISTS levels (
                    token_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    side TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    PRIMARY KEY (token_id, captured_at, side, level)
                );
                """
            )

    def append(self, capture: L2Capture) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.token_id,
                        _utc(row.captured_at).isoformat(),
                        row.status,
                        row.error,
                        row.best_bid,
                        row.best_ask,
                        int(row.bid_levels),
                        int(row.ask_levels),
                    )
                    for row in capture.snapshots.itertuples(index=False)
                ],
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO levels VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.token_id,
                        _utc(row.captured_at).isoformat(),
                        row.side,
                        int(row.level),
                        float(row.price),
                        float(row.size),
                    )
                    for row in capture.levels.itertuples(index=False)
                ],
            )

    def read_snapshots(self) -> pd.DataFrame:
        with self._connect() as connection:
            frame = pd.read_sql_query(
                "SELECT * FROM snapshots ORDER BY captured_at, token_id",
                connection,
            )
        frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
        return frame

    def read_levels(self) -> pd.DataFrame:
        with self._connect() as connection:
            frame = pd.read_sql_query(
                """
                SELECT * FROM levels
                ORDER BY captured_at, token_id, side, level
                """,
                connection,
            )
        frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
        return frame

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _token_ids(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(token_id) for token_id in value if str(token_id)]
