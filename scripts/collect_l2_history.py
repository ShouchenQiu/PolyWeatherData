"""Continuously collect real L2 books for active weather markets."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from data.clob.l2_history import (
    GammaActiveTokenSource,
    L2HistoryStore,
    L2SnapshotCollector,
)
from data.clob.orderbook import ClobRestClient, OrderbookFetcher
from db.schema import DbClient


def collect_active_books_once(
    db: Any,
    collector: Any,
    store: Any,
    *,
    now: pd.Timestamp,
    token_source: Any | None = None,
) -> dict[str, int]:
    now_utc = _utc(now)
    if token_source is not None:
        active = token_source.discover(now_utc)
        event_count = int(active.event_count)
        token_ids = list(active.token_ids)
    else:
        events = db.get_tradeable_events(now_utc.date().isoformat())
        token_ids = []
        for event in events:
            token_ids.extend(
                str(market["token_id"])
                for market in db.get_markets_for_event(event["condition_id"])
            )
        token_ids = list(dict.fromkeys(token_ids))
        event_count = len(events)
    capture = collector.capture(token_ids, captured_at=now_utc)
    store.append(capture)
    statuses = capture.snapshots.get("status", pd.Series(dtype=str))
    return {
        "events": event_count,
        "tokens": len(token_ids),
        "ok": int((statuses == "ok").sum()),
        "errors": int((statuses == "error").sum()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll current active weather-market L2 books."
    )
    parser.add_argument(
        "--source",
        choices=["gamma", "db"],
        default="gamma",
        help="discover tokens directly from Gamma (default) or read a bot DB",
    )
    parser.add_argument("--db", type=Path, default=Path("polyweather.db"))
    parser.add_argument("--station", choices=sorted(STATIONS), default="KLGA")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_store/backtest/l2/live.sqlite"),
    )
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument(
        "--retry-initial-seconds",
        type=int,
        default=30,
        help="initial delay after a failed polling cycle (default: 30)",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=int,
        default=300,
        help="maximum exponential retry delay (default: 300)",
    )
    parser.add_argument("--once", action="store_true")
    return parser


def _build_book_reader(
    host: str = "https://clob.polymarket.com",
) -> OrderbookFetcher:
    return OrderbookFetcher(ClobRestClient(host=host))


def main(
    argv: list[str] | None = None,
    *,
    token_source: Any | None = None,
    collector: Any | None = None,
    store: Any | None = None,
    db: Any | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    if args.retry_initial_seconds <= 0:
        raise ValueError("--retry-initial-seconds must be positive")
    if args.retry_max_seconds < args.retry_initial_seconds:
        raise ValueError(
            "--retry-max-seconds must be at least --retry-initial-seconds"
        )
    settings = Settings()
    if token_source is None and args.source == "gamma":
        token_source = GammaActiveTokenSource(
            station=STATIONS[args.station],
            gamma_api=settings.gamma_api_url,
        )
    if args.source == "db" and db is None:
        db = DbClient(args.db)
    collector = collector or L2SnapshotCollector(
        _build_book_reader(settings.clob_api_url)
    )
    store = store or L2HistoryStore(args.output)
    clock = now_fn or (lambda: pd.Timestamp(datetime.now(timezone.utc)))
    sleep = sleep_fn or time.sleep
    retry_delay = args.retry_initial_seconds
    while True:
        now = _utc(clock())
        try:
            result = collect_active_books_once(
                db,
                collector,
                store,
                now=now,
                token_source=token_source,
            )
        except Exception as exc:
            if args.once:
                raise
            print(
                f"{now.isoformat()} status=error "
                f"error_type={type(exc).__name__} "
                f"retry_in_seconds={retry_delay} error={str(exc)!r}",
                file=sys.stderr,
                flush=True,
            )
            sleep(retry_delay)
            retry_delay = min(retry_delay * 2, args.retry_max_seconds)
            continue
        retry_delay = args.retry_initial_seconds
        status = (
            "ok" if result["tokens"] > 0 else "no_active_markets"
        )
        print(
            f"{now.isoformat()} status={status} events={result['events']} "
            f"tokens={result['tokens']} ok={result['ok']} "
            f"errors={result['errors']}",
            flush=True,
        )
        if args.once:
            return 0 if result["tokens"] > 0 else 2
        sleep(args.interval_seconds)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


if __name__ == "__main__":
    raise SystemExit(main())
