from __future__ import annotations

import pandas as pd
import pytest

from data.clob.l2_history import ActiveTokens, L2Capture
from data.clob.orderbook import OrderbookFetcher
from scripts.collect_l2_history import (
    _build_book_reader,
    collect_active_books_once,
    main,
)


class _Db:
    def get_tradeable_events(self, now_date: str) -> list[dict]:
        assert now_date == "2026-07-02"
        return [{"condition_id": "event-a"}]

    def get_markets_for_event(self, condition_id: str) -> list[dict]:
        assert condition_id == "event-a"
        return [{"token_id": "yes"}, {"token_id": "no"}]


class _Collector:
    def capture(self, token_ids, *, captured_at):
        assert token_ids == ["yes", "no"]
        return L2Capture(
            snapshots=pd.DataFrame(
                [
                    {"token_id": token, "status": "ok"}
                    for token in token_ids
                ]
            ),
            levels=pd.DataFrame(),
        )


class _Store:
    def __init__(self) -> None:
        self.capture = None

    def append(self, capture) -> None:
        self.capture = capture


class _TokenSource:
    def discover(self, now):
        assert now == pd.Timestamp("2026-07-02T10:00:00Z")
        return ActiveTokens(event_count=2, token_ids=["yes", "no"])


class _EmptyTokenSource:
    def discover(self, now):
        return ActiveTokens(event_count=0, token_ids=[])


class _TransientlyFailingTokenSource:
    def __init__(self) -> None:
        self.calls = 0

    def discover(self, now):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("SSL EOF")
        return ActiveTokens(event_count=2, token_ids=["yes", "no"])


class _EmptyCollector:
    def capture(self, token_ids, *, captured_at):
        assert token_ids == []
        return L2Capture(
            snapshots=pd.DataFrame(columns=["status"]),
            levels=pd.DataFrame(),
        )


def test_collect_active_books_once_discovers_both_outcome_tokens() -> None:
    store = _Store()

    result = collect_active_books_once(
        _Db(),
        _Collector(),
        store,
        now=pd.Timestamp("2026-07-02T10:00:00Z"),
    )

    assert result == {"events": 1, "tokens": 2, "ok": 2, "errors": 0}
    assert store.capture is not None


def test_collect_once_uses_gamma_tokens_when_bot_db_is_empty() -> None:
    store = _Store()

    result = collect_active_books_once(
        None,
        _Collector(),
        store,
        now=pd.Timestamp("2026-07-02T10:00:00Z"),
        token_source=_TokenSource(),
    )

    assert result == {"events": 2, "tokens": 2, "ok": 2, "errors": 0}
    assert store.capture is not None


def test_collector_builds_public_rest_reader_without_optional_sdk() -> None:
    reader = _build_book_reader()

    assert isinstance(reader, OrderbookFetcher)


def test_cli_once_collects_from_token_source_without_bot_db(
    tmp_path,
) -> None:
    store = _Store()

    result = main(
        [
            "--once",
            "--output",
            str(tmp_path / "l2.sqlite"),
        ],
        token_source=_TokenSource(),
        collector=_Collector(),
        store=store,
        now_fn=lambda: pd.Timestamp("2026-07-02T10:00:00Z"),
    )

    assert result == 0
    assert store.capture is not None


def test_cli_once_fails_clearly_when_gamma_has_no_active_tokens(
    tmp_path,
    capsys,
) -> None:
    result = main(
        [
            "--once",
            "--output",
            str(tmp_path / "l2.sqlite"),
        ],
        token_source=_EmptyTokenSource(),
        collector=_EmptyCollector(),
        store=_Store(),
        now_fn=lambda: pd.Timestamp("2026-07-02T10:00:00Z"),
    )

    assert result == 2
    assert "status=no_active_markets" in capsys.readouterr().out


def test_continuous_cli_retries_transient_discovery_failure(
    tmp_path,
    capsys,
) -> None:
    source = _TransientlyFailingTokenSource()
    store = _Store()
    sleeps: list[float] = []

    def stop_after_recovery(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        main(
            [
                "--output",
                str(tmp_path / "l2.sqlite"),
                "--interval-seconds",
                "300",
                "--retry-initial-seconds",
                "10",
                "--retry-max-seconds",
                "60",
            ],
            token_source=source,
            collector=_Collector(),
            store=store,
            now_fn=lambda: pd.Timestamp("2026-07-02T10:00:00Z"),
            sleep_fn=stop_after_recovery,
        )

    captured = capsys.readouterr()
    assert "status=error" in captured.err
    assert "error_type=RuntimeError" in captured.err
    assert "retry_in_seconds=10" in captured.err
    assert "status=ok events=2 tokens=2 ok=2 errors=0" in captured.out
    assert sleeps == [10, 300]
    assert source.calls == 2
    assert store.capture is not None
