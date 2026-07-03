from __future__ import annotations

import pandas as pd

from config.stations import STATIONS
from data.clob.l2_history import (
    GammaActiveTokenSource,
    L2HistoryStore,
    L2SnapshotCollector,
)


class _Reader:
    def get_book(self, token_id: str) -> dict:
        if token_id == "bad":
            raise RuntimeError("network error")
        return {
            "bids": [{"price": 0.49, "size": 10.0}],
            "asks": [{"price": 0.51, "size": 20.0}],
        }


class _GammaTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, url: str, params: dict) -> object:
        self.calls.append((url, dict(params)))
        slug = str(params["slug"])
        if "july-2-2026" in slug:
            return [
                {
                    "id": "event-2",
                    "slug": slug,
                    "closed": False,
                    "markets": [
                        {
                            "closed": False,
                            "clobTokenIds": '["yes-2", "no-2"]',
                        },
                        {
                            "closed": True,
                            "clobTokenIds": '["closed-yes", "closed-no"]',
                        },
                    ],
                }
            ]
        if "july-3-2026" in slug:
            return [
                {
                    "id": "event-3",
                    "slug": slug,
                    "closed": False,
                    "markets": [
                        {
                            "closed": False,
                            "clobTokenIds": ["yes-3", "no-3"],
                        }
                    ],
                }
            ]
        return []


def test_gamma_token_source_discovers_open_books_without_bot_db() -> None:
    transport = _GammaTransport()
    source = GammaActiveTokenSource(
        station=STATIONS["KLGA"],
        transport=transport,
        horizon_days=2,
    )

    active = source.discover(
        pd.Timestamp("2026-07-02T22:00:00Z")
    )

    assert active.event_count == 2
    assert active.token_ids == ["yes-2", "no-2", "yes-3", "no-3"]
    assert [params["slug"] for _url, params in transport.calls] == [
        "highest-temperature-in-nyc-on-july-2-2026",
        "highest-temperature-in-nyc-on-july-3-2026",
        "highest-temperature-in-nyc-on-july-4-2026",
    ]


def test_l2_capture_preserves_levels_and_records_per_token_failure() -> None:
    capture = L2SnapshotCollector(_Reader()).capture(
        ["yes", "bad"],
        captured_at="2026-07-02T10:00:00Z",
    )

    assert capture.snapshots["token_id"].tolist() == ["yes", "bad"]
    assert capture.snapshots["status"].tolist() == ["ok", "error"]
    assert capture.snapshots.iloc[1]["error"] == "network error"
    assert capture.levels[["token_id", "side", "level"]].to_dict("records") == [
        {"token_id": "yes", "side": "BID", "level": 0},
        {"token_id": "yes", "side": "ASK", "level": 0},
    ]
    assert capture.levels["captured_at"].unique().tolist() == [
        pd.Timestamp("2026-07-02T10:00:00Z")
    ]


def test_l2_store_appends_capture_idempotently_in_one_sqlite_file(tmp_path) -> None:
    capture = L2SnapshotCollector(_Reader()).capture(
        ["yes"],
        captured_at="2026-07-02T10:00:00Z",
    )
    store = L2HistoryStore(tmp_path / "l2.sqlite")

    store.append(capture)
    store.append(capture)

    assert len(store.read_snapshots()) == 1
    assert len(store.read_levels()) == 2
