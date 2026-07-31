from __future__ import annotations

import json

import pytest

from config.stations import STATIONS
from data.clob.history import (
    HistoricalClobSource,
    HistoricalDataIncomplete,
    _events_by_target_date,
)


class _FakeTransport:
    def __init__(self, event: dict, trades: list[dict]) -> None:
        self._event = event
        self._trades = trades
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, url: str, params: dict) -> object:
        self.calls.append((url, dict(params)))
        if url.endswith("/events"):
            return [self._event]
        if url.endswith("/trades"):
            return list(self._trades)
        raise AssertionError(f"unexpected URL {url}")


class _PagedTransport(_FakeTransport):
    def __init__(self, event: dict, pages: dict[int, list[dict]]) -> None:
        super().__init__(event, [])
        self._pages = pages

    def get_json(self, url: str, params: dict) -> object:
        self.calls.append((url, dict(params)))
        if url.endswith("/events"):
            return [self._event]
        if url.endswith("/trades"):
            return list(self._pages.get(int(params["offset"]), []))
        raise AssertionError(f"unexpected URL {url}")


class _CoverageTransport(_FakeTransport):
    def get_json(self, url: str, params: dict) -> object:
        self.calls.append((url, dict(params)))
        if url.endswith("/events"):
            if params.get("series_id") == "10005":
                return []
            return [self._event] if "april-2-" in params["slug"] else []
        if url.endswith("/public-search"):
            return {"events": [], "pagination": {"hasMore": False}}
        if url.endswith("/trades"):
            return list(self._trades)
        raise AssertionError(f"unexpected URL {url}")


class _ArchivedEventTransport(_FakeTransport):
    def get_json(self, url: str, params: dict) -> object:
        self.calls.append((url, dict(params)))
        if url.endswith("/events"):
            return []
        if url.endswith("/public-search"):
            return {
                "events": [
                    {
                        "id": "unrelated",
                        "slug": "highest-temperature-in-nyc-on-may-16-2026",
                        "markets": [],
                    },
                    self._event,
                ],
                "pagination": {"hasMore": False, "totalResults": 2},
            }
        if url.endswith("/trades"):
            return list(self._trades)
        raise AssertionError(f"unexpected URL {url}")


class _SeriesTransport(_FakeTransport):
    def get_json(self, url: str, params: dict) -> object:
        self.calls.append((url, dict(params)))
        if url.endswith("/events") and params.get("series_id") == "10005":
            return [self._event] if int(params["offset"]) == 0 else []
        if url.endswith("/events"):
            return []
        if url.endswith("/trades"):
            return list(self._trades)
        if url.endswith("/public-search"):
            return {"events": []}
        raise AssertionError(f"unexpected URL {url}")


def test_historical_source_normalizes_current_api_rows_without_synthetic_fill() -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "title": "Highest temperature in NYC on April 2?",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": json.dumps(["yes-1", "no-1"]),
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["1", "0"]),
                "feesEnabled": True,
                "feeType": "weather_fees",
                "feeSchedule": {
                    "rate": 0.05,
                    "exponent": 1,
                    "takerOnly": True,
                    "rebateRate": 0.25,
                },
                "makerBaseFee": 1000,
                "takerBaseFee": 1000,
                "orderPriceMinTickSize": 0.01,
                "orderMinSize": 5,
                "acceptingOrdersTimestamp": "2026-04-01T12:00:00Z",
                "closedTime": "2026-04-03T04:00:00Z",
            }
        ],
    }
    duplicate_trade = {
        "asset": "yes-1",
        "conditionId": "condition-1",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.42,
        "size": 25.0,
        "timestamp": 1775106300,
        "transactionHash": "tx-1",
        "proxyWallet": "wallet-1",
    }
    later_trade = {
        **duplicate_trade,
        "price": 0.44,
        "timestamp": 1775106600,
        "transactionHash": "tx-2",
    }
    transport = _FakeTransport(
        event,
        [later_trade, duplicate_trade, duplicate_trade],
    )
    source = HistoricalClobSource(transport=transport)

    dataset = source.fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )

    assert dataset.events[["event_id", "target_date", "station_id"]].to_dict(
        "records"
    ) == [
        {
            "event_id": "event-1",
            "target_date": "2026-04-02",
            "station_id": "KLGA",
        }
    ]
    assert dataset.markets[
        [
            "condition_id",
            "yes_token_id",
            "no_token_id",
            "bucket_lo_f",
            "bucket_hi_f",
        ]
    ].to_dict("records") == [
        {
            "condition_id": "condition-1",
            "yes_token_id": "yes-1",
            "no_token_id": "no-1",
            "bucket_lo_f": 70.0,
            "bucket_hi_f": 72.0,
        }
    ]
    market = dataset.markets.iloc[0]
    assert market["fees_enabled"]
    assert market["fee_type"] == "weather_fees"
    assert market["fee_rate"] == 0.05
    assert market["fee_exponent"] == 1.0
    assert market["fee_taker_only"]
    assert market["fee_rebate_rate"] == 0.25
    assert market["maker_base_fee"] == 1000.0
    assert market["taker_base_fee"] == 1000.0
    assert market["min_tick_size"] == 0.01
    assert market["min_order_size"] == 5.0
    assert market["yes_outcome_price"] == 1.0
    assert market["no_outcome_price"] == 0.0
    assert market["resolved_yes"]
    assert str(market["accepting_orders_at"]) == "2026-04-01 12:00:00+00:00"
    assert str(market["closed_at"]) == "2026-04-03 04:00:00+00:00"
    assert dataset.trades["transaction_hash"].tolist() == ["tx-1", "tx-2"]
    assert dataset.trades["source"].unique().tolist() == ["data_api_trade"]
    assert "is_filled" not in dataset.trades.columns
    trade_call = next(call for call in transport.calls if call[0].endswith("/trades"))
    assert trade_call[1]["takerOnly"] == "true"


def test_historical_source_recovers_legacy_events_from_daily_weather_series() -> None:
    event = {
        "id": "legacy-event",
        "slug": "highest-temperature-in-nyc-on-april-2",
        "endDate": "2025-04-02T12:00:00Z",
        "markets": [
            {
                "id": "legacy-market",
                "conditionId": "legacy-condition",
                "slug": "highest-temperature-in-nyc-on-april-2-70-71f",
                "clobTokenIds": ["yes", "no"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    transport = _SeriesTransport(event, [])

    dataset = HistoricalClobSource(transport=transport).fetch(
        start_date="2025-04-02",
        end_date="2025-04-02",
        station=STATIONS["KLGA"],
    )

    assert dataset.events[["event_id", "target_date"]].to_dict("records") == [
        {"event_id": "legacy-event", "target_date": "2025-04-02"}
    ]
    calls = [params for url, params in transport.calls if url.endswith("/events")]
    assert calls == [
        {"series_id": "10005", "closed": "true", "limit": 100, "offset": 0}
    ]


def test_series_target_date_comes_from_legacy_event_slug_not_close_date() -> None:
    indexed = _events_by_target_date(
        [
            {
                "slug": "highest-temperature-in-nyc-on-may-12",
                "endDate": "2025-05-13T12:00:00Z",
            }
        ]
    )

    assert list(indexed) == ["2025-05-12"]


def test_historical_source_recovers_archived_event_by_exact_slug(tmp_path) -> None:
    event = {
        "id": "event-archived",
        "slug": "highest-temperature-in-nyc-on-may-17-2026",
        "endDate": "2026-05-18T03:59:00Z",
        "markets": [
            {
                "id": "market-archived",
                "conditionId": "condition-archived",
                "slug": "highest-temperature-in-nyc-on-may-17-2026-70-71f",
                "clobTokenIds": ["yes-archived", "no-archived"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    trade = {
        "asset": "yes-archived",
        "conditionId": "condition-archived",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.42,
        "size": 25.0,
        "timestamp": 1779075000,
        "transactionHash": "tx-archived",
    }
    cache_dir = tmp_path / "raw-cache"
    cached_event_dir = cache_dir / "events"
    cached_event_dir.mkdir(parents=True)
    (cached_event_dir / f"{event['slug']}.json").write_text(
        "[]",
        encoding="utf-8",
    )
    transport = _ArchivedEventTransport(event, [trade])
    source = HistoricalClobSource(
        transport=transport,
        cache_dir=cache_dir,
    )

    first = source.fetch(
        start_date="2026-05-17",
        end_date="2026-05-17",
        station=STATIONS["KLGA"],
    )
    second = source.fetch(
        start_date="2026-05-17",
        end_date="2026-05-17",
        station=STATIONS["KLGA"],
    )

    assert first.events["event_id"].tolist() == ["event-archived"]
    assert first.markets["condition_id"].tolist() == ["condition-archived"]
    assert second.trades.equals(first.trades)
    searches = [
        params
        for url, params in transport.calls
        if url.endswith("/public-search")
    ]
    assert searches == [
        {
            "q": "highest temperature in nyc on may 17 2026",
            "events_status": "closed",
            "keep_closed_markets": 1,
            "limit_per_type": 10,
        }
    ]
    assert json.loads(
        (
            cache_dir
            / "events-recovered"
            / f"{event['slug']}.json"
        ).read_text(encoding="utf-8")
    ) == [event]


def test_historical_source_paginates_trades_until_a_short_page() -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }

    def trade(tx: str, timestamp: int) -> dict:
        return {
            "asset": "yes-1",
            "conditionId": "condition-1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.42,
            "size": 25.0,
            "timestamp": timestamp,
            "transactionHash": tx,
        }

    transport = _PagedTransport(
        event,
        {
            0: [trade("tx-1", 1775106300), trade("tx-2", 1775106600)],
            2: [trade("tx-3", 1775106900)],
        },
    )
    source = HistoricalClobSource(
        transport=transport,
        trade_page_size=2,
        max_trade_offset=4,
    )

    dataset = source.fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )

    assert dataset.trades["transaction_hash"].tolist() == [
        "tx-1",
        "tx-2",
        "tx-3",
    ]
    offsets = [
        params["offset"]
        for url, params in transport.calls
        if url.endswith("/trades")
    ]
    assert offsets == [0, 2]


def test_historical_source_default_page_size_matches_data_api_cap() -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }

    def trade(index: int) -> dict:
        return {
            "asset": "yes-1",
            "conditionId": "condition-1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.42,
            "size": 25.0,
            "timestamp": 1775106300 + index,
            "transactionHash": f"tx-{index}",
        }

    transport = _PagedTransport(
        event,
        {
            0: [trade(index) for index in range(1000)],
            1000: [trade(1000)],
        },
    )

    dataset = HistoricalClobSource(transport=transport).fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )

    assert len(dataset.trades) == 1001
    offsets = [
        params["offset"]
        for url, params in transport.calls
        if url.endswith("/trades")
    ]
    assert offsets == [0, 1000]


def test_historical_source_rechecks_legacy_cache_at_page_boundary(
    tmp_path,
) -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }

    def trade(index: int) -> dict:
        return {
            "asset": "yes-1",
            "conditionId": "condition-1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.42,
            "size": 25.0,
            "timestamp": 1775106300 + index,
            "transactionHash": f"tx-{index}",
        }

    cache_dir = tmp_path / "raw-cache"
    trade_cache = cache_dir / "trades"
    trade_cache.mkdir(parents=True)
    (trade_cache / "condition-1.json").write_text(
        json.dumps([trade(index) for index in range(1000)]),
        encoding="utf-8",
    )
    transport = _PagedTransport(
        event,
        {
            0: [trade(index) for index in range(1000)],
            1000: [trade(1000)],
        },
    )

    dataset = HistoricalClobSource(
        transport=transport,
        cache_dir=cache_dir,
    ).fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )

    assert len(dataset.trades) == 1001
    offsets = [
        params["offset"]
        for url, params in transport.calls
        if url.endswith("/trades")
    ]
    assert offsets == [0, 1000]


def test_historical_source_rejects_trade_history_truncated_at_api_boundary() -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    row = {
        "asset": "yes-1",
        "conditionId": "condition-1",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.42,
        "size": 25.0,
        "timestamp": 1775106300,
        "transactionHash": "tx",
    }
    transport = _PagedTransport(
        event,
        {
            0: [{**row, "transactionHash": "tx-1"}, {**row, "transactionHash": "tx-2"}],
            2: [{**row, "transactionHash": "tx-3"}, {**row, "transactionHash": "tx-4"}],
        },
    )
    source = HistoricalClobSource(
        transport=transport,
        trade_page_size=2,
        max_trade_offset=2,
    )

    with pytest.raises(HistoricalDataIncomplete, match="condition-1"):
        source.fetch(
            start_date="2026-04-02",
            end_date="2026-04-02",
            station=STATIONS["KLGA"],
        )


def test_historical_dataset_save_writes_immutable_manifest(tmp_path) -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    transport = _FakeTransport(event, [])
    dataset = HistoricalClobSource(transport=transport).fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )
    artifact_dir = tmp_path / "history"

    manifest = dataset.save(
        artifact_dir,
        fetched_at="2026-07-02T12:00:00Z",
    )

    persisted = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest
    assert manifest["schema_version"] == 2
    assert manifest["data_class"] == "raw_point_in_time"
    assert manifest["requested_range"] == ["2026-04-02", "2026-04-02"]
    assert manifest["row_counts"] == {
        "events": 1,
        "markets": 1,
        "trades": 0,
    }
    assert manifest["missing_target_dates"] == []
    assert manifest["incomplete_target_dates"] == ["2026-04-02"]
    assert set(manifest["sha256"]) == {
        "coverage.parquet",
        "events.parquet",
        "markets.parquet",
        "trades.parquet",
    }
    assert (artifact_dir / "coverage.parquet").exists()
    with pytest.raises(FileExistsError):
        dataset.save(artifact_dir)


def test_historical_dataset_coverage_includes_missing_target_dates() -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    trades = [
        {
            "asset": "yes-1",
            "conditionId": "condition-1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.42,
            "size": 25.0,
            "timestamp": 1775106300,
            "transactionHash": "tx-1",
        }
    ]
    dataset = HistoricalClobSource(
        transport=_CoverageTransport(event, trades)
    ).fetch(
        start_date="2026-04-02",
        end_date="2026-04-03",
        station=STATIONS["KLGA"],
    )

    assert dataset.coverage().to_dict("records") == [
        {
            "target_date": "2026-04-02",
            "event_count": 1,
            "market_count": 1,
            "token_count": 2,
            "trade_count": 1,
            "complete": True,
        },
        {
            "target_date": "2026-04-03",
            "event_count": 0,
            "market_count": 0,
            "token_count": 0,
            "trade_count": 0,
            "complete": False,
        },
    ]


def test_historical_dataset_load_verifies_artifact_hashes(tmp_path) -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    dataset = HistoricalClobSource(
        transport=_FakeTransport(event, [])
    ).fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )
    artifact_dir = tmp_path / "history"
    dataset.save(artifact_dir)

    loaded = dataset.load(artifact_dir)

    assert loaded.events.equals(dataset.events)
    (artifact_dir / "events.parquet").write_bytes(b"tampered")
    with pytest.raises(HistoricalDataIncomplete, match="events.parquet"):
        dataset.load(artifact_dir)


def test_historical_source_reuses_completed_raw_api_cache(tmp_path) -> None:
    event = {
        "id": "event-1",
        "slug": "highest-temperature-in-nyc-on-april-2-2026",
        "endDate": "2026-04-03T03:59:00Z",
        "markets": [
            {
                "id": "market-1",
                "conditionId": "condition-1",
                "slug": "highest-temperature-in-nyc-on-april-2-2026-70-71f",
                "clobTokenIds": ["yes-1", "no-1"],
                "outcomes": ["Yes", "No"],
            }
        ],
    }
    trades = [
        {
            "asset": "yes-1",
            "conditionId": "condition-1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.42,
            "size": 25.0,
            "timestamp": 1775106300,
            "transactionHash": "tx-1",
        }
    ]
    transport = _FakeTransport(event, trades)
    source = HistoricalClobSource(
        transport=transport,
        cache_dir=tmp_path / "raw-cache",
    )

    first = source.fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )
    second = source.fetch(
        start_date="2026-04-02",
        end_date="2026-04-02",
        station=STATIONS["KLGA"],
    )

    assert second.trades.equals(first.trades)
    event_calls = [
        params for url, params in transport.calls
        if url == "https://gamma-api.polymarket.com/events"
    ]
    assert sum("series_id" in params for params in event_calls) == 1
    assert sum("slug" in params for params in event_calls) == 0
    assert [url for url, _params in transport.calls].count(
        "https://data-api.polymarket.com/trades"
    ) == 1
