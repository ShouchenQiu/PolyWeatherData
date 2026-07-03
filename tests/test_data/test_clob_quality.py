from __future__ import annotations

import pandas as pd
import pytest

from data.clob.history import HistoricalClobDataset, HistoricalDataIncomplete
from data.clob.quality import (
    audit_historical_clob,
    build_clob_audit_report,
)


def _dataset() -> HistoricalClobDataset:
    return HistoricalClobDataset(
        events=pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "target_date": "2026-04-01",
                }
            ]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "condition_id": "condition-1",
                    "yes_token_id": "yes-1",
                    "no_token_id": "no-1",
                }
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "condition-1",
                    "token_id": "yes-1",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "price": 0.5,
                    "size": 10.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-02",
        station_id="KLGA",
    )


def test_clob_audit_accepts_only_explicitly_excluded_missing_dates() -> None:
    report = audit_historical_clob(
        _dataset(),
        exclusions={"2026-04-02": "provider_event_unavailable"},
    )

    assert report["status"] == "accepted_with_exclusions"
    assert report["requested_days"] == 2
    assert report["eligible_days"] == 1
    assert report["excluded_days"] == {
        "2026-04-02": "provider_event_unavailable"
    }
    assert report["quality"] == {
        "duplicate_trades": 0,
        "invalid_prices": 0,
        "nonpositive_sizes": 0,
        "null_timestamps": 0,
        "unknown_conditions": 0,
        "unknown_tokens": 0,
    }


def test_clob_audit_rejects_unexplained_coverage_gap() -> None:
    with pytest.raises(
        HistoricalDataIncomplete,
        match="unexplained incomplete target dates: 2026-04-02",
    ):
        audit_historical_clob(_dataset(), exclusions={})


def test_clob_audit_report_flags_bucket_partition_gaps() -> None:
    dataset = HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event-1", "target_date": "2026-04-01"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "condition_id": "low",
                    "yes_token_id": "low-yes",
                    "no_token_id": "low-no",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": 60.0,
                },
                {
                    "event_id": "event-1",
                    "condition_id": "high",
                    "yes_token_id": "high-yes",
                    "no_token_id": "high-no",
                    "bucket_lo_f": 61.0,
                    "bucket_hi_f": float("inf"),
                },
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "low",
                    "token_id": "low-yes",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "price": 0.5,
                    "size": 10.0,
                    "usdc": 5.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    )

    report = build_clob_audit_report(dataset)

    assert not report.passed
    assert report.event_checks.iloc[0]["bucket_partition_status"] == "FAIL"
    assert "gap [60.0, 61.0)" in report.event_checks.iloc[0]["bucket_issue"]
    assert "bucket_partitions" in report.failures
    with pytest.raises(HistoricalDataIncomplete, match="bucket_partitions"):
        audit_historical_clob(dataset, exclusions={})


def test_clob_audit_report_rejects_repeated_data_api_page_caps() -> None:
    markets = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "condition_id": "low",
                "yes_token_id": "low-yes",
                "no_token_id": "low-no",
                "bucket_lo_f": float("-inf"),
                "bucket_hi_f": 60.0,
            },
            {
                "event_id": "event-1",
                "condition_id": "high",
                "yes_token_id": "high-yes",
                "no_token_id": "high-no",
                "bucket_lo_f": 60.0,
                "bucket_hi_f": float("inf"),
            },
        ]
    )
    rows = []
    for condition, token in (("low", "low-yes"), ("high", "high-yes")):
        for index in range(1000):
            rows.append(
                {
                    "condition_id": condition,
                    "token_id": token,
                    "timestamp": (
                        pd.Timestamp("2026-04-01T00:00:00Z")
                        + pd.Timedelta(seconds=index)
                    ),
                    "price": 0.5,
                    "size": float(index + 1),
                    "usdc": 0.5 * (index + 1),
                }
            )
    dataset = HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event-1", "target_date": "2026-04-01"}]
        ),
        markets=markets,
        trades=pd.DataFrame(rows),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    )

    report = build_clob_audit_report(dataset)

    assert not report.passed
    assert "trade_history_truncation" in report.failures
    assert report.summary["suspected_truncated_markets"] == 2


def test_clob_audit_report_accepts_consistent_quotes_and_actuals() -> None:
    dataset = HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event-1", "target_date": "2026-04-01"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "condition_id": "low",
                    "yes_token_id": "low-yes",
                    "no_token_id": "low-no",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": 60.0,
                },
                {
                    "event_id": "event-1",
                    "condition_id": "high",
                    "yes_token_id": "high-yes",
                    "no_token_id": "high-no",
                    "bucket_lo_f": 60.0,
                    "bucket_hi_f": float("inf"),
                },
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "high",
                    "token_id": "high-yes",
                    "timestamp": "2026-04-01T11:59:00Z",
                    "price": 0.4,
                    "size": 10.0,
                    "usdc": 4.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    )
    quotes = pd.DataFrame(
        [
            {
                "condition_id": "high",
                "timestamp": "2026-04-01T12:00:00Z",
                "bid_price_yes": 0.39,
                "mid_price_yes": 0.40,
                "ask_price_yes": 0.41,
                "estimated_spread": 0.02,
                "impact_per_usdc": 0.001,
                "capacity_usdc": 5.0,
                "rolling_volume_usdc": 10.0,
                "last_trade_at": "2026-04-01T11:59:00Z",
                "staleness_seconds": 60.0,
                "quote_quality": "two_sided_trade_proxy",
            }
        ]
    )
    actuals = pd.DataFrame(
        [{"target_date": "2026-04-01", "actual_high_f": 61.0}]
    )

    report = build_clob_audit_report(
        dataset,
        quotes=quotes,
        actual_highs=actuals,
        clob_manifest_sha256="a" * 64,
        quote_source_sha256="a" * 64,
        quote_max_staleness="30min",
    )

    assert report.passed
    assert report.failures == ()
    assert report.summary["quotes"] == 1
    assert report.summary["actual_days"] == 1
    assert set(report.checks["status"]) == {"PASS"}
    assert report.quote_checks.iloc[0]["status"] == "PASS"


def test_clob_audit_report_rejects_token_from_another_market() -> None:
    dataset = HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event-1", "target_date": "2026-04-01"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "condition_id": "low",
                    "yes_token_id": "low-yes",
                    "no_token_id": "low-no",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": 60.0,
                },
                {
                    "event_id": "event-1",
                    "condition_id": "high",
                    "yes_token_id": "high-yes",
                    "no_token_id": "high-no",
                    "bucket_lo_f": 60.0,
                    "bucket_hi_f": float("inf"),
                },
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "low",
                    "token_id": "high-yes",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "price": 0.5,
                    "size": 10.0,
                    "usdc": 5.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    )

    report = build_clob_audit_report(dataset)

    assert not report.passed
    assert "trades" in report.failures


def test_clob_audit_report_accumulates_trade_quote_and_resolution_failures() -> None:
    dataset = HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event-1", "target_date": "2026-04-01"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "condition_id": "low",
                    "yes_token_id": "low-yes",
                    "no_token_id": "low-no",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": 60.0,
                },
                {
                    "event_id": "event-1",
                    "condition_id": "high",
                    "yes_token_id": "high-yes",
                    "no_token_id": "high-no",
                    "bucket_lo_f": 60.0,
                    "bucket_hi_f": float("inf"),
                },
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "low",
                    "token_id": "low-yes",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "price": 1.5,
                    "size": 0.0,
                    "usdc": 2.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    )
    quotes = pd.DataFrame(
        [
            {
                "condition_id": "unknown",
                "timestamp": "2026-04-01T12:00:00Z",
                "bid_price_yes": 0.60,
                "mid_price_yes": 0.50,
                "ask_price_yes": 0.40,
                "estimated_spread": 0.25,
                "impact_per_usdc": -0.1,
                "capacity_usdc": -1.0,
                "rolling_volume_usdc": -2.0,
                "last_trade_at": "2026-04-01T12:05:00Z",
                "staleness_seconds": 9999.0,
            }
        ]
    )

    report = build_clob_audit_report(
        dataset,
        quotes=quotes,
        actual_highs=pd.DataFrame(
            [{"target_date": "2026-04-01", "actual_high_f": float("nan")}]
        ),
        clob_manifest_sha256="a" * 64,
        quote_source_sha256="b" * 64,
        quote_max_staleness="30min",
    )

    assert not report.passed
    assert {"trades", "synthetic_quotes", "actual_bucket_resolution"}.issubset(
        report.failures
    )
    assert set(
        report.quote_checks.loc[
            report.quote_checks["status"] == "FAIL", "check"
        ]
    ) == {
        "known_conditions",
        "price_order",
        "spread",
        "capacity_and_impact",
        "past_only_timestamps",
        "staleness",
        "source_manifest",
    }
