from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset
from data.clob.history import HistoricalDataIncomplete
from scripts.run_market_calibration import main


def test_market_calibration_cli_rejects_schema_v1_inputs(tmp_path) -> None:
    clob_root = tmp_path / "clob-v1"
    clob_root.mkdir()
    (clob_root / "manifest.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    with pytest.raises(HistoricalDataIncomplete, match="schema-v2 CLOB"):
        main(
            [
                "--clob-root",
                str(clob_root),
                "--actual-highs",
                str(tmp_path / "actuals.csv"),
                "--quotes-root",
                str(tmp_path / "quotes"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "output").exists()


def test_market_calibration_cli_writes_governed_artifact(tmp_path) -> None:
    clob_root = tmp_path / "clob-v2"
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "event_slug": ["event"],
            "event_title": ["Event"],
            "target_date": ["2025-04-01"],
            "event_end_at": pd.to_datetime(["2025-04-01T12:00:00Z"], utc=True),
            "station_id": ["KLGA"],
        }
    )
    markets = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "market_id": ["m1", "m2"],
            "condition_id": ["c1", "c2"],
            "market_slug": ["below-50", "above-50"],
            "yes_token_id": ["y1", "y2"],
            "no_token_id": ["n1", "n2"],
            "bucket_lo_f": [float("-inf"), 50.0],
            "bucket_hi_f": [50.0, float("inf")],
            "yes_outcome_price": [0.0, 1.0],
            "no_outcome_price": [1.0, 0.0],
            "resolved_yes": [False, True],
            "fees_enabled": [False, False],
            "fee_type": ["", ""],
            "fee_rate": [None, None],
            "fee_exponent": [None, None],
            "fee_taker_only": [False, False],
            "fee_rebate_rate": [None, None],
            "maker_base_fee": [None, None],
            "taker_base_fee": [None, None],
            "min_tick_size": [0.01, 0.01],
            "min_order_size": [5.0, 5.0],
            "accepting_orders": [False, False],
            "active": [True, True],
            "closed": [True, True],
            "neg_risk": [True, True],
            "market_start_at": pd.to_datetime(
                ["2025-03-30T00:00:00Z"] * 2, utc=True
            ),
            "market_end_at": pd.to_datetime(
                ["2025-04-01T12:00:00Z"] * 2, utc=True
            ),
            "accepting_orders_at": pd.to_datetime(
                ["2025-03-30T00:00:00Z"] * 2, utc=True
            ),
            "closed_at": pd.to_datetime(
                ["2025-04-02T05:00:00Z"] * 2, utc=True
            ),
        }
    )
    trades = pd.DataFrame(
        {
            "condition_id": ["c1", "c2"],
            "token_id": ["y1", "y2"],
            "timestamp": pd.to_datetime(
                ["2025-04-01T12:00:00Z", "2025-04-01T12:00:00Z"], utc=True
            ),
            "outcome": ["YES", "YES"],
            "side": ["BUY", "BUY"],
            "price": [0.4, 0.6],
            "size": [10.0, 10.0],
            "usdc": [4.0, 6.0],
            "transaction_hash": ["tx1", "tx2"],
            "proxy_wallet": ["w1", "w2"],
            "source": ["test", "test"],
        }
    )
    excluded_event = events.copy()
    excluded_event["event_id"] = "e2"
    excluded_event["event_slug"] = "excluded-event"
    excluded_event["target_date"] = "2025-04-02"
    excluded_event["event_end_at"] = pd.Timestamp("2025-04-02T12:00:00Z")
    events = pd.concat([events, excluded_event], ignore_index=True)
    excluded_markets = markets.copy()
    excluded_markets["event_id"] = "e2"
    excluded_markets["market_id"] = ["m3", "m4"]
    excluded_markets["condition_id"] = ["c3", "c4"]
    excluded_markets["yes_token_id"] = ["y3", "y4"]
    excluded_markets["no_token_id"] = ["n3", "n4"]
    excluded_markets["closed_at"] = pd.Timestamp("2025-04-03T05:00:00Z")
    markets = pd.concat([markets, excluded_markets], ignore_index=True)
    excluded_trades = trades.copy()
    excluded_trades["condition_id"] = ["c3", "c4"]
    excluded_trades["token_id"] = ["y3", "y4"]
    excluded_trades["timestamp"] = pd.Timestamp("2025-04-02T12:00:00Z")
    excluded_trades["transaction_hash"] = ["tx3", "tx4"]
    trades = pd.concat([trades, excluded_trades], ignore_index=True)
    HistoricalClobDataset(
        events=events,
        markets=markets,
        trades=trades,
        start_date="2025-04-01",
        end_date="2025-04-02",
        station_id="KLGA",
    ).save(clob_root, fetched_at="2026-01-01T00:00:00Z")

    actuals_path = tmp_path / "actuals.csv"
    pd.DataFrame(
        {"target_date": ["2025-04-01"], "actual_high_f": [55.0]}
    ).to_csv(actuals_path, index=False)
    truth_exclusions_path = tmp_path / "actuals.truth_exclusions.parquet"
    pd.DataFrame(
        {
            "station_id": ["KLGA"],
            "target_date": ["2025-04-02"],
            "observed_hours": [23],
            "expected_hours": [24],
            "exclusion_reason": ["incomplete_station_local_day"],
        }
    ).to_parquet(truth_exclusions_path, index=False)
    actual_manifest = {
        "schema_version": 2,
        "data_class": "market_resolution_actual_highs",
        "daily_high_source": "weather_com_raw_observation_max",
        "hour_coverage_contract": (
            "complete_station_local_day_or_explicit_exclusion"
        ),
        "station_id": "KLGA",
        "requested_range": ["2025-04-01", "2025-04-02"],
        "requested_day_count": 2,
        "row_count": 1,
        "exclusion_count": 1,
        "truth_exclusions_file": truth_exclusions_path.name,
        "sha256": {
            actuals_path.name: hashlib.sha256(actuals_path.read_bytes()).hexdigest(),
            truth_exclusions_path.name: hashlib.sha256(
                truth_exclusions_path.read_bytes()
            ).hexdigest(),
        },
    }
    actuals_path.with_suffix(".csv.manifest.json").write_text(
        json.dumps(actual_manifest), encoding="utf-8"
    )

    quotes_root = tmp_path / "quotes"
    quote_rows = []
    for condition_id, mid in (
        ("c1", 0.4),
        ("c2", 0.8),
        ("c3", 0.4),
        ("c4", 0.8),
    ):
        quote_rows.append(
            {
                "condition_id": condition_id,
                "timestamp": pd.Timestamp("2025-04-01T12:05:00Z"),
                "mid_price_yes": mid,
                "bid_price_yes": mid - 0.01,
                "ask_price_yes": mid + 0.01,
                "estimated_spread": 0.02,
                "impact_per_usdc": 0.0,
                "capacity_usdc": 10.0,
                "rolling_volume_usdc": 10.0,
                "trade_count": 1,
                "buy_yes_trade_count": 1,
                "sell_yes_trade_count": 0,
                "last_trade_at": pd.Timestamp("2025-04-01T12:00:00Z"),
                "staleness_seconds": 300.0,
                "quote_quality": "observed",
                "source": "test",
            }
        )
    clob_sha = hashlib.sha256((clob_root / "manifest.json").read_bytes()).hexdigest()
    SyntheticQuoteDataset(
        quotes=pd.DataFrame(quote_rows),
        start_at="2025-04-01T12:05:00Z",
        end_at="2025-04-01T12:05:00Z",
        decision_tick="5min",
        lookback="60min",
        max_staleness="30min",
    ).save(quotes_root, source_sha256=clob_sha)

    output = tmp_path / "result"
    assert main(
        [
            "--clob-root",
            str(clob_root),
            "--actual-highs",
            str(actuals_path),
            "--quotes-root",
            str(quotes_root),
            "--output",
            str(output),
            "--bootstrap-samples",
            "20",
            "--min-event-days",
            "1",
            "--min-condition-cells",
            "1",
        ]
    ) == 0
    expected = {
        "transaction_panel.parquet",
        "calibration_bins.parquet",
        "price_lead_cells.parquet",
        "fold_stability.parquet",
        "resolution_latency.parquet",
        "calibration_candidates.parquet",
        "buy_settlement_returns.parquet",
        "multibucket_decomposition.parquet",
        "question_ledger.json",
        "report.json",
        "manifest.json",
        "calibration_curve.html",
        "price_lead_heatmap.html",
        "fold_stability.html",
        "input_truth_exclusions.parquet",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    ledger = json.loads((output / "question_ledger.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert ledger["executable_edge"]["status"] == "not_measured"
    assert ledger["paper_style_weather_miscalibration"][
        "data_contract_limitations"
    ] == [
        {
            "artifact": "input_truth_exclusions.parquet",
            "excluded_event_days": 1,
            "reason": "incomplete_station_local_day",
        }
    ]
    assert report["eligible_input_rows"]["events"] == 1
    assert report["truth_exclusions"] == {
        "count": 1,
        "target_dates": ["2025-04-02"],
    }
