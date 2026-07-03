from __future__ import annotations

import json

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.reporting import ReplayArtifact
from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset
from scripts.run_point_in_time_backtest import main


def test_cli_runs_immutable_inputs_through_trade_settlement_and_report(
    tmp_path,
) -> None:
    forecast_root = tmp_path / "forecast"
    ForecastCache(
        snapshots=pd.DataFrame(
            [
                {
                    "snapshot_id": "snapshot",
                    "station_id": "KLGA",
                    "init_time": "2026-06-05T11:00:00Z",
                    "target_date": "2026-06-05",
                    "available_at": "2026-06-05T12:05:00Z",
                    "p10_f": 70.0,
                    "p50_f": 75.0,
                    "p90_f": 80.0,
                    "lineage_id": "fixed",
                }
            ]
        ),
        probabilities=pd.DataFrame(
            [
                {
                    "snapshot_id": "snapshot",
                    "condition_id": "condition",
                    "yes_probability": 0.80,
                }
            ]
        ),
        lineages=pd.DataFrame(
            [
                {
                    "lineage_id": "fixed",
                    "checkpoint_sha256": "a" * 64,
                    "calibration_sha256": "b" * 64,
                    "split_role": "post_test_holdout",
                    "mc_seed": 42,
                    "mc_samples": 2000,
                }
            ]
        ),
    ).save(forecast_root, created_at="2026-07-02T00:00:00Z")

    quotes_root = tmp_path / "quotes"
    SyntheticQuoteDataset(
        quotes=pd.DataFrame(
            [
                {
                    "condition_id": "condition",
                    "timestamp": "2026-06-05T12:05:00Z",
                    "bid_price_yes": 0.49,
                    "ask_price_yes": 0.50,
                    "estimated_spread": 0.01,
                    "capacity_usdc": 100.0,
                    "impact_per_usdc": 0.0,
                }
            ]
        ),
        start_at="2026-06-05T12:05:00Z",
        end_at="2026-06-05T12:05:00Z",
        decision_tick="5min",
        lookback="60min",
        max_staleness="30min",
    ).save(
        quotes_root,
        source_sha256="c" * 64,
        created_at="2026-07-02T00:00:00Z",
    )

    clob_root = tmp_path / "clob"
    HistoricalClobDataset(
        events=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "event_slug": "event-slug",
                    "event_title": "NYC high",
                    "target_date": "2026-06-05",
                    "event_end_at": pd.Timestamp(
                        "2026-06-05T12:00:00Z"
                    ),
                    "station_id": "KLGA",
                },
                {
                    "event_id": "outside-event",
                    "event_slug": "outside-event-slug",
                    "event_title": "Outside range",
                    "target_date": "2026-06-06",
                    "event_end_at": pd.Timestamp(
                        "2026-06-06T12:10:00Z"
                    ),
                    "station_id": "KLGA",
                },
            ]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "market_id": "market",
                    "condition_id": "condition",
                    "market_slug": "market-slug",
                    "yes_token_id": "yes",
                    "no_token_id": "no",
                    "bucket_lo_f": 70.0,
                    "bucket_hi_f": 80.0,
                    "fees_enabled": False,
                    "fee_rate": None,
                    "fee_exponent": None,
                    "market_end_at": pd.Timestamp(
                        "2026-06-05T12:00:00Z"
                    ),
                    "closed_at": pd.Timestamp(
                        "2026-06-05T12:10:00Z"
                    ),
                },
                {
                    "event_id": "outside-event",
                    "market_id": "outside-market",
                    "condition_id": "outside-condition",
                    "market_slug": "outside-market-slug",
                    "yes_token_id": "outside-yes",
                    "no_token_id": "outside-no",
                    "bucket_lo_f": 70.0,
                    "bucket_hi_f": 80.0,
                    "fees_enabled": False,
                    "fee_rate": None,
                    "fee_exponent": None,
                    "market_end_at": pd.Timestamp(
                        "2026-06-06T12:10:00Z"
                    ),
                },
            ]
        ),
        trades=pd.DataFrame(
            columns=[
                "condition_id",
                "token_id",
                "timestamp",
                "outcome",
                "side",
                "price",
                "size",
                "usdc",
                "transaction_hash",
                "proxy_wallet",
                "source",
            ]
        ),
        start_date="2026-06-05",
        end_date="2026-06-05",
        station_id="KLGA",
    ).save(clob_root, fetched_at="2026-07-02T00:00:00Z")

    actual_highs = tmp_path / "actual.csv"
    pd.DataFrame(
        [{"target_date": "2026-06-05", "actual_high_f": 75.0}]
    ).to_csv(actual_highs, index=False)
    output = tmp_path / "result"

    exit_code = main(
        [
            "--forecast-root",
            str(forecast_root),
            "--quotes-root",
            str(quotes_root),
            "--clob-root",
            str(clob_root),
            "--actual-highs",
            str(actual_highs),
            "--output",
            str(output),
            "--strategy",
            "baseline",
            "--start-target",
            "2026-06-05",
            "--end-target",
            "2026-06-05",
        ]
    )

    artifact = ReplayArtifact.load(output)
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert artifact.result.ending_cash_usd == 550.0
    assert len(artifact.result.trades) == 1
    assert len(artifact.result.settlements) == 1
    assert summary["net_pnl_usd"] == 50.0
    assert artifact.result.decisions["approved"].tolist() == [True]
    assert artifact.result.decisions["reason"].isna().all()
