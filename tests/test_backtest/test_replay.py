from __future__ import annotations

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.replay import PointInTimeReplay, ReplayStrategy


def test_replay_cannot_trade_on_forecast_before_available_at() -> None:
    cache = ForecastCache(
        snapshots=pd.DataFrame(
            [
                {
                    "snapshot_id": "snapshot",
                    "station_id": "KLGA",
                    "init_time": "2026-06-05T11:00:00Z",
                    "target_date": "2026-06-05",
                    "available_at": "2026-06-05T12:05:00Z",
                    "p10_f": 70.0,
                    "p50_f": 74.0,
                    "p90_f": 78.0,
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
    )
    quotes = pd.DataFrame(
        [
            {
                "condition_id": "condition",
                "timestamp": tick,
                "bid_price_yes": 0.49,
                "ask_price_yes": 0.50,
                "estimated_spread": 0.01,
                "capacity_usdc": 100.0,
                "impact_per_usdc": 0.0,
            }
            for tick in pd.to_datetime(
                ["2026-06-05T12:00:00Z", "2026-06-05T12:05:00Z"],
                utc=True,
            )
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "condition_id": "condition",
                "event_id": "event",
                "target_date": "2026-06-05",
                "yes_token_id": "yes",
                "no_token_id": "no",
                "bucket_lo_f": 70.0,
                "bucket_hi_f": 80.0,
            }
        ]
    )
    replay = PointInTimeReplay(
        forecast_cache=cache,
        quotes=quotes,
        markets=markets,
        station_id="KLGA",
    )

    before = replay.tick("2026-06-05T12:00:00Z")
    after = replay.tick("2026-06-05T12:05:00Z")

    assert before.empty
    assert after.iloc[0]["token_id"] == "yes"
    assert after.iloc[0]["side"] == "BUY"
    assert after.iloc[0]["size_usd"] == 50.0
    assert replay.portfolio.cash_usd == 450.0


def test_replay_settlement_redeems_winning_inventory_and_releases_capital() -> None:
    cache = ForecastCache(
        snapshots=pd.DataFrame(
            columns=[
                "snapshot_id", "station_id", "init_time", "target_date",
                "available_at", "p10_f", "p50_f", "p90_f", "lineage_id",
            ]
        ),
        probabilities=pd.DataFrame(
            columns=["snapshot_id", "condition_id", "yes_probability"]
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
    )
    markets = pd.DataFrame(
        [
            {
                "condition_id": "condition",
                "event_id": "event",
                "target_date": "2026-06-05",
                "yes_token_id": "yes",
                "no_token_id": "no",
                "bucket_lo_f": 70.0,
                "bucket_hi_f": 80.0,
            }
        ]
    )
    replay = PointInTimeReplay(
        forecast_cache=cache,
        quotes=pd.DataFrame(
            columns=[
                "condition_id", "timestamp", "bid_price_yes",
                "ask_price_yes", "estimated_spread", "capacity_usdc",
                "impact_per_usdc",
            ]
        ),
        markets=markets,
        station_id="KLGA",
    )
    replay.portfolio.cash_usd = 450.0
    replay.portfolio.shares_by_token["yes"] = 100.0
    replay.portfolio.notional_by_token["yes"] = 50.0

    settlement = replay.settle_event(
        "event",
        actual_high_f=75.0,
        settled_at="2026-06-06T04:00:00Z",
    )

    assert settlement.iloc[0]["payout_usd"] == 100.0
    assert settlement.iloc[0]["realized_pnl_usd"] == 50.0
    assert replay.portfolio.cash_usd == 550.0
    assert replay.portfolio.shares_by_token["yes"] == 0.0


def test_liquid_strategy_skips_wide_spread_quote() -> None:
    # Reuse the point-in-time fixture from the first test with a wide quote.
    cache = ForecastCache(
        snapshots=pd.DataFrame([{
            "snapshot_id": "s", "station_id": "KLGA",
            "init_time": "2026-06-05T11:00:00Z",
            "target_date": "2026-06-05",
            "available_at": "2026-06-05T12:00:00Z",
            "p10_f": 70.0, "p50_f": 75.0, "p90_f": 80.0,
            "lineage_id": "fixed",
        }]),
        probabilities=pd.DataFrame([{
            "snapshot_id": "s", "condition_id": "condition",
            "yes_probability": 0.90,
        }]),
        lineages=pd.DataFrame([{
            "lineage_id": "fixed", "checkpoint_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "split_role": "post_test_holdout", "mc_seed": 42,
            "mc_samples": 2000,
        }]),
    )
    replay = PointInTimeReplay(
        forecast_cache=cache,
        quotes=pd.DataFrame([{
            "condition_id": "condition",
            "timestamp": "2026-06-05T12:00:00Z",
            "bid_price_yes": 0.40, "ask_price_yes": 0.50,
            "estimated_spread": 0.10, "capacity_usdc": 100.0,
            "impact_per_usdc": 0.0,
        }]),
        markets=pd.DataFrame([{
            "condition_id": "condition", "event_id": "event",
            "target_date": "2026-06-05", "yes_token_id": "yes",
            "no_token_id": "no", "bucket_lo_f": 70.0,
            "bucket_hi_f": 80.0,
        }]),
        station_id="KLGA",
        strategy=ReplayStrategy.liquid(),
    )

    assert replay.tick("2026-06-05T12:00:00Z").empty


def test_near_term_strategy_skips_market_more_than_24_hours_away() -> None:
    # Use the same compact evidence shape; only the event horizon differs.
    cache = ForecastCache(
        snapshots=pd.DataFrame([{
            "snapshot_id": "s", "station_id": "KLGA",
            "init_time": "2026-06-05T11:00:00Z",
            "target_date": "2026-06-07",
            "available_at": "2026-06-05T12:00:00Z",
            "p10_f": 70.0, "p50_f": 75.0, "p90_f": 80.0,
            "lineage_id": "fixed",
        }]),
        probabilities=pd.DataFrame([{
            "snapshot_id": "s", "condition_id": "condition",
            "yes_probability": 0.90,
        }]),
        lineages=pd.DataFrame([{
            "lineage_id": "fixed", "checkpoint_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "split_role": "post_test_holdout", "mc_seed": 42,
            "mc_samples": 2000,
        }]),
    )
    replay = PointInTimeReplay(
        forecast_cache=cache,
        quotes=pd.DataFrame([{
            "condition_id": "condition",
            "timestamp": "2026-06-05T12:00:00Z",
            "bid_price_yes": 0.49, "ask_price_yes": 0.50,
            "estimated_spread": 0.01, "capacity_usdc": 100.0,
            "impact_per_usdc": 0.0,
        }]),
        markets=pd.DataFrame([{
            "condition_id": "condition", "event_id": "event",
            "target_date": "2026-06-07", "yes_token_id": "yes",
            "no_token_id": "no", "bucket_lo_f": 70.0,
            "bucket_hi_f": 80.0,
            "market_end_at": "2026-06-07T04:00:00Z",
        }]),
        station_id="KLGA",
        strategy=ReplayStrategy.near_term(),
    )

    assert replay.tick("2026-06-05T12:00:00Z").empty


def test_run_replays_ticks_then_settles_event_without_trading_after_close() -> None:
    cache = ForecastCache(
        snapshots=pd.DataFrame([{
            "snapshot_id": "s", "station_id": "KLGA",
            "init_time": "2026-06-05T11:00:00Z",
            "target_date": "2026-06-05",
            "available_at": "2026-06-05T12:05:00Z",
            "p10_f": 70.0, "p50_f": 75.0, "p90_f": 80.0,
            "lineage_id": "fixed",
        }]),
        probabilities=pd.DataFrame([{
            "snapshot_id": "s", "condition_id": "condition",
            "yes_probability": 0.80,
        }]),
        lineages=pd.DataFrame([{
            "lineage_id": "fixed", "checkpoint_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "split_role": "post_test_holdout", "mc_seed": 42,
            "mc_samples": 2000,
        }]),
    )
    quotes = pd.DataFrame([{
        "condition_id": "condition", "timestamp": tick,
        "bid_price_yes": 0.49, "ask_price_yes": 0.50,
        "estimated_spread": 0.01, "capacity_usdc": 100.0,
        "impact_per_usdc": 0.0,
    } for tick in pd.date_range(
        "2026-06-05T12:00:00Z", periods=3, freq="5min"
    )])
    replay = PointInTimeReplay(
        forecast_cache=cache,
        quotes=quotes,
        markets=pd.DataFrame([{
            "condition_id": "condition", "event_id": "event",
            "target_date": "2026-06-05", "yes_token_id": "yes",
            "no_token_id": "no", "bucket_lo_f": 70.0,
            "bucket_hi_f": 80.0,
            "market_end_at": "2026-06-05T12:10:00Z",
        }]),
        station_id="KLGA",
    )

    result = replay.run(actual_highs={"2026-06-05": 75.0})

    assert len(result.trades) == 1
    assert len(result.settlements) == 1
    assert result.ending_cash_usd == 550.0
    assert result.equity_curve.iloc[-1]["equity_usd"] == 550.0


def test_run_returns_typed_empty_frames_when_no_markets_or_quotes() -> None:
    cache = ForecastCache(
        snapshots=pd.DataFrame(
            columns=[
                "snapshot_id", "station_id", "init_time", "target_date",
                "available_at", "p10_f", "p50_f", "p90_f", "lineage_id",
            ]
        ),
        probabilities=pd.DataFrame(
            columns=["snapshot_id", "condition_id", "yes_probability"]
        ),
        lineages=pd.DataFrame(
            columns=[
                "lineage_id", "checkpoint_sha256", "calibration_sha256",
                "split_role", "mc_seed", "mc_samples",
            ]
        ),
    )
    replay = PointInTimeReplay(
        forecast_cache=cache,
        quotes=pd.DataFrame(
            columns=[
                "condition_id", "timestamp", "bid_price_yes",
                "ask_price_yes", "estimated_spread", "capacity_usdc",
                "impact_per_usdc",
            ]
        ),
        markets=pd.DataFrame(
            columns=[
                "condition_id", "event_id", "target_date",
                "yes_token_id", "no_token_id", "bucket_lo_f",
                "bucket_hi_f", "market_end_at",
            ]
        ),
        station_id="KLGA",
    )

    result = replay.run(actual_highs={})

    assert result.trades.empty
    assert result.settlements.empty
    assert result.decisions.empty
    assert result.equity_curve.empty
    assert result.equity_curve.columns.tolist() == [
        "timestamp",
        "cash_usd",
        "positions_value_usd",
        "equity_usd",
        "gross_exposure_usd",
    ]
