from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtest.synthetic_quotes import (
    SyntheticQuoteDataset,
    build_synthetic_book,
    build_synthetic_quotes,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition_id": "c1",
                "timestamp": "2026-06-05T12:01:00Z",
                "outcome": "YES",
                "side": "BUY",
                "price": 0.60,
                "size": 100.0,
                "usdc": 60.0,
            },
            {
                "condition_id": "c1",
                "timestamp": "2026-06-05T12:04:00Z",
                "outcome": "NO",
                "side": "BUY",
                "price": 0.45,
                "size": 100.0,
                "usdc": 45.0,
            },
        ]
    )


def test_synthetic_quote_at_tick_never_changes_from_future_trade() -> None:
    trades = _trades()
    before = build_synthetic_quotes(
        trades,
        start="2026-06-05T12:05:00Z",
        end="2026-06-05T12:05:00Z",
    )
    with_future = pd.concat(
        [
            trades,
            pd.DataFrame(
                [
                    {
                        "condition_id": "c1",
                        "timestamp": "2026-06-05T12:06:00Z",
                        "outcome": "YES",
                        "side": "BUY",
                        "price": 0.99,
                        "size": 1000.0,
                        "usdc": 990.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    after = build_synthetic_quotes(
        with_future,
        start="2026-06-05T12:05:00Z",
        end="2026-06-05T12:05:00Z",
    )

    assert_frame_equal(before, after)


def test_no_trade_is_normalized_to_yes_price_and_direction() -> None:
    quotes = build_synthetic_quotes(
        _trades(),
        start="2026-06-05T12:05:00Z",
        end="2026-06-05T12:05:00Z",
    )

    row = quotes.iloc[0]
    assert row["mid_price_yes"] == 0.55
    assert row["buy_yes_trade_count"] == 1
    assert row["sell_yes_trade_count"] == 1
    assert 0.0 <= row["bid_price_yes"] <= row["mid_price_yes"]
    assert row["mid_price_yes"] <= row["ask_price_yes"] <= 1.0


def test_synthetic_quotes_do_not_backfill_or_carry_stale_prices() -> None:
    quotes = build_synthetic_quotes(
        _trades().iloc[:1],
        start="2026-06-05T11:55:00Z",
        end="2026-06-05T12:40:00Z",
        max_staleness="30min",
    )

    assert quotes["timestamp"].min() == pd.Timestamp("2026-06-05T12:05:00Z")
    assert quotes["timestamp"].max() == pd.Timestamp("2026-06-05T12:30:00Z")


def test_synthetic_quotes_report_progress_per_condition() -> None:
    calls: list[tuple[int, int, str, int]] = []
    trades = pd.concat(
        [_trades(), _trades().assign(condition_id="c2")], ignore_index=True
    )

    build_synthetic_quotes(
        trades,
        start="2026-06-05T12:05:00Z",
        end="2026-06-05T12:05:00Z",
        progress_callback=lambda done, total, condition, rows: calls.append(
            (done, total, condition, rows)
        ),
    )

    assert [(done, total, condition) for done, total, condition, _ in calls] == [
        (1, 2, "c1"), (2, 2, "c2")
    ]


def test_synthetic_quote_artifact_is_immutable_and_hash_verified(tmp_path) -> None:
    quotes = build_synthetic_quotes(
        _trades(),
        start="2026-06-05T12:05:00Z",
        end="2026-06-05T12:05:00Z",
    )
    root = tmp_path / "quotes"
    dataset = SyntheticQuoteDataset(
        quotes=quotes,
        start_at="2026-06-05T12:05:00Z",
        end_at="2026-06-05T12:05:00Z",
        decision_tick="5min",
        lookback="60min",
        max_staleness="30min",
    )

    manifest = dataset.save(root, source_sha256="a" * 64)
    assert manifest["source_sha256"] == "a" * 64
    assert len(SyntheticQuoteDataset.load(root).quotes) == 1
    with pytest.raises(FileExistsError):
        dataset.save(root, source_sha256="a" * 64)

    with (root / "quotes.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        SyntheticQuoteDataset.load(root)


def test_synthetic_no_book_is_complement_of_yes_book() -> None:
    quote = {
        "bid_price_yes": 0.54,
        "ask_price_yes": 0.58,
        "capacity_usdc": 100.0,
        "impact_per_usdc": 0.0001,
    }

    yes = build_synthetic_book(quote, outcome="YES")
    no = build_synthetic_book(quote, outcome="NO")

    assert yes["bids"][0]["price"] == 0.54
    assert yes["asks"][0]["price"] == 0.58
    assert no["bids"][0]["price"] == 0.42
    assert no["asks"][0]["price"] == 0.46
