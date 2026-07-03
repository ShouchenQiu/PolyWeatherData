from __future__ import annotations

import pandas as pd

from data.clob.history import HistoricalClobDataset
from scripts.build_synthetic_quotes import main


def test_build_synthetic_quotes_cli_writes_past_only_artifact(tmp_path) -> None:
    clob_root = tmp_path / "clob"
    HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event", "target_date": "2026-06-05"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "condition_id": "condition",
                    "yes_token_id": "yes",
                    "no_token_id": "no",
                    "min_tick_size": 0.01,
                }
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "condition",
                    "timestamp": pd.Timestamp("2026-06-05T12:01:00Z"),
                    "outcome": "YES",
                    "side": "BUY",
                    "price": 0.60,
                    "size": 10.0,
                    "usdc": 6.0,
                }
            ]
        ),
        start_date="2026-06-05",
        end_date="2026-06-05",
        station_id="KLGA",
    ).save(clob_root)
    output = tmp_path / "quotes"

    result = main(
        [
            "--clob-root",
            str(clob_root),
            "--output",
            str(output),
            "--start",
            "2026-06-05T12:00:00Z",
            "--end",
            "2026-06-05T12:10:00Z",
        ]
    )

    assert result == 0
    quotes = pd.read_parquet(output / "quotes.parquet")
    assert quotes["timestamp"].tolist() == [
        pd.Timestamp("2026-06-05T12:05:00Z"),
        pd.Timestamp("2026-06-05T12:10:00Z"),
    ]
