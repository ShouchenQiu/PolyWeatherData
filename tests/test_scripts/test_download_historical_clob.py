from __future__ import annotations

import pandas as pd

from data.clob.history import HistoricalClobDataset
from scripts.download_historical_clob import main


class _Source:
    def fetch(self, **kwargs) -> HistoricalClobDataset:
        assert kwargs["start_date"] == "2026-04-01"
        assert kwargs["end_date"] == "2026-04-02"
        assert kwargs["station"].station_id == "KLGA"
        return HistoricalClobDataset(
            events=pd.DataFrame(
                columns=[
                    "event_id",
                    "event_slug",
                    "event_title",
                    "target_date",
                    "event_end_at",
                    "station_id",
                ]
            ),
            markets=pd.DataFrame(
                columns=[
                    "event_id",
                    "market_id",
                    "condition_id",
                    "market_slug",
                    "yes_token_id",
                    "no_token_id",
                    "bucket_lo_f",
                    "bucket_hi_f",
                    "fees_enabled",
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
            start_date="2026-04-01",
            end_date="2026-04-02",
            station_id="KLGA",
        )


def test_download_historical_clob_cli_writes_artifact(tmp_path) -> None:
    output = tmp_path / "artifact"

    result = main(
        [
            "--start",
            "2026-04-01",
            "--end",
            "2026-04-02",
            "--station",
            "KLGA",
            "--output",
            str(output),
        ],
        source=_Source(),
    )

    assert result == 0
    assert (output / "manifest.json").exists()
    assert (output / "coverage.parquet").exists()
