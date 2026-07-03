from __future__ import annotations

import json

import pandas as pd

from data.clob.history import HistoricalClobDataset
from scripts.audit_historical_clob import main


def test_audit_cli_persists_exclusions_and_source_hash(tmp_path) -> None:
    clob_root = tmp_path / "clob"
    HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event", "target_date": "2026-04-01"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "condition_id": "condition",
                    "yes_token_id": "yes",
                    "no_token_id": "no",
                }
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "condition",
                    "token_id": "yes",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "price": 0.5,
                    "size": 10.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-02",
        station_id="KLGA",
    ).save(clob_root)
    output = tmp_path / "quality.json"

    result = main(
        [
            "--clob-root",
            str(clob_root),
            "--exclude",
            "2026-04-02=provider_event_unavailable",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["status"] == "accepted_with_exclusions"
    assert report["source_manifest_sha256"]
    assert report["excluded_days"] == {
        "2026-04-02": "provider_event_unavailable"
    }
