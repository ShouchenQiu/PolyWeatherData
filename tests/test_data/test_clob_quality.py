from __future__ import annotations

import pandas as pd
import pytest

from data.clob.history import HistoricalClobDataset, HistoricalDataIncomplete
from data.clob.quality import audit_historical_clob


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
