from __future__ import annotations

import pandas as pd
import pytest

from backtest.replay import ReplayResult
from backtest.reporting import ReplayArtifact, build_replay_report


def test_replay_report_exposes_cash_pnl_fees_turnover_and_event_breakdown() -> None:
    result = ReplayResult(
        trades=pd.DataFrame(
            [
                {
                    "timestamp": "2026-06-05T12:05:00Z",
                    "condition_id": "condition",
                    "token_id": "yes",
                    "side": "BUY",
                    "size_usd": 50.0,
                    "shares": 100.0,
                    "price": 0.50,
                    "edge": 0.20,
                    "fee_usd": 1.25,
                    "cash_after_usd": 448.75,
                }
            ]
        ),
        settlements=pd.DataFrame(
            [
                {
                    "timestamp": "2026-06-06T04:00:00Z",
                    "event_id": "event",
                    "condition_id": "condition",
                    "token_id": "yes",
                    "shares": 100.0,
                    "payout_usd": 100.0,
                    "cost_basis_usd": 50.0,
                    "realized_pnl_usd": 50.0,
                    "cash_after_usd": 548.75,
                }
            ]
        ),
        starting_cash_usd=500.0,
        ending_cash_usd=548.75,
        equity_curve=pd.DataFrame(
            [
                {"timestamp": "2026-06-05T12:00:00Z", "equity_usd": 500.0},
                {"timestamp": "2026-06-05T12:05:00Z", "equity_usd": 480.0},
                {"timestamp": "2026-06-06T04:00:00Z", "equity_usd": 548.75},
            ]
        ),
        decisions=pd.DataFrame(
            [
                {"approved": True, "reason": None},
                {"approved": False, "reason": "insufficient_depth"},
            ]
        ),
    )

    report = build_replay_report(result)

    assert report.summary["net_pnl_usd"] == 48.75
    assert report.summary["return_pct"] == 9.75
    assert report.summary["fees_usd"] == 1.25
    assert report.summary["turnover_usd"] == 50.0
    assert report.summary["max_drawdown_usd"] == 20.0
    assert report.summary["rejected_decisions"] == 1
    assert report.by_rejection.iloc[0].to_dict() == {
        "reason": "insufficient_depth",
        "count": 1,
    }
    assert report.by_event.iloc[0]["realized_pnl_usd"] == 50.0


def test_replay_artifact_is_immutable_and_checksum_verified(tmp_path) -> None:
    result = ReplayResult(
        trades=pd.DataFrame(),
        settlements=pd.DataFrame(),
        starting_cash_usd=500.0,
        ending_cash_usd=500.0,
    )
    root = tmp_path / "replay"

    ReplayArtifact(result=result, strategy="baseline").save(
        root,
        source_sha256={"forecast": "a" * 64, "quotes": "b" * 64},
    )

    loaded = ReplayArtifact.load(root)
    assert loaded.strategy == "baseline"
    assert loaded.result.ending_cash_usd == 500.0
    with pytest.raises(FileExistsError):
        ReplayArtifact(result=result, strategy="baseline").save(root)

    with (root / "trades.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ReplayArtifact.load(root)
