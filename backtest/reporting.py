"""Detailed tabular reporting for stateful replay results."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.replay import ReplayResult


@dataclass(frozen=True)
class ReplayReport:
    summary: dict[str, float | int]
    by_event: pd.DataFrame
    by_day: pd.DataFrame
    by_rejection: pd.DataFrame
    trades: pd.DataFrame
    settlements: pd.DataFrame


@dataclass(frozen=True)
class ReplayArtifact:
    result: ReplayResult
    strategy: str

    def save(
        self,
        root: Path | str,
        *,
        source_sha256: dict[str, str] | None = None,
    ) -> dict:
        artifact_dir = Path(root)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        frames = {
            "trades.parquet": self.result.trades,
            "settlements.parquet": self.result.settlements,
            "equity_curve.parquet": self.result.equity_curve,
            "decisions.parquet": self.result.decisions,
        }
        hashes: dict[str, str] = {}
        for filename, frame in frames.items():
            path = artifact_dir / filename
            frame.to_parquet(path, index=False)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = build_replay_report(self.result)
        (artifact_dir / "summary.json").write_text(
            json.dumps(report.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        hashes["summary.json"] = hashlib.sha256(
            (artifact_dir / "summary.json").read_bytes()
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "data_class": "point_in_time_replay_result",
            "strategy": self.strategy,
            "starting_cash_usd": self.result.starting_cash_usd,
            "ending_cash_usd": self.result.ending_cash_usd,
            "row_counts": {
                "trades": len(self.result.trades),
                "settlements": len(self.result.settlements),
                "equity_curve": len(self.result.equity_curve),
                "decisions": len(self.result.decisions),
            },
            "source_sha256": dict(sorted((source_sha256 or {}).items())),
            "sha256": hashes,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, root: Path | str) -> "ReplayArtifact":
        artifact_dir = Path(root)
        try:
            manifest = json.loads(
                (artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid replay manifest: {exc}") from exc
        if manifest.get("schema_version") != 1:
            raise RuntimeError(
                "unsupported replay schema_version="
                f"{manifest.get('schema_version')!r}"
            )
        for filename, expected in manifest.get("sha256", {}).items():
            path = artifact_dir / filename
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise RuntimeError(f"missing replay file {filename}") from exc
            if actual != expected:
                raise RuntimeError(
                    f"replay checksum mismatch: {filename}"
                )
        return cls(
            result=ReplayResult(
                trades=pd.read_parquet(artifact_dir / "trades.parquet"),
                settlements=pd.read_parquet(
                    artifact_dir / "settlements.parquet"
                ),
                equity_curve=pd.read_parquet(
                    artifact_dir / "equity_curve.parquet"
                ),
                decisions=pd.read_parquet(
                    artifact_dir / "decisions.parquet"
                ),
                starting_cash_usd=float(manifest["starting_cash_usd"]),
                ending_cash_usd=float(manifest["ending_cash_usd"]),
            ),
            strategy=str(manifest["strategy"]),
        )


def build_replay_report(result: ReplayResult) -> ReplayReport:
    trades = result.trades.copy()
    settlements = result.settlements.copy()
    fees = float(trades.get("fee_usd", pd.Series(dtype=float)).sum())
    turnover = float(trades.get("size_usd", pd.Series(dtype=float)).sum())
    net_pnl = float(result.ending_cash_usd - result.starting_cash_usd)
    return_pct = (
        net_pnl / result.starting_cash_usd * 100.0
        if result.starting_cash_usd
        else float("nan")
    )
    realized = settlements.get(
        "realized_pnl_usd",
        pd.Series(dtype=float),
    )
    decisions = result.decisions
    rejected = (
        decisions[~decisions["approved"].astype(bool)]
        if not decisions.empty
        else pd.DataFrame()
    )
    by_rejection = (
        rejected.groupby("reason", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "reason"], ascending=[False, True])
        .reset_index(drop=True)
        if not rejected.empty
        else pd.DataFrame(columns=["reason", "count"])
    )
    if result.equity_curve.empty:
        max_drawdown = float("nan")
    else:
        equity = result.equity_curve["equity_usd"].astype(float)
        max_drawdown = float((equity.cummax() - equity).max())
    summary: dict[str, float | int] = {
        "starting_cash_usd": float(result.starting_cash_usd),
        "ending_cash_usd": float(result.ending_cash_usd),
        "net_pnl_usd": net_pnl,
        "return_pct": return_pct,
        "trade_count": len(trades),
        "buy_count": int(
            (trades.get("side", pd.Series(dtype=str)) == "BUY").sum()
        ),
        "sell_count": int(
            (trades.get("side", pd.Series(dtype=str)) == "SELL").sum()
        ),
        "turnover_usd": turnover,
        "fees_usd": fees,
        "settled_positions": len(settlements),
        "win_rate": (
            float((realized > 0).mean()) if len(realized) else float("nan")
        ),
        "average_edge": (
            float(trades["edge"].mean())
            if "edge" in trades and len(trades)
            else float("nan")
        ),
        "max_drawdown_usd": max_drawdown,
        "approved_decisions": (
            int(decisions["approved"].astype(bool).sum())
            if not decisions.empty
            else 0
        ),
        "rejected_decisions": len(rejected),
    }
    if settlements.empty:
        by_event = pd.DataFrame(
            columns=[
                "event_id",
                "positions",
                "cost_basis_usd",
                "payout_usd",
                "realized_pnl_usd",
            ]
        )
        by_day = pd.DataFrame(
            columns=["event_day", "realized_pnl_usd"]
        )
    else:
        by_event = (
            settlements.groupby("event_id", as_index=False)
            .agg(
                positions=("token_id", "count"),
                cost_basis_usd=("cost_basis_usd", "sum"),
                payout_usd=("payout_usd", "sum"),
                realized_pnl_usd=("realized_pnl_usd", "sum"),
            )
            .sort_values("event_id")
            .reset_index(drop=True)
        )
        settlement_time = pd.to_datetime(
            settlements["timestamp"],
            utc=True,
        )
        by_day = (
            settlements.assign(event_day=settlement_time.dt.date.astype(str))
            .groupby("event_day", as_index=False)["realized_pnl_usd"]
            .sum()
        )
    return ReplayReport(
        summary=summary,
        by_event=by_event,
        by_day=by_day,
        by_rejection=by_rejection,
        trades=trades,
        settlements=settlements,
    )
