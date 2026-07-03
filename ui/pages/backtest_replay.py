"""Point-in-time replay evidence and paper-promotion gate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backtest.reporting import ReplayArtifact, build_replay_report
from config.settings import Settings


def render(
    *,
    selection_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> None:
    settings = Settings()
    selection_dir = Path(
        selection_root
        or settings.data_store_dir
        / "backtest"
        / "selection"
        / "KLGA"
        / "development_v2"
    )
    results_dir = Path(
        results_root
        or settings.data_store_dir
        / "backtest"
        / "results"
        / "KLGA"
        / "development_v2"
    )
    st.header("Point-in-time Backtest & Strategy Selection")
    if not (selection_dir / "manifest.json").exists():
        st.warning(
            f"No strategy-selection artifact found at {selection_dir}."
        )
        return
    manifest = json.loads(
        (selection_dir / "manifest.json").read_text(encoding="utf-8")
    )
    selection = pd.read_parquet(selection_dir / "selection.parquet")
    target_range = manifest.get("target_range", ["?", "?"])
    st.caption(
        f"Development target range: {target_range[0]} → {target_range[1]}. "
        "Selection uses event-day bootstrap evidence."
    )
    selected = list(manifest.get("selected", []))
    if selected:
        st.success(
            "Paper promotion eligible: " + ", ".join(selected)
        )
    else:
        st.warning(
            "Paper promotion blocked: no strategy passed the positive "
            "90% event-day PnL lower-bound gate."
        )

    best_total = (
        float(selection["total_pnl_usd"].max())
        if not selection.empty
        else float("nan")
    )
    columns = st.columns(4)
    columns[0].metric("Strategies tested", len(selection))
    columns[1].metric("Selected", len(selected))
    columns[2].metric("Best total PnL", f"${best_total:,.2f}")
    columns[3].metric(
        "Paper status",
        "ELIGIBLE" if selected else "BLOCKED",
    )
    st.subheader("Selection evidence")
    st.dataframe(selection, use_container_width=True)
    if selection.empty:
        return

    strategy = st.selectbox(
        "Strategy evidence",
        selection["strategy"].astype(str).tolist(),
    )
    artifact_path = results_dir / strategy
    if not (artifact_path / "manifest.json").exists():
        st.warning(f"Replay artifact missing: {artifact_path}")
        return
    artifact = ReplayArtifact.load(artifact_path)
    report = build_replay_report(artifact.result)
    summary = report.summary
    st.subheader(f"{strategy}: replay summary")
    metrics = st.columns(6)
    metrics[0].metric("Net PnL", f"${summary['net_pnl_usd']:,.2f}")
    metrics[1].metric("Return", f"{summary['return_pct']:.2f}%")
    metrics[2].metric("Trades", int(summary["trade_count"]))
    metrics[3].metric("Turnover", f"${summary['turnover_usd']:,.2f}")
    metrics[4].metric("Fees", f"${summary['fees_usd']:,.2f}")
    metrics[5].metric(
        "Max drawdown",
        f"${summary['max_drawdown_usd']:,.2f}",
    )

    daily_path = selection_dir / f"{strategy}_event_day_pnl.parquet"
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        if not daily.empty:
            st.subheader("Event-day realized PnL")
            st.bar_chart(
                daily.set_index("event_day")["realized_pnl_usd"]
            )
    if not artifact.result.equity_curve.empty:
        st.subheader("Replay equity")
        equity = artifact.result.equity_curve.copy()
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        st.line_chart(equity.set_index("timestamp")["equity_usd"])
    if not report.by_rejection.empty:
        st.subheader("Decision rejection reasons")
        st.dataframe(report.by_rejection, use_container_width=True)
    with st.expander("Trades and settlements"):
        st.dataframe(artifact.result.trades, use_container_width=True)
        st.dataframe(artifact.result.settlements, use_container_width=True)
