"""Point-in-time replay evidence and paper-promotion gate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backtest.reporting import ReplayArtifact, build_replay_report
from backtest.signal_slice_study import load_signal_slice_artifact
from backtest.signal_study import load_signal_study_artifact
from config.settings import Settings


def render(
    *,
    signal_root: Path | str | None = None,
    slice_root: Path | str | None = None,
    selection_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> None:
    settings = Settings()
    signal_dir = Path(
        signal_root
        or settings.data_store_dir
        / "backtest"
        / "signal"
        / "KLGA"
        / "development_full_day_v1"
    )
    slice_dir = Path(
        slice_root
        or settings.data_store_dir
        / "backtest"
        / "signal_slices"
        / "KLGA"
        / "development_full_day_v1"
    )
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
    if (signal_dir / "manifest.json").exists():
        _render_signal_study_artifact(signal_dir)
    else:
        st.caption(f"No Forecast Signal Study artifact found at {signal_dir}.")
    if (slice_dir / "manifest.json").exists():
        _render_signal_slice_artifact(slice_dir)
    else:
        st.caption(
            f"No Forecast Signal Slice Study artifact found at {slice_dir}."
        )
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
    artifact_manifest = json.loads(
        (artifact_path / "manifest.json").read_text(encoding="utf-8")
    )
    if artifact_manifest.get("data_class") == "backtest_edge_study":
        _render_edge_study_artifact(artifact_path, strategy)
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


def _render_edge_study_artifact(path: Path, strategy: str) -> None:
    summary = json.loads(
        (path / "summary.json").read_text(encoding="utf-8")
    )
    signal = summary["signal"]
    money = summary["money"]
    st.subheader(f"{strategy}: governed edge study")
    metrics = st.columns(6)
    metrics[0].metric("Net PnL", f"${money['net_pnl_usd']:,.2f}")
    metrics[1].metric("Trades", int(money["trade_count"]))
    metrics[2].metric("Event days", int(money["event_days"]))
    metrics[3].metric("Fees", f"${money['fees_usd']:,.2f}")
    metrics[4].metric(
        "Realized edge LCB",
        f"{signal['realized_edge_lcb_90']:.3f}",
    )
    metrics[5].metric(
        "Paper status",
        "ELIGIBLE" if summary["paper_eligible"] else "BLOCKED",
    )
    st.caption(
        "Signal quality is reported independently from fixed-risk money "
        "outcomes. Gate reasons: "
        + (", ".join(summary["gate_reasons"]) or "none")
    )
    by_stratum = pd.read_parquet(path / "by_stratum.parquet")
    trades = pd.read_parquet(path / "trades.parquet")
    settlements = pd.read_parquet(path / "settlements.parquet")
    decisions = pd.read_parquet(path / "decisions.parquet")
    st.subheader("Signal calibration by stratum")
    st.dataframe(by_stratum, use_container_width=True)
    if not settlements.empty:
        daily = settlements.groupby("target_date", as_index=False)[
            "realized_pnl_usd"
        ].sum()
        st.subheader("Event-day realized PnL")
        st.bar_chart(daily.set_index("target_date")["realized_pnl_usd"])
    if not decisions.empty:
        st.subheader("Decision evidence")
        st.dataframe(decisions, use_container_width=True)
    with st.expander("Trades and settlements"):
        st.dataframe(trades, use_container_width=True)
        st.dataframe(settlements, use_container_width=True)


def _render_signal_study_artifact(path: Path) -> None:
    try:
        artifact = load_signal_study_artifact(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Forecast Signal Study artifact is invalid: {exc}")
        return
    summary = artifact.summary
    sample = summary["sample"]
    st.subheader("Signal Quality: model vs normalized midpoint")
    metrics = st.columns(4)
    metrics[0].metric("Events", int(sample["event_count"]))
    metrics[1].metric("5-minute vectors", int(sample["vector_count"]))
    metrics[2].metric("Buckets", int(sample["bucket_count"]))
    metrics[3].metric(
        "Midpoint overround",
        f"{summary['midpoint']['mean_overround']:.3f}",
    )
    st.caption(
        "Strategy-free OOF development evidence. Lower is better for "
        "Brier, Log Score, RPS, ECE and MCE; sharpness is descriptive only."
    )
    source_table = (
        pd.DataFrame.from_dict(summary["sources"], orient="index")
        .rename_axis("source")
        .reset_index()
    )
    comparison_table = (
        pd.DataFrame.from_dict(summary["comparison"], orient="index")
        .rename_axis("metric")
        .reset_index()
    )
    st.dataframe(source_table, use_container_width=True)
    st.subheader("Paired event-day comparison")
    st.dataframe(comparison_table, use_container_width=True)
    st.subheader("Calibration reliability")
    st.dataframe(artifact.reliability, use_container_width=True)
    st.subheader("Signal scores by lead band")
    st.dataframe(artifact.by_lead, use_container_width=True)


def _render_signal_slice_artifact(path: Path) -> None:
    try:
        artifact = load_signal_slice_artifact(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Forecast Signal Slice Study artifact is invalid: {exc}")
        return
    summary = artifact.summary
    sample = summary["sample"]
    tier_counts = summary["shortlist"]["tier_counts"]
    st.subheader("Advantage Slices: development evidence")
    metrics = st.columns(4)
    metrics[0].metric("Events", int(sample["event_count"]))
    metrics[1].metric(
        "Fixed slices", int(sample["fixed_slice_count"])
    )
    metrics[2].metric(
        "Cross-fit rules", int(sample["crossfit_rule_count"])
    )
    metrics[3].metric(
        "Stable candidates", int(tier_counts["stable_candidate"])
    )
    st.warning(
        "Theoretical midpoint excess is non-executable development evidence. "
        "It excludes VWAP, fees, capacity and fill risk."
    )
    st.subheader("Nested cross-fit shortlist")
    st.dataframe(artifact.shortlist, use_container_width=True)
    st.subheader("Fixed slice atlas")
    st.dataframe(artifact.slice_atlas, use_container_width=True)
    heatmap_rows = artifact.slice_atlas[
        artifact.slice_atlas["dimensions"] == "lead_band|tv_band"
    ].copy()
    if not heatmap_rows.empty:
        split_values = heatmap_rows["values"].str.split(
            "|", n=1, expand=True
        )
        heatmap_rows["lead_band"] = split_values[0]
        heatmap_rows["tv_band"] = split_values[1]
        heatmap = heatmap_rows.pivot(
            index="lead_band",
            columns="tv_band",
            values="brier_delta",
        )
        st.subheader("Lead × disagreement Brier delta")
        st.dataframe(heatmap, use_container_width=True)
    st.subheader("Cross-fit fold stability")
    st.dataframe(artifact.crossfit_results, use_container_width=True)
