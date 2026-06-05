"""Verification Console home page."""
from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


_STAGE_TABLES = (
    "stations",
    "events",
    "markets",
    "forecast_snapshots",
    "forecast_bucket_probs",
    "signals",
    "orders",
    "pnl_snapshots",
    "model_versions",
    "redeem_tasks",
)


def render(db: "DbClient") -> None:
    st.header("Verification Console")

    counts = queries.table_counts(db, _STAGE_TABLES)
    st.subheader("Stage Evidence")
    st.dataframe(counts, use_container_width=True)

    metrics = queries.overview_metrics(db, is_paper=None)
    cols = st.columns(4)
    cols[0].metric("Events", metrics["active_events"])
    cols[1].metric("Tradeable Tokens", metrics["tradeable_markets"])
    cols[2].metric("Signals", metrics["approved_signals"] + metrics["rejected_signals"])
    cols[3].metric("Open Orders", metrics["open_orders"])

    ticks = queries.loop_ticks(db, limit=20)
    st.subheader("Recent Loop Ticks")
    if ticks.empty:
        st.info("No loop ticks recorded yet.")
    else:
        st.dataframe(ticks, use_container_width=True)
