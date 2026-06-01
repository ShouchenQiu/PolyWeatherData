"""Overview page: production monitoring summary."""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient
    from ui.alerts import StalenessState


def render(db: "DbClient", staleness_state: "StalenessState | None" = None, is_paper: bool | None = None) -> None:
    st.header("Overview")

    if staleness_state is not None and staleness_state.is_stale:
        station_label = staleness_state.station or "unknown station"
        since_label = (
            staleness_state.stale_since.strftime("%H:%M UTC")
            if staleness_state.stale_since is not None
            else "unknown time"
        )
        st.warning(f"Data stale since {since_label} - {station_label}")

    metrics = queries.overview_metrics(db, is_paper=is_paper)
    cols = st.columns(6)
    cols[0].metric("Active Events", metrics["active_events"])
    cols[1].metric("Tradeable Markets", metrics["tradeable_markets"])
    cols[2].metric("Open Orders", metrics["open_orders"])
    cols[3].metric("Approved", metrics["approved_signals"])
    cols[4].metric("Rejected", metrics["rejected_signals"])
    cols[5].metric("Total PnL", f"{metrics['total_pnl']:+.2f}")

    pnl_df = queries.pnl_history(db, is_paper=is_paper)
    if not pnl_df.empty:
        st.plotly_chart(
            px.line(pnl_df.sort_values("tick_utc"), x="tick_utc", y="total_pnl"),
            use_container_width=True,
        )

    recent = queries.loop_ticks(db, limit=20)
    if recent.empty:
        st.info("No loop ticks recorded yet.")
    else:
        st.dataframe(recent, use_container_width=True)
