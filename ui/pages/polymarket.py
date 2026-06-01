"""Polymarket page: Events, Markets, Tokens, and latest CLOB observations."""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient") -> None:
    st.header("Polymarket")

    events = queries.events_overview(db)
    if events.empty:
        st.info("No events found.")
        return

    statuses = ["all"] + sorted(events["status"].dropna().unique().tolist())
    status = st.selectbox("Status", statuses)
    view = events if status == "all" else queries.events_overview(db, status=status)
    st.dataframe(view, use_container_width=True)

    condition_ids = [""] + view["condition_id"].tolist()
    condition_id = st.selectbox("Event", condition_ids)
    ladder = queries.market_ladder(db, condition_id=condition_id or None)
    if ladder.empty:
        st.info("No markets found.")
        return

    chart_df = ladder.dropna(subset=["bucket_lo_f", "mid_price"])
    if not chart_df.empty:
        st.plotly_chart(
            px.bar(chart_df, x="bucket_lo_f", y="mid_price", color="outcome"),
            use_container_width=True,
        )
    st.dataframe(ladder, use_container_width=True)
