"""Signals & Orders page: decision funnel and order lifecycle."""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient", is_paper: bool | None = None) -> None:
    st.header("Signals & Orders")

    signals = queries.recent_signal_decisions(db)
    if signals.empty:
        st.info("No signals recorded yet.")
    else:
        counts = signals["approved"].value_counts().rename_axis("approved").reset_index(name="count")
        st.plotly_chart(px.bar(counts, x="approved", y="count"), use_container_width=True)
        st.dataframe(signals, use_container_width=True)

    orders = queries.orders_history(db, is_paper=is_paper)
    st.subheader("Orders")
    if orders.empty:
        st.info("No orders recorded yet.")
    else:
        st.dataframe(orders, use_container_width=True)
