"""Forecast & Model page: TFT quantiles, MC buckets, and model snapshots."""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient") -> None:
    st.header("Forecast & Model")

    history = queries.forecast_bucket_history(db)
    if history.empty:
        st.info("No forecast snapshots recorded yet.")
        return

    condition_ids = ["all"] + sorted(history["condition_id"].dropna().unique().tolist())
    selected = st.selectbox("Event", condition_ids)
    view = history if selected == "all" else queries.forecast_bucket_history(db, condition_id=selected)

    st.plotly_chart(
        px.line(
            view.sort_values("tick_utc"),
            x="tick_utc",
            y="mc_prob",
            color="token_id",
        ),
        use_container_width=True,
    )
    latest_snapshot = int(view.sort_values("tick_utc").iloc[-1]["snapshot_id"])
    q = queries.forecast_quantiles(db, latest_snapshot)
    if not q.empty:
        st.plotly_chart(px.line(q, x="fxx", y=["p10", "p50", "p90"]), use_container_width=True)
    st.dataframe(view, use_container_width=True)
