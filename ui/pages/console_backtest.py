"""Backtest verification page."""
from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from ui import queries
from ui.pages import backtest

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient") -> None:
    st.header("Backtest & Replay Verification")

    st.subheader("Available Evidence")
    counts = queries.table_counts(
        db,
        (
            "model_versions",
            "forecast_snapshots",
            "forecast_bucket_probs",
            "market_price_snapshots",
            "signals",
            "orders",
            "pnl_snapshots",
        ),
    )
    st.dataframe(counts, use_container_width=True)

    versions = queries.model_versions(db)
    if versions.empty:
        st.info("No model versions registered yet.")
    else:
        st.subheader("Model Versions")
        st.dataframe(versions, use_container_width=True)

    st.divider()
    backtest.render()
