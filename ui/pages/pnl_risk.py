"""PNL & Risk page: production PnL and exposure views."""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient", is_paper: bool | None = None) -> None:
    st.header("PNL & Risk")

    pnl = queries.pnl_history(db, is_paper=is_paper)
    if pnl.empty:
        st.info("No PnL snapshots recorded yet.")
    else:
        latest = pnl.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Realized", f"{latest['realized_pnl']:+.2f}")
        cols[1].metric("Unrealized", f"{latest['unrealized_pnl']:+.2f}")
        cols[2].metric("Total", f"{latest['total_pnl']:+.2f}")
        cols[3].metric("Exposure", f"{latest['gross_exposure']:.2f}")
        st.plotly_chart(
            px.line(pnl.sort_values("tick_utc"), x="tick_utc", y=["total_pnl", "gross_exposure"]),
            use_container_width=True,
        )

    orders = queries.orders_history(db, is_paper=is_paper)
    if not orders.empty:
        exposure = orders.groupby("condition_id", dropna=False)["size_usd"].sum().reset_index()
        st.plotly_chart(px.bar(exposure, x="condition_id", y="size_usd"), use_container_width=True)
