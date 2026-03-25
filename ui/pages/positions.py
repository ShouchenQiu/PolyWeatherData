"""Positions page: order history and paper/live mode indicator."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from db.schema import DbClient


def _load_full_history(db: "DbClient") -> pd.DataFrame:
    """All orders with associated signal metadata."""
    cur = db._conn.execute(
        """
        SELECT o.order_id, o.status, o.size_usd, o.price, o.filled_at,
               s.timestamp AS signal_time, s.side, s.edge,
               s.mc_prob, s.clob_price, s.token_id
        FROM orders o
        JOIN signals s ON o.signal_id = s.signal_id
        ORDER BY s.timestamp DESC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render(db: "DbClient", trading_mode: str = "paper") -> None:
    """Render the Positions page.

    Parameters
    ----------
    db:
        DbClient instance.
    trading_mode:
        Either 'paper' or 'live'.
    """
    st.header("Positions")

    # -- Mode badge -------------------------------------------------------
    badge_color = "green" if trading_mode == "live" else "orange"
    st.markdown(
        f"**Trading mode:** :{badge_color}[{trading_mode.upper()}]"
    )
    st.divider()

    df = _load_full_history(db)

    if df.empty:
        st.info("No orders in history.")
        return

    # -- Filter controls --------------------------------------------------
    col1, col2 = st.columns(2)
    status_opts = ["all"] + sorted(df["status"].unique().tolist())
    sel_status  = col1.selectbox("Filter by status", status_opts)
    side_opts   = ["all"] + sorted(df["side"].dropna().unique().tolist())
    sel_side    = col2.selectbox("Filter by side", side_opts)

    view = df.copy()
    if sel_status != "all":
        view = view[view["status"] == sel_status]
    if sel_side != "all":
        view = view[view["side"] == sel_side]

    # -- Summary metrics --------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Shown rows",    len(view))
    c2.metric("Filled",        int((view["status"] == "filled").sum()))
    c3.metric("Total notional ($)",
              f"{view['size_usd'].sum():.2f}" if not view.empty else "0.00")

    st.divider()

    # -- Main table -------------------------------------------------------
    st.subheader("Order History")
    st.dataframe(view, use_container_width=True)
