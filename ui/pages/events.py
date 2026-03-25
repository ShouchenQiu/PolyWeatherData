"""Events page: event list, lifecycle status, and bucket prices."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from db.schema import DbClient


def _load_events(db: "DbClient") -> pd.DataFrame:
    cur = db._conn.execute(
        """
        SELECT condition_id, question, event_slug, event_end_date,
               station_id, status
        FROM events
        ORDER BY event_end_date DESC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_markets(db: "DbClient", condition_id: str) -> pd.DataFrame:
    cur = db._conn.execute(
        """
        SELECT token_id, market_slug, bucket_lo_f, bucket_hi_f, outcome
        FROM markets
        WHERE condition_id = ?
        ORDER BY bucket_lo_f
        """,
        (condition_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_latest_prices(db: "DbClient", condition_id: str) -> pd.DataFrame:
    """Most recent signal price per token."""
    cur = db._conn.execute(
        """
        SELECT s.token_id, s.mc_prob, s.clob_price, s.edge, s.side,
               s.timestamp
        FROM signals s
        JOIN markets m ON s.token_id = m.token_id
        WHERE m.condition_id = ?
        AND s.timestamp = (
            SELECT MAX(s2.timestamp) FROM signals s2
            WHERE s2.token_id = s.token_id
        )
        """,
        (condition_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render(db: "DbClient") -> None:
    """Render the Events page."""
    st.header("Events")

    events_df = _load_events(db)

    if events_df.empty:
        st.info("No events found.")
        return

    # -- Filter by status ------------------------------------------------
    status_opts = ["all"] + sorted(events_df["status"].unique().tolist())
    sel_status  = st.selectbox("Filter by status", status_opts)
    view = events_df if sel_status == "all" else events_df[events_df["status"] == sel_status]

    st.markdown(f"**{len(view)} event(s)** shown.")
    st.dataframe(
        view[["condition_id", "station_id", "event_end_date", "status", "question"]],
        use_container_width=True,
    )

    st.divider()

    # -- Drill-down toggle -----------------------------------------------
    st.subheader("Event Detail")
    cid = st.selectbox(
        "Select condition_id",
        options=[""] + view["condition_id"].tolist(),
    )

    if not cid:
        return

    markets_df = _load_markets(db, cid)
    prices_df  = _load_latest_prices(db, cid)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Buckets**")
        if markets_df.empty:
            st.info("No markets found.")
        else:
            st.dataframe(markets_df, use_container_width=True)

    with col2:
        st.markdown("**Latest Signal Prices**")
        if prices_df.empty:
            st.info("No signals yet.")
        else:
            st.dataframe(prices_df, use_container_width=True)
