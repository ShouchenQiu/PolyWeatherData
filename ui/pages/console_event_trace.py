"""Event trace page for the Verification Console."""
from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient") -> None:
    st.header("Event Trace")

    events = queries.events_overview(db)
    if events.empty:
        st.info("No events found.")
        return

    active = events[events["status"] == "active"]
    view = active if not active.empty else events
    condition_ids = view["condition_id"].tolist()
    condition_id = st.selectbox("Event", condition_ids)
    event_row = view[view["condition_id"] == condition_id].head(1)
    st.subheader("Event")
    st.dataframe(event_row, use_container_width=True)

    ladder = queries.market_ladder(db, condition_id=condition_id).drop(
        columns=["token_id"],
        errors="ignore",
    )
    if ladder.empty:
        st.info("No markets found for this event.")
        return

    st.subheader("Markets and Latest CLOB")
    st.dataframe(ladder, use_container_width=True)
    if ladder["price_tick_utc"].isna().all():
        st.info("No CLOB snapshots recorded yet.")
