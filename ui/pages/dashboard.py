"""Dashboard page: real-time P&L and per-station status."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from db.schema import DbClient
    from ui.alerts import StalenessState


def _load_orders(db: "DbClient") -> pd.DataFrame:
    """Return orders as a DataFrame."""
    rows = db.query(
        """
        SELECT order_id, token_id, condition_id, side, size_usd, price,
               fill_price, status, created_at, filled_at, settled_payout
        FROM orders
        ORDER BY created_at DESC
        """
    )
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _compute_pnl(df: pd.DataFrame) -> float:
    """Net P&L across all settled orders: sum of (payout - cost)."""
    if df.empty:
        return 0.0
    settled = df[df["status"] == "settled"].copy()
    if settled.empty:
        return 0.0
    payouts = settled["settled_payout"].fillna(0.0)
    costs   = settled["size_usd"].fillna(0.0)
    return float((payouts - costs).sum())


def _load_station_summary(db: "DbClient") -> pd.DataFrame:
    """Active events grouped by station."""
    rows = db.query(
        """
        SELECT e.station_id,
               COUNT(*) AS active_events,
               GROUP_CONCAT(DISTINCT e.status) AS statuses
        FROM events e
        WHERE e.status NOT IN ('resolved', 'cancelled')
        GROUP BY e.station_id
        """
    )
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def render(db: "DbClient", staleness_state: "StalenessState | None" = None) -> None:
    """Render the Dashboard page.

    Parameters
    ----------
    db:
        DbClient for loading orders and event summary.
    staleness_state:
        Optional shared staleness tracker.  When provided, a warning banner
        is shown whenever the bot has reported stale forecast data.
    """
    st.header("Dashboard")

    if staleness_state is not None and staleness_state.is_stale:
        station_label = staleness_state.station or "unknown station"
        since_label   = (
            staleness_state.stale_since.strftime("%H:%M UTC")
            if staleness_state.stale_since is not None
            else "unknown time"
        )
        st.warning(
            f"Data stale since {since_label} -- {station_label} forecast not updated.",
            icon="warning",
        )

    orders_df  = _load_orders(db)
    station_df = _load_station_summary(db)

    # -- KPI row ----------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    total_orders  = len(orders_df)
    open_orders   = int((orders_df["status"] == "pending").sum()) if not orders_df.empty else 0
    approx_pnl    = _compute_pnl(orders_df)

    col1.metric("Total Orders",  total_orders)
    col2.metric("Open Orders",   open_orders)
    col3.metric("Approx P&L ($)", f"{approx_pnl:+.2f}")

    st.divider()

    # -- Station status table --------------------------------------------
    st.subheader("Active Events by Station")
    if station_df.empty:
        st.info("No active events found.")
    else:
        st.dataframe(station_df, use_container_width=True)

    # -- Recent signals ---------------------------------------------------
    st.subheader("Recent Signals")
    if orders_df.empty:
        st.info("No signals recorded yet.")
    else:
        cols = ["order_id", "token_id", "side", "size_usd", "price",
                "status", "created_at", "filled_at", "settled_payout"]
        display = orders_df[[c for c in cols if c in orders_df.columns]]
        st.dataframe(display.head(50), use_container_width=True)
