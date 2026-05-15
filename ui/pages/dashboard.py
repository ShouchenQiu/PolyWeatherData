"""Dashboard page: real-time P&L and per-station status."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from db.schema import DbClient
    from ui.alerts import StalenessState


def _load_orders(db: "DbClient") -> pd.DataFrame:
    """Return orders joined with signal data as a DataFrame."""
    cur = db._conn.execute(
        """
        SELECT o.order_id, o.status, o.size_usd, o.price, o.filled_at,
               s.timestamp, s.mc_prob, s.clob_price, s.edge, s.side,
               s.token_id
        FROM orders o
        JOIN signals s ON o.signal_id = s.signal_id
        ORDER BY s.timestamp DESC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_pnl(df: pd.DataFrame) -> float:
    """Rough mid-price P&L across all filled orders."""
    if df.empty:
        return 0.0
    filled = df[df["status"] == "filled"].copy()
    if filled.empty:
        return 0.0
    # For YES buys: pnl = (1 - price) * size_usd if won, else (-price) * size_usd
    # We approximate with edge * size_usd as a proxy when outcome unknown.
    return float((filled["edge"] * filled["size_usd"]).sum())


def _load_station_summary(db: "DbClient") -> pd.DataFrame:
    """Active events grouped by station."""
    cur = db._conn.execute(
        """
        SELECT e.station_id,
               COUNT(*) AS active_events,
               GROUP_CONCAT(DISTINCT e.status) AS statuses
        FROM events e
        WHERE e.status NOT IN ('resolved', 'cancelled')
        GROUP BY e.station_id
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


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
            f"Data stale since {since_label} ¡ª {station_label} forecast not updated.",
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
        cols = ["timestamp", "token_id", "side", "mc_prob", "clob_price",
                "edge", "size_usd", "status"]
        display = orders_df[[c for c in cols if c in orders_df.columns]]
        st.dataframe(display.head(50), use_container_width=True)
