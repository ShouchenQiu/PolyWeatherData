"""Streamlit entry point for PolyWeatherBot UI.

Run with:
    streamlit run ui/app.py

Sidebar navigation:
    Dashboard  -- real-time P&L and station status
    Positions  -- order history and paper/live toggle
    Events     -- event list, lifecycle, bucket prices
    Backtest   -- interactive backtest controls
    Model      -- model versions and per-station RMSE
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from config.settings import Settings
from core.alerts import AlertBus
from db.schema import DbClient
from ui.alerts import StalenessState, register_ui_handlers

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PolyWeatherBot",
    page_icon=":partly_sunny:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared state: DbClient (cached for the session)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_db() -> DbClient:
    settings  = Settings()
    db_path   = os.environ.get(
        "POLYWEATHER_DB",
        str(settings.project_root / "polyweather.db"),
    )
    return DbClient(db_path)


@st.cache_resource
def _get_alert_bus() -> AlertBus:
    return AlertBus()


@st.cache_resource
def _get_staleness_state() -> StalenessState:
    state = StalenessState()
    register_ui_handlers(_get_alert_bus(), state)
    return state


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = ["Dashboard", "Positions", "Events", "Backtest", "Model"]

st.sidebar.title("PolyWeatherBot")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", PAGES, index=0)

settings = Settings()
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Mode:** `{settings.trading_mode.upper()}`  \n"
    f"**Min edge:** `{settings.min_edge}`  \n"
    f"**Max pos:** `${settings.max_position_usd}`"
)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

db = _get_db()
staleness_state = _get_staleness_state()

if page == "Dashboard":
    from ui.pages import dashboard
    dashboard.render(db, staleness_state=staleness_state)

elif page == "Positions":
    from ui.pages import positions
    positions.render(db, trading_mode=settings.trading_mode)

elif page == "Events":
    from ui.pages import events
    events.render(db)

elif page == "Backtest":
    from ui.pages import backtest
    backtest.render()

elif page == "Model":
    from ui.pages import model
    model.render(db)
