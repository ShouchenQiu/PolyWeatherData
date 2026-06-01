"""Streamlit entry point for PolyWeatherBot UI.

Run with:
    streamlit run ui/app.py

Sidebar navigation:
    Overview          -- production monitoring summary
    Data Health       -- ingestion and forecast freshness
    Polymarket        -- events, markets, tokens, latest prices
    Signals & Orders  -- decision funnel and order lifecycle
    Forecast & Model  -- forecast snapshots, TFT quantiles, MC buckets
    PNL & Risk        -- PnL and exposure
    Backtest & Replay -- backtest result analysis
"""
from __future__ import annotations

import os

import streamlit as st

from config.settings import Settings
from db.schema import DbClient
from ui.alerts import StalenessState

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


def _load_staleness_from_db(db: DbClient) -> StalenessState:
    """Read current staleness from the shared bot_state DB table.

    The bot (TradingLoop) writes via db.set_data_stale() when it detects a
    stale forecast.  Because the bot and UI may run as separate processes,
    the DB is the canonical cross-process state store for staleness.
    """
    state = StalenessState()
    result = db.get_data_stale()
    if result is not None:
        station, occurred_at = result
        state.mark_stale(occurred_at, station)
    return state


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Overview",
    "Data Health",
    "Polymarket",
    "Signals & Orders",
    "Forecast & Model",
    "PNL & Risk",
    "Backtest & Replay",
]

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
staleness_state = _load_staleness_from_db(db)

is_paper = settings.trading_mode != "live"

if page == "Overview":
    from ui.pages import overview
    overview.render(db, staleness_state=staleness_state, is_paper=is_paper)

elif page == "Data Health":
    from ui.pages import data_health
    data_health.render(db)

elif page == "Polymarket":
    from ui.pages import polymarket
    polymarket.render(db)

elif page == "Signals & Orders":
    from ui.pages import signals_orders
    signals_orders.render(db, is_paper=is_paper)

elif page == "Forecast & Model":
    from ui.pages import forecast_model
    forecast_model.render(db)

elif page == "PNL & Risk":
    from ui.pages import pnl_risk
    pnl_risk.render(db, is_paper=is_paper)

elif page == "Backtest & Replay":
    from ui.pages import backtest_replay
    backtest_replay.render()
