"""Streamlit entry point for the PolyWeatherBot Verification Console."""
from __future__ import annotations

import os

import streamlit as st

from config.settings import Settings
from db.schema import DbClient

st.set_page_config(
    page_title="PolyWeatherBot Console",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _get_db() -> DbClient:
    settings = Settings()
    db_path = os.environ.get(
        "POLYWEATHER_DB",
        str(settings.project_root / "polyweather.db"),
    )
    return DbClient(db_path)


PAGES = [
    "Home",
    "Data Health",
    "Training",
    "Forecast & Model",
    "MC / Backtest",
    "Polymarket",
    "Signals & Orders",
    "PNL & Risk",
]


settings = Settings()
db = _get_db()

st.sidebar.title("Verification Console")
page = st.sidebar.radio("Navigate", PAGES, index=0)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**DB:** `{os.environ.get('POLYWEATHER_DB', 'polyweather.db')}`  \n"
    f"**Data store:** `{os.environ.get('POLYWEATHER_DATA_STORE', str(settings.data_store_dir))}`"
)

if page == "Home":
    from ui.pages import console_home
    console_home.render(db)

elif page == "Data Health":
    from ui.pages import data_health
    data_health.render(db)

elif page == "Training":
    from ui.pages import console_training
    console_training.render(db, settings=settings)

elif page == "Forecast & Model":
    from ui.pages import forecast_model
    forecast_model.render(db)

elif page == "MC / Backtest":
    from ui.pages import console_backtest
    console_backtest.render(db)

elif page == "Polymarket":
    from ui.pages import polymarket
    polymarket.render(db)

elif page == "Signals & Orders":
    from ui.pages import signals_orders
    signals_orders.render(db, is_paper=settings.trading_mode != "live")

elif page == "PNL & Risk":
    from ui.pages import pnl_risk
    pnl_risk.render(db, is_paper=settings.trading_mode != "live")
