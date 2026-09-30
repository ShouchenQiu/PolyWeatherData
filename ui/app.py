"""Streamlit entry point for PolyWeatherData UI.

Run with:
    streamlit run ui/app.py

Sidebar navigation (each item maps to a file in ui/pages/):
    Data Overview           -- per-station CLOB artifact status
    First-Layer (Aligned Dates) -- D_h by horizon + reliability diagram
    Event-Layer Scores      -- Brier / Log Loss / RPS by horizon
    Strategy Backtest       -- favorite / longshot buy-NO results

    Calendar Stability      -- placeholder, awaiting a rolling-window module
    Cross-Station           -- placeholder, awaiting a leave-one-out module

Every non-placeholder entry below resolves to a real file under ui/pages/;
keep this list and ui/pages/ in sync when adding or removing a page.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="PolyWeatherData",
    page_icon=":partly_sunny:",
    layout="wide",
)

st.sidebar.title("PolyWeatherData")
page = st.sidebar.radio(
    "Navigation",
    [
        "Data Overview",
        "First-Layer (Aligned Dates)",
        "Event-Layer Scores (Brier/LogL/RPS)",
        "Strategy Backtest (Favorite/Longshot)",
        "Calendar Stability",
        "Cross-Station",
    ],
)

st.title(f"PolyWeatherData — {page}")

# page label -> file under ui/pages/
PAGE_FILES = {
    "Data Overview": "data_overview.py",
    "First-Layer (Aligned Dates)": "first_layer_aligned_dates.py",
    "Event-Layer Scores (Brier/LogL/RPS)": "event_layer_scores.py",
    "Strategy Backtest (Favorite/Longshot)": "strategy_backtest.py",
}

if page in PAGE_FILES:
    import runpy

    runpy.run_path(
        str(Path(__file__).parent / "pages" / PAGE_FILES[page]),
        run_name="__main__",
    )

elif page == "Calendar Stability":
    st.markdown(
        """
        ### Rolling-Window Calibration Gap

        30-day rolling window of the calibration gap, showing how
        market mispricing evolves over time.
        """
    )
    st.info("Dashboard awaiting data. Pull CLOB snapshots to populate charts.")

elif page == "Cross-Station":
    st.markdown(
        """
        ### Leave-One-Station-Out Generalisation

        Calibration gap measured on held-out stations (train on 4,
        test on 1, repeat for all 5). A significant gap on held-out
        stations indicates the mispricing pattern generalises.
        """
    )
    st.info("Dashboard awaiting data. Pull CLOB snapshots to populate charts.")
