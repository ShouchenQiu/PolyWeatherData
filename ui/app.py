"""Streamlit entry point for PolyWeatherData UI.

Run with:
    streamlit run ui/app.py

Sidebar navigation:
    Mispricing Overview        -- calibration gap heatmap by station x period
    Calendar Stability         -- rolling-window calibration gap trend
    Cross-Station              -- leave-one-station-out generalisation
"""
from __future__ import annotations

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
        "Mispricing Overview",
        "Calendar Stability",
        "Cross-Station",
    ],
)

st.title(f"PolyWeatherData — {page}")

if page == "Mispricing Overview":
    st.markdown(
        """
        ### Calibration Gap by Station x Period

        **calibration gap** = E[settlement_truth] - E[market_probability]

        - Positive gap -> market underpricing
        - Negative gap -> market overpricing

        Reported with bootstrap 95% CI per stratum.
        """
    )
    st.info("Dashboard awaiting data. Pull CLOB snapshots to populate charts.")

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
