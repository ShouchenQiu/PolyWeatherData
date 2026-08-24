"""Streamlit entry point for PolyWeatherData UI.

Run with:
    streamlit run ui/app.py

Sidebar navigation:
    Mispricing Overview        -- calibration gap heatmap by station x period
    Calendar Stability         -- rolling-window calibration gap trend
    Cross-Station              -- leave-one-station-out generalisation
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
        "First-Layer Calibration",
        "First-Layer (Equal-Weighted)",
        "First-Layer (Aligned Dates)",
        "Event-Layer Scores (Brier/LogL/RPS)",
        "Calendar Stability",
        "Cross-Station",
    ],
)

st.title(f"PolyWeatherData — {page}")

if page == "Data Overview":
    # render the data overview page inline
    import runpy

    runpy.run_path(str(Path(__file__).parent / "pages" / "data_overview.py"), run_name="__main__")

elif page == "First-Layer Calibration":
    import runpy

    runpy.run_path(
        str(Path(__file__).parent / "pages" / "first_layer_calibration.py"),
        run_name="__main__",
    )

elif page == "First-Layer (Equal-Weighted)":
    import runpy

    runpy.run_path(
        str(Path(__file__).parent / "pages" / "first_layer_equal_weighted.py"),
        run_name="__main__",
    )

elif page == "First-Layer (Aligned Dates)":
    import runpy

    runpy.run_path(
        str(Path(__file__).parent / "pages" / "first_layer_aligned_dates.py"),
        run_name="__main__",
    )

elif page == "Event-Layer Scores (Brier/LogL/RPS)":
    import runpy

    runpy.run_path(
        str(Path(__file__).parent / "pages" / "event_layer_scores.py"),
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
