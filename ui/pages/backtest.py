"""Backtest page: interactive backtest controls and result charts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backtest.analysis import BacktestAnalysis


def _threshold_chart(by_thr: pd.DataFrame) -> None:
    if by_thr.empty:
        st.info("No signals passed any threshold.")
        return
    st.subheader("ROI by Edge Threshold")
    chart_df = by_thr.set_index("threshold")[["ROI_mid", "ROI_real"]]
    st.line_chart(chart_df)
    st.dataframe(by_thr, use_container_width=True)


def _lead_time_chart(by_lead: pd.DataFrame) -> None:
    if by_lead.empty:
        st.info("No lead-time breakdown available.")
        return
    st.subheader("ROI by Lead Time")
    pivot = (
        by_lead[["lead_bin", "threshold", "ROI_mid"]]
        .pivot(index="lead_bin", columns="threshold", values="ROI_mid")
    )
    st.bar_chart(pivot)
    with st.expander("Raw data"):
        st.dataframe(by_lead, use_container_width=True)


def render(results_df: pd.DataFrame | None = None) -> None:
    """Render the Backtest page.

    Parameters
    ----------
    results_df:
        Pre-computed BacktestEngine result DataFrame, or None to allow
        the user to upload a CSV.
    """
    st.header("Backtest")

    # -- Data source -------------------------------------------------
    if results_df is None:
        st.markdown("Upload a backtest results CSV or run a backtest first.")
        uploaded = st.file_uploader("Upload results CSV", type=["csv"])
        if uploaded is not None:
            results_df = pd.read_csv(uploaded)
        else:
            st.stop()

    st.success(f"Loaded {len(results_df):,} rows.")

    ana = BacktestAnalysis(results_df)

    # -- Threshold slider --------------------------------------------
    st.subheader("Controls")
    min_thr = st.slider(
        "Min edge threshold for summary",
        min_value=0.00,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    # -- Summary metrics ---------------------------------------------
    summary = ana.summary(threshold=min_thr)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N total",   summary.get("N_total", 0))
    c2.metric("ROI (mid)", f"{summary.get('ROI_mid', 0):.3f}")
    c3.metric("BSS",       f"{summary.get('bss', float('nan')):.4f}")
    c4.metric("Max Drawdown", f"{summary.get('max_drawdown_mid', 0):.3f}")

    st.divider()

    # -- Threshold breakdown -----------------------------------------
    thr_range = st.multiselect(
        "Thresholds to show",
        options=[0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        default=[0.05, 0.10, 0.15],
    )
    if thr_range:
        by_thr = ana.by_threshold(thresholds=sorted(thr_range))
        _threshold_chart(by_thr)

    st.divider()

    # -- Lead-time breakdown -----------------------------------------
    by_lead = ana.by_lead_time(thresholds=[min_thr])
    _lead_time_chart(by_lead)

    # -- Raw data -------------------------------------------------------
    with st.expander("Show raw results"):
        st.dataframe(results_df, use_container_width=True)
