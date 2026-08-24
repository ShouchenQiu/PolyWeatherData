"""First-Layer Calibration page (ICDM paper).

Renders, from docs/icdm/output/first_layer_panel.parquet:
  - Reliability diagrams per horizon (48h/24h/1h default, all selectable)
  - D_h(0.10) long-shot bias curve across all 6 horizons
  - Per-node summary table

Data source stays in docs/icdm/output (paper-only directory); this page
only reads it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

PANEL = Path(r"d:\NN\PolyWeatherData\docs\icdm\output\first_layer_panel_argmax_30min.parquet")
HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
HOR_COLS = [f"p_{int(h)}h" for h in HORIZONS]

st.title(":microscope: First-Layer: Binary Contract Calibration")

st.caption(
    "Prices = mean of ALL trades within +/-30min of each horizon point, "
    "measured BEFORE the day's high temperature occurred (ex-ante)."
)

if not PANEL.exists():
    st.error(
        "Panel not found. Run docs/icdm/code/02_build_first_layer_panel.py first."
    )
    st.stop()

panel = pd.read_parquet(PANEL)
st.caption(
    f"Source: {PANEL} | "
    f"{panel['event_id'].nunique()} events | {len(panel)} event-bucket rows | "
    f"{panel['city'].nunique()} nodes"
)


# Fine bins at low prices (long-shot), coarse bins at high prices.
BINS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.01]
BIN_LABELS = [
    "0-1%", "1-2%", "2-5%", "5-10%", "10-20%",
    "20-30%", "30-50%", "50-70%", "70-90%", "90-100%",
]


def reliability_points(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    """Event-averaged reliability: bin by price, average within event, then across events."""
    df = df[df[pcol].notna()].copy()
    df["bin"] = pd.cut(
        df[pcol],
        bins=BINS,
        labels=BIN_LABELS,
        include_lowest=True,
    )
    # per (event, bin): mean price and mean outcome
    ev = df.groupby(["event_id", "bin"], observed=True).agg(
        p=(pcol, "mean"),
        y=("y", "mean"),
    ).reset_index()
    # across events
    agg = ev.groupby("bin", observed=True).agg(
        mean_p=("p", "mean"),
        mean_y=("y", "mean"),
        n_events=("event_id", "nunique"),
    ).reset_index()
    return agg


st.subheader("Reliability diagram")
sel = st.multiselect(
    "Horizons", HORIZONS, default=[48.0, 24.0, 12.0, 6.0, 1.0],
    format_func=lambda h: f"{int(h)}h",
)
if sel:
    n_sel = len(sel)
    ncols = 2 if n_sel > 2 else 1
    nrows = (n_sel + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"{int(h)}h before high" for h in sel],
        shared_yaxes=False,
    )
    for i, h in enumerate(sel):
        row = i // ncols + 1
        col = i % ncols + 1
        col_name = f"p_{int(h)}h"
        rp = reliability_points(panel, col_name)
        se_y = np.sqrt(rp["mean_y"] * (1 - rp["mean_y"]) / rp["n_events"])
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=rp["mean_p"], y=rp["mean_y"],
            mode="lines+markers", name=f"{int(h)}h",
            error_y=dict(
                type="data", array=se_y, visible=True,
                thickness=1.0, width=3,
            ),
            customdata=np.stack([rp["n_events"]], axis=-1),
            hovertemplate=(
                "price ~%{x:.3f}<br>freq %{y:.3f}<br>n=%{customdata[0]:,}"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=col)
        fig.update_xaxes(range=[0, 1], row=row, col=col, title="mean price")
        fig.update_yaxes(range=[0, 1], row=row, col=col, title="win freq")
    fig.update_layout(
        height=280 * nrows,
        showlegend=False,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each panel = one horizon. Error bars = binomial SE of observed "
        "frequency (per event count). Hover a point to see n."
    )

    with st.expander("Reliability table"):
        for h in sel:
            col = f"p_{int(h)}h"
            rp = reliability_points(panel, col)
            st.markdown(f"**{int(h)}h horizon**")
            st.dataframe(
                rp.rename(columns={"bin": "Price bin", "mean_p": "Mean price",
                                   "mean_y": "Observed freq", "n_events": "Events"}),
                use_container_width=True, hide_index=True,
            )

    # ---- Relative bias plot: reveals strong relative deviations at tiny prices ----
    st.subheader("Relative bias: (observed - price) / price")
    st.caption(
        "Shows percentage over/under-pricing per bin. Low-price bins can "
        "have huge relative bias (e.g. 0.44% quoted vs 0.05% realised = "
        "-88%) even though absolute differences are tiny."
    )
    fig_rb = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, h in enumerate(sel):
        col_name = f"p_{int(h)}h"
        rp = reliability_points(panel, col_name)
        # relative bias in percent; cap display at +/-100 for readability
        rel = (rp["mean_y"] - rp["mean_p"]) / rp["mean_p"] * 100
        # x position: bin index (categorical), keep fine bins spaced
        fig_rb.add_trace(go.Scatter(
            x=rp["bin"], y=rel,
            mode="lines+markers", name=f"{int(h)}h",
            line=dict(color=colors[i % len(colors)]),
            customdata=np.stack([rp["n_events"], rp["mean_p"], rp["mean_y"]], axis=-1),
            hovertemplate=(
                "bin %{x}<br>rel bias %{y:.1f}%<br>"
                "quoted %{customdata[1]:.4f} | observed %{customdata[2]:.4f} | "
                "n=%{customdata[0]:,}<extra></extra>"
            ),
        ))
    fig_rb.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rb.update_layout(
        height=480,
        xaxis_title="Price bin",
        yaxis_title="(observed - quoted) / quoted  (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_rb, use_container_width=True)


st.subheader("Long-shot bias $D_h(0.10)$ across horizons")
THRESH = 0.10
dh_rows = []
for h in HORIZONS:
    col = f"p_{int(h)}h"
    sub = panel[panel[col].notna() & (panel[col] <= THRESH)]
    if len(sub) == 0:
        dh_rows.append({"horizon_h": int(h), "D_h": np.nan,
                        "n_conditions": 0, "n_events": 0})
        continue
    # event-average then cross-event average
    ev = sub.groupby("event_id").agg(
        mean_p=(col, "mean"),
        mean_y=("y", "mean"),
    )
    dh = float(ev["mean_y"].mean() - ev["mean_p"].mean())
    dh_rows.append({
        "horizon_h": int(h),
        "D_h": dh,
        "n_conditions": int(len(sub)),
        "n_events": int(len(ev)),
    })

dh = pd.DataFrame(dh_rows)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=dh["horizon_h"], y=dh["D_h"], mode="lines+markers",
    customdata=dh[["n_conditions", "n_events"]],
    hovertemplate="h=%{x:.0f}h<br>D_h=%{y:+.4f}<br>conditions=%{customdata[0]}<br>events=%{customdata[1]}<extra></extra>",
))
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(
    height=420,
    xaxis_title="Hours before settlement h",
    yaxis_title="D_h(0.10) = mean(y - p), p<=0.10",
    xaxis=dict(tickmode="array", tickvals=HORIZONS,
               ticktext=[f"{int(h)}h" for h in HORIZONS]),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "**D_h < 0** means low-price contracts win less often than quoted "
    "(long-shot overpricing)."
)

st.subheader("Per-node summary")
summary = panel.groupby("city").agg(
    events=("event_id", "nunique"),
    conditions=("condition_id", "nunique"),
    winners=("y", "sum"),
)
summary["buckets_per_event"] = (summary["conditions"] / summary["events"]).round(1)
st.dataframe(summary, use_container_width=True)

# per-node D_h table
st.subheader("Per-node long-shot bias (p <= 10%)")
node_dh_rows = []
node_ne_rows = []
for city, sub in panel.groupby("city"):
    for h in HORIZONS:
        col = f"p_{int(h)}h"
        s2 = sub[sub[col].notna() & (sub[col] <= THRESH)]
        if len(s2) == 0:
            node_dh_rows.append({"city": city, "horizon_h": int(h), "D_h": np.nan})
            node_ne_rows.append({"city": city, "horizon_h": int(h), "n_events": 0})
            continue
        ev = s2.groupby("event_id").agg(
            mp=(col, "mean"), my=("y", "mean")
        )
        node_dh_rows.append({
            "city": city, "horizon_h": int(h),
            "D_h": float(ev["my"].mean() - ev["mp"].mean()),
        })
        node_ne_rows.append({
            "city": city, "horizon_h": int(h),
            "n_events": int(len(ev)),
        })
node_dh = pd.DataFrame(node_dh_rows)
piv = node_dh.pivot(index="city", columns="horizon_h", values="D_h")
piv.columns = [f"{int(c)}h" for c in piv.columns]
node_ne = pd.DataFrame(node_ne_rows)
piv_ne = node_ne.pivot(index="city", columns="horizon_h", values="n_events")
piv_ne.columns = [f"{int(c)}h" for c in piv_ne.columns]

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**D_h(0.10)**")
    st.dataframe(piv.round(4), use_container_width=True)
with col_r:
    st.markdown("**Events per node-horizon** (small = unreliable)")
    st.dataframe(piv_ne.astype("Int64"), use_container_width=True)
st.caption(
    "60h/72h horizons were dropped: coverage too sparse "
    "(median market lead time ~49h)."
)
