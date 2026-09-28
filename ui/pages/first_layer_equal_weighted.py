"""First-Layer Calibration - Station-Equally-Weighted view.

Same data as first_layer_calibration.py but every station gets ONE vote:
compute D_h(0.10) per station first, then average across stations. This
removes the sample-size bias (e.g. London's 562 events dominating).

Also shows per-station D_h trajectories so the reader can see dispersion.
Dense horizon grid (48h..1h) makes the trajectory shape visible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PANEL = Path(r"d:\NN\PolyWeatherData\docs\icdm\output\first_layer_panel_argmax_30min.parquet")
HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
THRESH = 0.10

st.title(":balance_scale: First-Layer (Station-Equal-Weighted)")

if not PANEL.exists():
    st.error("Panel not found.")
    st.stop()

panel = pd.read_parquet(PANEL)
st.caption(
    "Every station is weighted equally (one vote per station), regardless "
    "of event count. Complements the pooled/event-weighted view."
)


def station_dh(sub: pd.DataFrame, col: str) -> tuple[float | None, int, int]:
    """Return (D_h, n_events, n_buckets) for a station at one horizon.

    Buckets = event-condition rows with price <= THRESH; events = distinct
    event_ids among them. Requires at least 20 buckets, else (None, 0, 0).
    """
    s2 = sub[(sub[col].notna()) & (sub[col] <= THRESH)]
    if len(s2) < 20:
        return None, 0, 0
    ev = s2.groupby("event_id").agg(mp=(col, "mean"), my=("y", "mean"))
    return float(ev["my"].mean() - ev["mp"].mean()), len(ev), len(s2)


# ---- per-station D_h matrix ----
st.subheader("Per-station $D_h(0.10)$ matrix")
rows = []
for city, sub in panel.groupby("city"):
    for h in HORIZONS:
        col = f"p_{int(h)}h"
        dh_v, n_ev, n_bk = station_dh(sub, col)
        rows.append({"city": city, "horizon_h": int(h),
                     "D_h": dh_v, "n_events": n_ev, "n_buckets": n_bk})
dh = pd.DataFrame(rows)
piv = dh.pivot(index="city", columns="horizon_h", values="D_h")
piv.columns = [f"{int(c)}h" for c in piv.columns]
piv = piv[[f"{int(h)}h" for h in HORIZONS]]
st.dataframe(piv.round(4), use_container_width=True)
st.caption(
    "Dots are hollow where fewer than 30 events back the station point. "
    "Horizons beyond 48h were dropped: 60h/72h coverage is too sparse."
)

# ---- equal-weighted D_h curve vs pooled ----
st.subheader("Equal-weighted vs Pooled $D_h(0.10)$")
eq_rows = []
pool_rows = []
for h in HORIZONS:
    col = f"p_{int(h)}h"
    # equal: mean of station-level D_h (ignore None)
    subh = dh[dh["horizon_h"] == int(h)]
    vals = [v for v in subh["D_h"] if v is not None]
    eq = float(np.mean(vals)) if vals else np.nan
    ne_all = [ne for ne in subh["n_events"] if ne > 0]
    avg_events = float(np.mean(ne_all)) if ne_all else np.nan
    eq_rows.append({
        "horizon_h": int(h), "D_h": eq,
        "n_stations": len(vals), "avg_events": avg_events,
    })
    # pooled: all events merged
    sub = panel[(panel[col].notna()) & (panel[col] <= THRESH)]
    ev = sub.groupby("event_id").agg(mp=(col, "mean"), my=("y", "mean"))
    pool_rows.append({"horizon_h": int(h), "D_h": float(ev["my"].mean() - ev["mp"].mean()),
                      "n_events_pooled": int(len(ev))})

eq = pd.DataFrame(eq_rows)
pool = pd.DataFrame(pool_rows)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=eq["horizon_h"], y=eq["D_h"], mode="lines+markers",
    name="Equal-weighted (per station)",
    line=dict(color="#2E86AB"),
    customdata=eq[["n_stations", "avg_events"]].values,
    hovertemplate="%{x:.0f}h: D_h=%{y:+.4f}<br>stations=%{customdata[0]}<br>avg events/station=%{customdata[1]:.0f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=pool["horizon_h"], y=pool["D_h"], mode="lines+markers",
    name="Pooled (per event)",
    line=dict(color="#A23B72", dash="dash"),
    customdata=pool[["n_events_pooled"]].values,
    hovertemplate="%{x:.0f}h: D_h=%{y:+.4f}<br>events=%{customdata[0]}<extra></extra>",
))
fig.add_hline(y=0, line_dash="dot", line_color="gray")
fig.update_layout(
    height=420,
    xaxis_title="Hours Before The Highest Temperature",
    yaxis_title="D_h(0.10)",
    xaxis=dict(tickmode="array", tickvals=HORIZONS,
               ticktext=[f"{int(h)}h" for h in HORIZONS]),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Pooled = every event has equal weight (London-dominated). "
    "Equal-weighted = every station has equal weight. Similar direction = robust."
)

# ---- per-station trajectories (small multiples) ----
st.subheader("Per-station $D_h(0.10)$ trajectories")
cities = sorted(panel["city"].unique())
fig2 = go.Figure()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
          "#393b79", "#637939", "#8c6d31"]
for i, city in enumerate(cities):
    sub = dh[(dh["city"] == city)].set_index("horizon_h")
    vals = sub["D_h"]
    # hollow marker where fewer than 30 events support the point
    symbols = ["circle" if ne >= 30 else "circle-open"
               for ne in sub["n_events"]]
    hover = [
        f"{int(h)}h<br>D_h={d:+.4f}<br>events={ne}<br>buckets={nb}"
        for h, d, ne, nb in zip(vals.index, vals.values,
                                sub["n_events"], sub["n_buckets"])
    ]
    fig2.add_trace(go.Scatter(
        x=vals.index, y=vals.values, mode="lines+markers",
        name=city, line=dict(color=colors[i % len(colors)], width=1.5),
        opacity=0.85, marker=dict(symbol=symbols, size=7),
        customdata=sub[["n_events", "n_buckets"]].values,
        hovertemplate="%{x:.0f}h: D_h=%{y:+.4f}<br>events=%{customdata[0]}<br>buckets=%{customdata[1]}<extra></extra>",
    ))
fig2.add_hline(y=0, line_dash="dot", line_color="gray")
fig2.update_layout(
    height=520,
    xaxis_title="Hours Before The Highest Temperature",
    yaxis_title="D_h(0.10)",
    xaxis=dict(tickmode="array", tickvals=HORIZONS,
               ticktext=[f"{int(h)}h" for h in HORIZONS]),
    legend=dict(font=dict(size=10)),
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Summary statistics")
eq_ok = eq.dropna()
st.markdown(
    f"- Equal-weighted D_h at 6h: **{eq_ok.loc[eq_ok['horizon_h']==6, 'D_h'].iloc[0]:+.4f}**\n"
    f"- Pooled D_h at 6h: **{pool.loc[pool['horizon_h']==6, 'D_h'].iloc[0]:+.4f}**\n"
    f"- Stations with positive (underpriced) 6h D_h: "
    f"**{(dh[(dh['horizon_h']==6) & dh['D_h'].notna()]['D_h']>0).sum()}/"
    f"{dh[(dh['horizon_h']==6) & dh['D_h'].notna()].shape[0]}**"
)
