"""First-Layer Aligned-Dates page (canonical integrated window).

Uses the canonical panel (every node clamped to [2025-12-05, 2026-08-10],
Tokyo/Shanghai start 2026-03) built by step 05 from the unified
`20251205_20260810` artifacts. Presents:

  1. D_h(0.10) vs horizon, three weighting schemes
  2. D_h(0.75~0.95) high-probability buckets, three weighting schemes
  3. Combined D_h(0.10) + D_h(0.75~0.95) on dual axes (event-weighted +
     station-equal for each range, color-coded by range)
  4. Reliability diagram with ALL horizons overlaid (light->dark = far->near)
  5. Per-station D_h(0.10) matrix (reference table)

Reads:
  - first_layer_panel_argmax_30min.parquet (canonical window)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUT_DIR, PANEL  # noqa: E402

HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
THRESH = 0.10
MIN_BUCKETS = 20

# ---- publication-ready style constants ----
AXIS_FONT = dict(size=18, color="#000000", family="Arial, sans-serif",
                 weight=700)
TICK_FONT = dict(size=16, color="#000000", family="Arial, sans-serif",
                 weight=700)
LEGEND_FONT = dict(size=16, color="#000000", family="Arial, sans-serif",
                   weight=700)
ZERO_LINE = dict(color="#666666", width=1.2, dash="dot")

st.title(":calendar: First-Layer: Aligned-Dates Experiment")

st.caption(
    "Canonical integrated window: every node clamped to "
    "[2025-12-05, 2026-08-10] (Tokyo/Shanghai start 2026-03). "
    "D_h(a~b) = mean(y) - mean(p) over contracts priced in [a, b]."
)

if not PANEL.exists():
    st.error("Panel not found. Run step 05 first.")
    st.stop()

panel = pd.read_parquet(PANEL)

OUTLIERS = ["DENVER", "DALLAS"]
exclude = st.toggle(
    "Exclude DENVER & DALLAS (outliers)", value=True,
    help="Their long-shot D_h trajectories differ systematically from the "
    "other 10 stations (continental climate, always-underpriced, late listing).",
)
if exclude:
    panel = panel[~panel["city"].isin(OUTLIERS)].reset_index(drop=True)

st.caption(
    f"{panel['event_id'].nunique()} events | {len(panel)} conditions | "
    f"{panel['city'].nunique()} cities"
    + (f" | Excluding: {', '.join(OUTLIERS)}" if exclude else "")
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _event_avg(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Collapse buckets to one row per event (mean price, mean outcome)."""
    return sub.groupby("event_id").agg(mp=(col, "mean"), my=("y", "mean"))


def _dh_one(col: str, sub: pd.DataFrame,
            equal_min_events: int = 5) -> dict:
    """D_h for one horizon: event, bucket, station-equal."""
    if len(sub) == 0:
        return {"event": np.nan, "bucket": np.nan, "equal": np.nan,
                "n_events": 0, "n_buckets": 0, "n_stations": 0}
    ev = _event_avg(sub, col)
    ev_dh = float(ev["my"].mean() - ev["mp"].mean())
    bk_dh = float(sub["y"].mean() - sub[col].mean())
    eq_vals, st_names = [], []
    for _, csub in sub.groupby("city"):
        e2 = _event_avg(csub, col)
        if len(e2) < equal_min_events:
            continue
        eq_vals.append(float(e2["my"].mean() - e2["mp"].mean()))
        st_names.append(csub["city"].iloc[0])
    eq_dh = float(np.mean(eq_vals)) if eq_vals else np.nan
    return {"event": ev_dh, "bucket": bk_dh, "equal": eq_dh,
            "n_events": int(len(ev)), "n_buckets": int(len(sub)),
            "n_stations": len(st_names)}


def dh_range(panel: pd.DataFrame, lo: float, hi: float,
             equal_min_events: int = 5) -> pd.DataFrame:
    """D_h over contracts with price in [lo, hi], per horizon, 3 schemes."""
    rows = []
    for h in HORIZONS:
        col = f"p_{int(h)}h"
        sub = panel[(panel[col].notna()) & (panel[col] >= lo)
                    & (panel[col] <= hi)].copy()
        r = _dh_one(col, sub, equal_min_events)
        rows.append({"horizon_h": int(h), **r})
    return pd.DataFrame(rows)


def _tick_layout(x_title: str, y_title: str, height: int) -> dict:
    return dict(
        height=height,
        xaxis=dict(
            tickmode="array", tickvals=HORIZONS,
            ticktext=[f"{int(h)}h" for h in HORIZONS],
            title=dict(text=x_title, font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=True, gridcolor="#E0E0E0",
        ),
        yaxis=dict(
            title=dict(text=y_title, font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=True, gridcolor="#E0E0E0",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0,
                    font=LEGEND_FONT),
        margin=dict(l=80, r=60, t=60, b=60),
        plot_bgcolor="white",
    )


# --------------------------------------------------------------------------
# 1) D_h(0.10) three ways
# --------------------------------------------------------------------------
st.subheader("$D_h(0.10)$ across horizons: three weighting schemes")
dh = dh_range(panel, 0.0, THRESH)

fig = go.Figure()
traces = [
    ("Event-weighted", "event", "#2E86AB", "solid"),
    ("Bucket-weighted", "bucket", "#A23B72", "dash"),
    ("Station-equal", "equal", "#3B7A57", "dot"),
]
for name, key, color, dash in traces:
    fig.add_trace(go.Scatter(
        x=dh["horizon_h"], y=dh[key], mode="lines+markers",
        name=name, line=dict(color=color, dash=dash, width=3),
        marker=dict(size=9),
        customdata=dh[["n_events", "n_buckets"]],
        hovertemplate="%{x:.0f}h: D_h=%{y:+.4f}<br>events=%{customdata[0]}<br>buckets=%{customdata[1]}<extra></extra>",
    ))
fig.add_hline(y=0, line_dash=ZERO_LINE["dash"], line_color=ZERO_LINE["color"],
              line_width=ZERO_LINE["width"])
fig.update_layout(**_tick_layout(
    "Hours Before The Highest Temperature", "D_h(0.10) = mean(y - p), p<=10%", 460))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "All three schemes agree in sign at every horizon: 48h-12h overpricing "
    "(D_h<0), 9h-3h underpricing (D_h>0), 1h back to overpricing. "
    "Magnitudes differ (bucket-weighted is closest to zero) but the "
    "long-shot bias pattern is weighting-robust."
)

# --------------------------------------------------------------------------
# 2) D_h(0.75~0.95): high-probability bucket bias
# --------------------------------------------------------------------------
st.subheader("$D_h(0.75\\!\\sim\\!0.95)$ across horizons: high-probability buckets")
st.caption(
    "Same three schemes over contracts priced in [0.75, 0.95]. Positive D_h "
    "= underpriced favorite, negative = overpriced. Far out the likely "
    "outcome is underpriced; as the high approaches it swings to overpriced "
    "(herding into the favorite). Each event has ~1 bucket in this range, so "
    "event- and bucket-weighted coincide; samples are smaller (30-435 "
    "events) and noisier. Station-equal requires >=5 events per station."
)
dh_hi = dh_range(panel, 0.75, 0.95)

fig1b = go.Figure()
for name, key, color, dash in traces:
    fig1b.add_trace(go.Scatter(
        x=dh_hi["horizon_h"], y=dh_hi[key], mode="lines+markers",
        name=name, line=dict(color=color, dash=dash, width=3),
        marker=dict(size=9),
        customdata=dh_hi[["n_events", "n_buckets"]],
        hovertemplate="%{x:.0f}h: D_h=%{y:+.4f}<br>events=%{customdata[0]}<br>buckets=%{customdata[1]}<extra></extra>",
    ))
fig1b.add_hline(y=0, line_dash=ZERO_LINE["dash"], line_color=ZERO_LINE["color"],
                line_width=ZERO_LINE["width"])
fig1b.update_layout(**_tick_layout(
    "Hours Before The Highest Temperature", "D_h(0.75~0.95) = mean(y - p)", 440))
st.plotly_chart(fig1b, use_container_width=True)
st.caption(
    "Event-weighted (solid blue) and bucket-weighted (dashed purple) overlap "
    "because each event contributes one bucket in [0.75, 0.95]. Reading "
    "(event-weighted): +0.007 at 48h -> +0.035..+0.039 at 30h-24h "
    "(underpriced) -> -0.058 at 12h -> -0.107 at 6h -> -0.087 at 3h (most "
    "overpriced) -> -0.033 at 1h. The swing is the mirror image of the "
    "long-shot pattern."
)

# --------------------------------------------------------------------------
# 3) combined D_h(0.10) + D_h(0.75~0.95), dual axes, event+equal only
# --------------------------------------------------------------------------
st.subheader("Combined: $D_h(0.10)$ vs $D_h(0.75\\!\\sim\\!0.95)$ (dual axes)")
st.caption(
    "D_h(0.10) on the LEFT axis (blue/purple), D_h(0.75~0.95) on the RIGHT "
    "axis (red/orange). Each range shows event-weighted (solid) and "
    "station-equal (dashed) lines. The two ranges move in opposite "
    "directions: long-shots overpriced where favorites are underpriced and "
    "vice versa - the market's probability mass is persistently mis-allocated."
)

fig4 = go.Figure()
# D_h(0.10): blue (event) / purple (equal) on left axis
fig4.add_trace(go.Scatter(
    x=dh["horizon_h"], y=dh["event"], mode="lines+markers",
    name="D_h(0.10) event-wtd", line=dict(color="#1f6fb4", width=3),
    marker=dict(size=9), yaxis="y",
    customdata=dh[["n_events"]],
    hovertemplate="D_h(0.10) event: %{y:+.4f} (n=%{customdata[0]})<extra></extra>",
))
fig4.add_trace(go.Scatter(
    x=dh["horizon_h"], y=dh["equal"], mode="lines+markers",
    name="D_h(0.10) station-eq", line=dict(color="#8e44ad", width=3, dash="dash"),
    marker=dict(size=9), yaxis="y",
    customdata=dh[["n_events"]],
    hovertemplate="D_h(0.10) equal: %{y:+.4f}<extra></extra>",
))
# D_h(0.75~0.95): red (event) / orange (equal) on right axis
fig4.add_trace(go.Scatter(
    x=dh_hi["horizon_h"], y=dh_hi["event"], mode="lines+markers",
    name="D_h(0.75~0.95) event-wtd", line=dict(color="#c1121f", width=3),
    marker=dict(size=9), yaxis="y2",
    customdata=dh_hi[["n_events"]],
    hovertemplate="D_h(0.75~0.95) event: %{y:+.4f} (n=%{customdata[0]})<extra></extra>",
))
fig4.add_trace(go.Scatter(
    x=dh_hi["horizon_h"], y=dh_hi["equal"], mode="lines+markers",
    name="D_h(0.75~0.95) station-eq", line=dict(color="#f39c12", width=3, dash="dash"),
    marker=dict(size=9), yaxis="y2",
    customdata=dh_hi[["n_events"]],
    hovertemplate="D_h(0.75~0.95) equal: %{y:+.4f}<extra></extra>",
))
fig4.add_hline(y=0, line_dash=ZERO_LINE["dash"], line_color=ZERO_LINE["color"],
               line_width=ZERO_LINE["width"], yref="y")
fig4.add_hline(y=0, line_dash=ZERO_LINE["dash"], line_color="#999999",
               line_width=1, yref="y2")
fig4.update_layout(
    height=480,
    xaxis=dict(
        tickmode="array", tickvals=HORIZONS,
        ticktext=[f"{int(h)}h" for h in HORIZONS],
        title=dict(text="Hours Before The Highest Temperature", font=AXIS_FONT),
        tickfont=TICK_FONT, showgrid=True, gridcolor="#E0E0E0",
    ),
    yaxis=dict(
        title=dict(text="D_h(0.10)", font=AXIS_FONT),
        tickfont=TICK_FONT, showgrid=True, gridcolor="#E0E0E0",
    ),
    yaxis2=dict(
        title=dict(text="D_h(0.75~0.95)", font=AXIS_FONT),
        tickfont=TICK_FONT, overlaying="y", side="right",
        showgrid=False,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0,
                font=LEGEND_FONT),
    margin=dict(l=80, r=80, t=60, b=60),
    plot_bgcolor="white",
)
st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "D_h(0.10) (left) goes negative at 48h-12h then positive at 9h-3h; "
    "D_h(0.75~0.95) (right) mirrors it: positive (underpriced favorites) "
    "far out, negative (overpriced favorites) as the high nears. The "
    "mirror-image pattern means the market consistently overprices the "
    "improbable AND the almost-certain outcomes relative to the middle."
)

# --------------------------------------------------------------------------
# 4) reliability diagram, all horizons in one figure
# --------------------------------------------------------------------------
st.subheader("Reliability diagram: all horizons overlaid")
st.caption(
    "For each horizon, contracts are binned by quoted price; x = mean quoted "
    "price per bin, y = mean realized outcome per bin (event-averaged). "
    "Solid red dashed line = perfect calibration. Lines below the diagonal "
    "at low prices = overpricing."
)

BINS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.01]
BIN_LABELS = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-30%",
              "30-50%", "50-70%", "70-90%", "90-100%"]


def reliability_points(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    df = df[df[pcol].notna()].copy()
    df["bin"] = pd.cut(df[pcol], bins=BINS, labels=BIN_LABELS,
                       include_lowest=True)
    ev = df.groupby(["event_id", "bin"], observed=True).agg(
        p=(pcol, "mean"), y=("y", "mean")).reset_index()
    agg = ev.groupby("bin", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean"),
        n_events=("event_id", "nunique")).reset_index()
    return agg


fig3 = go.Figure()
# Color ramp: far horizon (48h) = lightest, near horizon (1h) = darkest.
rel_colors = sample_colorscale("Blues", len(HORIZONS))
# Order traces so the SHORT names come first and the long "Perfect
# calibration" label goes last - the h-mode legend then wraps into 2 rows.
for i, h in enumerate(HORIZONS):
    col = f"p_{int(h)}h"
    rp = reliability_points(panel, col)
    rp = rp[rp["n_events"] >= 30]  # drop noisy bins
    fig3.add_trace(go.Scatter(
        x=rp["mean_p"], y=rp["mean_y"], mode="lines+markers",
        name=f"{int(h)}h",
        line=dict(color=rel_colors[i], width=1.3),
        marker=dict(size=5),
        showlegend=True,
        customdata=rp[["n_events"]],
        hovertemplate="%{x:.3f} -> %{y:.3f}<br>events=%{customdata[0]}<extra></extra>",
    ))
fig3.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(color="#E63946", dash="dash", width=3.5),
    name="Perfect calibration", hoverinfo="skip", showlegend=True,
))
fig3.update_layout(
    height=600,
    # Native plotly legend: clickable, attached to the chart, wrapped into
    # ~2 rows by packing short horizon names first and the long label last.
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.0,
        xanchor="left", x=0,
        font=dict(size=11, color="#000000", family="Arial, sans-serif",
                   weight=700),
        tracegroupgap=4,
    ),
    xaxis=dict(
        title=dict(text="Quoted price (bin mean)", font=AXIS_FONT),
        range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        tickfont=TICK_FONT,
        showgrid=True, gridcolor="#E0E0E0",
    ),
    yaxis=dict(
        title=dict(text="Realized frequency (bin mean)", font=AXIS_FONT),
        range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        tickfont=TICK_FONT,
        showgrid=True, gridcolor="#E0E0E0",
    ),
    margin=dict(l=80, r=40, t=70, b=60),
    plot_bgcolor="white",
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Color = time to high: 48h lightest, 1h darkest. At low prices the "
    "light (far) curves sit below the red diagonal (overpriced long-shots); "
    "at high prices the dark (near) curves sit below it (overpriced "
    "favorites). Bins with <30 events are dropped."
)

# --------------------------------------------------------------------------
# 4a-bis) bias heatmap: D per (price bin x horizon)
# --------------------------------------------------------------------------
st.subheader("Bias heatmap: $D$ per price bin and horizon")


def bias_matrix(df: pd.DataFrame):
    """Event-weighted D in percentage points per (price bin, horizon).

    Returns (D, n_events, standard_error) matrices, rows = price bins,
    columns = horizons (same order as HORIZONS).
    """
    Dm = np.full((len(BIN_LABELS), len(HORIZONS)), np.nan)
    Nm = np.zeros((len(BIN_LABELS), len(HORIZONS)), dtype=int)
    Sm = np.full((len(BIN_LABELS), len(HORIZONS)), np.nan)
    for j, h in enumerate(HORIZONS):
        col = f"p_{int(h)}h"
        dd = df[df[col].notna()].copy()
        dd["bin"] = pd.cut(dd[col], bins=BINS, labels=BIN_LABELS,
                           include_lowest=True)
        ev = dd.groupby(["event_id", "bin"], observed=True).agg(
            mp=(col, "mean"), my=("y", "mean")).reset_index()
        g = ev.groupby("bin", observed=True).agg(
            mp=("mp", "mean"), my=("my", "mean"),
            se=("my", "sem"), n=("event_id", "nunique"))
        for i, lab in enumerate(BIN_LABELS):
            if lab in g.index:
                r = g.loc[lab]
                Dm[i, j] = (r["my"] - r["mp"]) * 100.0
                Nm[i, j] = int(r["n"])
                Sm[i, j] = (float(r["se"]) * 100.0
                            if pd.notna(r["se"]) else np.nan)
    return Dm, Nm, Sm


_D, _N, _S = bias_matrix(panel)
_vmax = float(np.nanmax(np.abs(_D)))
# Diverging scale centred on ZERO: white = no bias, strong colour = big
# bias. Blue = D<0 (market OVERprices), red = D>0 (UNDERprices). Because the
# colour saturates with |D|, the intensity alone tells you the magnitude and
# the hue tells you the direction.
_BIAS_COLORS = [
    [0.00, "#4575b4"],   # darkest blue  -> largest negative D
    [0.25, "#abd0e6"],
    [0.50, "#ffffff"],   # white         -> D = 0 (no bias)
    [0.75, "#fdae61"],
    [1.00, "#d73027"],   # darkest red   -> largest positive D
]
_ann = np.empty_like(_D, dtype=object)
for _i in range(_D.shape[0]):
    for _j in range(_D.shape[1]):
        if np.isnan(_D[_i, _j]):
            _ann[_i, _j] = ""
            continue
        _star = ("*" if (np.isfinite(_S[_i, _j])
                         and abs(_D[_i, _j]) > 2 * _S[_i, _j]) else "")
        _ann[_i, _j] = f"{_D[_i, _j]:+.1f}{_star}"

bias_hm = go.Figure(go.Heatmap(
    z=_D,
    x=[f"{int(h)}h" for h in HORIZONS],
    y=BIN_LABELS,
    zmin=-_vmax, zmax=_vmax, zmid=0,
    colorscale=_BIAS_COLORS,
    text=_ann, texttemplate="%{text}",
    textfont=dict(size=10, color="#111111", family="Arial, sans-serif",
                  weight=700),
    colorbar=dict(title=dict(text="D (pp)", font=AXIS_FONT),
                  tickfont=TICK_FONT, tickvals=[-_vmax, 0, _vmax],
                  ticktext=[f"{-_vmax:.1f}", "0", f"{_vmax:.1f}"]),
    customdata=_N,
    hovertemplate=("bin %{y}, %{x}<br>D = %{z:+.2f} pp"
                   "<br>events = %{customdata}<extra></extra>"),
))
_SIG_NOTE = (
    "Cell text = D in percentage points (realized frequency - quoted "
    "price).<br>"
    "*  =  |D| exceeds 2 standard errors across events (~95%): "
    "distinguishable from 0.  Unmarked cells are only CONSISTENT with "
    "zero.<br>"
    "BLUE = negative D (market OVERprices)   |   WHITE = no bias   |   "
    "RED = positive D (market UNDERprices);  colour depth = |D|."
)
bias_hm.update_layout(
    height=620,
    xaxis=dict(title=dict(text="Hours Before The Highest Temperature",
                          font=AXIS_FONT), tickfont=TICK_FONT),
    yaxis=dict(title=dict(text="Quoted price bin", font=AXIS_FONT),
               tickfont=TICK_FONT, autorange="reversed"),
    margin=dict(l=90, r=40, t=50, b=150),
    plot_bgcolor="white",
    annotations=[
        dict(x=0.5, y=-0.20, xref="paper", yref="paper",
             xanchor="center", yanchor="top", showarrow=False,
             align="center", text=_SIG_NOTE,
             font=dict(size=12, color="#111111",
                       family="Arial, sans-serif", weight=700)),
    ],
)
st.plotly_chart(bias_hm, use_container_width=True)
st.caption(
    "Same quantity as the $D_h$ curves above, but resolved over BOTH axes: "
    "cell = event-weighted $D$ in percentage points, i.e. "
    "mean(realized frequency) - mean(quoted price) for the buckets in that "
    "price bin at that horizon. Colour is a DIVERGING scale centred on "
    "zero, so the intensity reads as bias magnitude at a glance: "
    "**white = no bias**, light TINT = small bias, deep colour = large "
    "bias. Hue gives the direction - BLUE = $D<0$ (market OVERprices, "
    "i.e. buckets win less often than quoted) and RED = $D>0$ (market "
    "UNDERprices). The scale is symmetric at +/-"
    f"{_vmax:.1f}pp, the largest |D| in the grid. Numbers are the cell "
    "value; a star (*) marks cells where |D| exceeds 2 standard errors of "
    "the event-level realized frequency (prices are treated as fixed "
    "observations, so all sampling variability is attributed to the "
    "outcome), i.e. statistically distinguishable from zero at roughly the "
    "95% level - unmarked cells are merely CONSISTENT with zero, not proven "
    "unbiased. Reading: "
    "the strongest signals are the deep blue cells in the expensive bins "
    "at 6h-12h (favorites most overpriced) and the deep red cell at 30h in "
    "the 70-90% bin, while the long-shot rows (0-1% to 5-10%) stay pale - "
    "their bias is small in every horizon. Only 19 of 110 cells are "
    "significant, so treat the pale cells as consistent-with-zero rather "
    "than as evidence of accurate pricing."
)

# --------------------------------------------------------------------------
# per-station matrix (reference)
# --------------------------------------------------------------------------
st.subheader("Per-station $D_h(0.10)$ matrix")
st.caption(
    "Stations with no valid entry have too few buckets (<20) at that horizon."
)


def station_long(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, sub in panel.groupby("city"):
        for h in HORIZONS:
            col = f"p_{int(h)}h"
            s2 = sub[(sub[col].notna()) & (sub[col] <= THRESH)]
            if len(s2) < MIN_BUCKETS:
                rows.append({"city": city, "horizon_h": int(h),
                             "D_h": np.nan, "n_events": 0, "n_buckets": 0})
                continue
            ev = _event_avg(s2, col)
            rows.append({"city": city, "horizon_h": int(h),
                         "D_h": float(ev["my"].mean() - ev["mp"].mean()),
                         "n_events": int(len(ev)), "n_buckets": int(len(s2))})
    return pd.DataFrame(rows)


long_dh = station_long(panel)
piv = long_dh.pivot(index="city", columns="horizon_h", values="D_h")
piv.columns = [f"{int(c)}h" for c in piv.columns]
piv = piv[[f"{int(h)}h" for h in HORIZONS]]
piv_ne = long_dh.pivot(index="city", columns="horizon_h", values="n_events")
piv_ne.columns = [f"{int(c)}h" for c in piv_ne.columns]
piv_ne = piv_ne[[f"{int(h)}h" for h in HORIZONS]]

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**D_h(0.10)**")
    st.dataframe(piv.round(4), use_container_width=True)
with col_r:
    st.markdown("**Events per node-horizon**")
    st.dataframe(piv_ne.astype("Int64"), use_container_width=True)
