"""Favorite/Longshot buy-NO strategy backtest results page.

Renders, from analysis_klga_l2_strategy/:
  - cross_city_fav_ls_summary.csv   (10-city favorite & longshot buy-NO)
  - favorite_fixed_h_bins.csv       (fixed signal-hour bins with CI)
Fee model: real Polymarket schedule Fee = Theta * C * p * (1 - p),
with Theta = 0.05, C = shares bought, p = traded token price.
Highlights: favorite (1-12h, YES 0.75-0.95) buy-NO is profitable post-fee
(+51c/$1 pooled; +42c at h=1), longshot (30-48h, YES<=0.10) buy-NO is roughly
breakeven (-0.23% ROI) under the real fee schedule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = Path(r"d:\NN\PolyWeatherData\analysis_klga_l2_strategy")
SUMMARY = OUT / "cross_city_fav_ls_summary.csv"
HBINS = OUT / "favorite_fixed_h_bins.csv"
NOTIONAL = 10.0

st.title("Strategy Backtest: Favorite/Longshot buy-NO")
st.caption(
    "10 cities (excl Denver/Dallas), pre-July settled history (<=2026-06-30). "
    "Buy NO at last trade in window, $10 notional, hold to settlement. "
    "Fee = Theta x C x p x (1 - p), with Theta = 0.05, C = shares bought, "
    "p = traded token price (Polymarket schedule)."
)

if not SUMMARY.exists() or not HBINS.exists():
    st.error(
        "Results not found. Run scripts/cross_city_fav_ls_pnl.py and "
        "scripts/favorite_fixed_h_bins.py first."
    )
    st.stop()

summary = pd.read_csv(SUMMARY)
hbins = pd.read_csv(HBINS)

fav = summary[summary["strategy"] == "favorite_NO"]
ls = summary[summary["strategy"] == "longshot_NO"]
fav_pool = fav[["n_trades", "pnl_pre", "pnl_post"]].sum()
ls_pool = ls[["n_trades", "pnl_pre", "pnl_post"]].sum()

# ---------- summary table: total investment, ROI, win rate, bootstrap CI ----------
FAV_TRADES = OUT / "cross_city_favorite_NO_trades.csv"
LS_TRADES = OUT / "cross_city_longshot_NO_trades.csv"


def event_bootstrap_ci(trades: pd.DataFrame, nboot: int = 500, seed: int = 42) -> tuple[float, float]:
    """95% CI of pooled post-fee per-$1 PNL under event-block resampling."""
    rng = np.random.default_rng(seed)
    ev = trades.groupby("event_id").agg(
        pnl=("pnl_post", "sum"), n=("pnl_post", "size")
    )
    pnl = ev["pnl"].to_numpy()
    cnt = ev["n"].to_numpy()
    eids = np.arange(len(ev))
    dist = np.empty(nboot)
    for b in range(nboot):
        idx = rng.choice(eids, size=len(eids), replace=True)
        dist[b] = pnl[idx].sum() / (cnt[idx].sum() * NOTIONAL)
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


fav_trades = pd.read_csv(FAV_TRADES) if FAV_TRADES.exists() else None
ls_trades = pd.read_csv(LS_TRADES) if LS_TRADES.exists() else None

fav_n = len(fav_trades) if fav_trades is not None else int(fav["n_trades"].sum())
ls_n = len(ls_trades) if ls_trades is not None else int(ls["n_trades"].sum())
fav_notional = fav_n * NOTIONAL
ls_notional = ls_n * NOTIONAL
fav_pnl_pre = fav_trades["pnl_pre"].sum() if fav_trades is not None else fav_pool["pnl_pre"]
ls_pnl_pre = ls_trades["pnl_pre"].sum() if ls_trades is not None else ls_pool["pnl_pre"]
fav_pnl_post = fav_trades["pnl_post"].sum() if fav_trades is not None else fav_pool["pnl_post"]
ls_pnl_post = ls_trades["pnl_post"].sum() if ls_trades is not None else ls_pool["pnl_post"]
fav_win = fav_trades["won"].mean() if fav_trades is not None else float(fav["win_rate"].mean())
ls_win = ls_trades["won"].mean() if ls_trades is not None else float(ls["win_rate"].mean())
fav_ci = event_bootstrap_ci(fav_trades) if fav_trades is not None else (np.nan, np.nan)
ls_ci = event_bootstrap_ci(ls_trades) if ls_trades is not None else (np.nan, np.nan)

def _usd(v: float, digits: int = 0) -> str:
    """Signed dollar with sign before the $: +$1,011 / -$153 / +$5.00."""
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.{digits}f}"


def _usd_pct(v_pnl: float, v_roi: float) -> str:
    """+$1,011 / +54.96%  or  -$153 / -0.13%."""
    return f"{_usd(v_pnl)} / {v_roi:+.2f}%"


fav_cells = [
    f"${fav_notional:,.0f} ({fav_n:,} Trades x $10)",
    _usd_pct(fav_pnl_pre, fav_pnl_pre / fav_notional * 100),
    _usd_pct(fav_pnl_post, fav_pnl_post / fav_notional * 100),
    _usd(fav_pnl_post / fav_n, 2),
    f"{fav_win * 100:.1f}%",
    f"[{fav_ci[0] * 100:+.2f}c, {fav_ci[1] * 100:+.2f}c]",
]
ls_cells = [
    f"${ls_notional:,.0f} ({ls_n:,} Trades x $10)",
    _usd_pct(ls_pnl_pre, ls_pnl_pre / ls_notional * 100),
    _usd_pct(ls_pnl_post, ls_pnl_post / ls_notional * 100),
    _usd(ls_pnl_post / ls_n, 2),
    f"{ls_win * 100:.1f}%",
    f"[{ls_ci[0] * 100:+.2f}c, {ls_ci[1] * 100:+.2f}c]",
]
labels = [
    "Total Notional Invested",
    "PnL Pre-Fee / ROI",
    "PnL Post-Fee / ROI",
    "Expected PnL Per Trade",
    "Win Rate",
    "Event-Block Bootstrap 95% CI (Post-Fee, Per $1)",
]

TH = 'style="text-align:center;font-weight:700;padding:6px 16px;white-space:nowrap;border:1px solid #555;background:#eee;"'
TD = 'style="text-align:center;font-weight:700;padding:6px 16px;white-space:nowrap;border:1px solid #555;"'
html_rows = "".join(
    f"<tr><td {TD}>{label}</td><td {TD}>{fv}</td><td {TD}>{lv}</td></tr>"
    for label, fv, lv in zip(labels, fav_cells, ls_cells)
)
summary_html = (
    '<div style="display:flex;justify-content:center;margin:8px 0;">'
    '<table style="border-collapse:collapse;font-size:14px;background:#fff;">'
    f'<tr><th {TH}></th><th {TH}>Favorite Buy-NO (1-12h)</th>'
    f'<th {TH}>Longshot Buy-NO (30-48h)</th></tr>'
    f"{html_rows}</table></div>"
)
st.subheader("Total Investment and Returns")
st.markdown(summary_html, unsafe_allow_html=True)
st.caption(
    "Bootstrap CI: 500 resamples of full events (all buckets kept together); "
    "interval entirely above zero means the profit is statistically distinct from luck."
)

st.markdown("---")

# ---------- pooled vs per-city table ----------
st.subheader("Per-City Results (Post-Fee Per $1)")
pivot = summary.pivot(index="city", columns="strategy",
                      values=["n_trades", "per1_post"]).sort_index()
pivot.columns = [
    f"{a}_{b}" for a, b in pivot.columns
]
show = pivot[["n_trades_favorite_NO", "per1_post_favorite_NO",
              "n_trades_longshot_NO", "per1_post_longshot_NO"]].copy()
show.columns = ["Fav Trades", "Fav Post-Fee Per $1", "LS Trades", "LS Post-Fee Per $1"]
show = show.sort_values("Fav Post-Fee Per $1", ascending=False)
st.dataframe(
    show.style.format({"Fav Post-Fee Per $1": "{:+.4f}", "LS Post-Fee Per $1": "{:+.4f}"}),
    use_container_width=True,
)

# ---------- per-city bar chart ----------
fig = px.bar(
    show.reset_index(),
    x="city", y=["Fav Post-Fee Per $1", "LS Post-Fee Per $1"],
    barmode="group",
    labels={"value": "Post-Fee Per-$1 PNL", "variable": "Strategy"},
    title="Post-Fee Per-$1 PNL by City: Favorite vs Longshot Buy-NO",
)
fig.update_layout(legend_title_text="Strategy")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- fixed-h bins ----------
st.subheader("Favorite Buy-NO by Fixed Signal Hour h")
st.caption(
    "Fixed h = last trade in [h, h+1h) before settlement; "
    "error bars = event-block bootstrap 95% CI."
)
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=hbins["h"], y=hbins["per1_post"] * 100,
    name="Post-Fee Per-$1 (Cents)",
    error_y=dict(
        type="data",
        array=(hbins["ci_high"] - hbins["per1_post"]) * 100,
        arrayminus=(hbins["per1_post"] - hbins["ci_low"]) * 100,
        visible=True,
    ),
    marker_color=np.where(hbins["per1_post"] >= 0, "#2ca02c", "#d62728"),
))
fig2.add_hline(y=0, line_dash="dash", line_color="grey")
fig2.update_layout(
    title="Favorite Buy-NO Post-Fee Per-$1 (Cents) vs Signal Hour h",
    xaxis_title="Signal Hour h Before Settlement",
    yaxis_title="Post-Fee Per-$1 (Cents)",
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

hbins_show = hbins.copy()
hbins_show.columns = [
    "Hour h", "Trades", "Events", "Cities", "Win Rate",
    "PnL Post-Fee", "Post-Fee Per $1", "ROI %", "CI Low", "CI High",
]
st.dataframe(
    hbins_show.style.format(
        {"Post-Fee Per $1": "{:+.4f}", "ROI %": "{:+.2f}",
         "CI Low": "{:+.4f}", "CI High": "{:+.4f}", "Win Rate": "{:.1%}"}
    ),
    use_container_width=True,
)

st.markdown("---")
st.markdown(
    """
    **Takeaway:** Favorite (0.75-0.95) Buy-NO at 1-12h before settlement is
    profitable post-fee (+50.71% ROI, +50.7c/$1 pooled; +41.9c at h=1,
    statistically positive), driven by favorite overpricing (~5.7pp).
    Longshot (<=0.10) Buy-NO at 30-48h has no exploitable bias (D ~ +0.4c) and
    is roughly breakeven under the real fee schedule (-0.23% ROI), because the
    fee Theta x p x (1-p) nearly vanishes at extreme prices.
    """
)
