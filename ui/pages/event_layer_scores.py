"""Event-distribution layer (Brier / Log Loss / RPS) page.

Implements Section 5 of the technical roadmap: treat the n temperature
buckets of each event as ONE probability distribution and score the
whole distribution.

Method:
  - Only FULL-COVERAGE events per horizon (every bucket of the event has
    a price in the +/-30min window) so normalization is well-defined.
  - Events with <7 buckets are dropped (broken bucket structure from
    unresolved markets, e.g. ATLANTA 106072).
  - S = sum of raw prices, q_k = p_k / S (normalized distribution).
  - BS = sum_k (q_k - y_k)^2
  - LogLoss = -log(q_winner)
  - RPS = sum_{k=1}^{n-1} (F_k - O_k)^2   (ordered buckets)
  - 95% CI via bootstrap resampling WHOLE events (roadmap 7.3).

Data: docs/icdm/output/event_layer_scores.csv + event_layer_scores_long.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

CSV_SUM = Path(r"d:\NN\PolyWeatherData\docs\icdm\output\event_layer_scores.csv")
CSV_LONG = Path(r"d:\NN\PolyWeatherData\docs\icdm\output\event_layer_scores_long.csv")

AXIS_FONT = dict(size=16, color="#000000", family="Arial, sans-serif", weight=700)
TICK_FONT = dict(size=14, color="#000000", family="Arial, sans-serif", weight=700)
LEGEND_FONT = dict(size=14, color="#000000", family="Arial, sans-serif", weight=700)

st.title(":dart: Event-Distribution Layer: Brier / Log Loss / RPS")
st.caption(
    "Second layer of the analysis: the n temperature buckets of each event "
    "are treated as ONE probability distribution (Section 5 of the roadmap). "
    "Scores are computed on normalized probabilities q = p / S and averaged "
    "over events; 95% CIs bootstrap whole events."
)

if not CSV_SUM.exists() or not CSV_LONG.exists():
    st.error("Run docs/icdm/code/10_event_level_scores.py first.")
    st.stop()

summ = pd.read_csv(CSV_SUM).sort_values("horizon_h")
long = pd.read_csv(CSV_LONG)

st.subheader("Scores vs time to high")
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Brier Score", "Log Loss", "RPS"),
    shared_yaxes=False,
)
metrics = [
    ("BS", "BS", "#1f6fb4", "BS"),
    ("LogLoss", "LogLoss", "#c1121f", "LogL"),
    ("RPS", "RPS", "#2e8b57", "RPS"),
]
for i, (key, title, color, ci_key) in enumerate(metrics, start=1):
    s = summ
    fig.add_trace(go.Scatter(
        x=s["horizon_h"], y=s[key], mode="lines+markers",
        name=title, line=dict(color=color, width=3),
        marker=dict(size=9),
        error_y=dict(
            type="data", symmetric=False,
            array=s[f"{ci_key}_ci_hi"] - s[key],
            arrayminus=s[key] - s[f"{ci_key}_ci_lo"],
            visible=True, thickness=1.2, color=color,
        ),
        hovertemplate="%{x:.0f}h: %{y:.4f}<extra></extra>",
    ), row=1, col=i)
    fig.update_xaxes(
        tickmode="array", tickvals=s["horizon_h"],
        ticktext=[f"{int(h)}h" for h in s["horizon_h"]],
        row=1, col=i, title=dict(text="Hours before high", font=AXIS_FONT),
        tickfont=TICK_FONT,
    )
    fig.update_yaxes(row=1, col=i, title=dict(text=title, font=AXIS_FONT),
                     tickfont=TICK_FONT)
fig.update_layout(
    height=420,
    showlegend=False,
    margin=dict(l=70, r=30, t=50, b=60),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "WARNING: per-horizon means are NOT directly comparable across "
    "horizons. Each horizon scores a DIFFERENT set of full-coverage events "
    "(adjacent horizons overlap only ~50%). The 24h and 9h 'dips' are "
    "sample-composition artifacts, not real effects: in paired comparisons "
    "on the SAME events, every adjacent pair improves monotonically (see "
    "the Paired comparison section below). Read this figure for the overall "
    "48h->1h improvement trend only, not for horizon-to-horizon wiggle. "
    "Error bars = bootstrap 95% CI over whole events."
)

st.subheader("Paired adjacent-horizon comparison (within common events)")
st.caption(
    "Because each horizon scores a DIFFERENT full-coverage event set, the "
    "means above are not comparable across horizons. The correct test is "
    "PAIRED: take events scored at BOTH horizons and compare on the same "
    "events. Every adjacent pair improves monotonically as the high "
    "approaches (delta < 0 in all three metrics), so the 24h and 9h 'dips' "
    "in the raw means are sample-composition artifacts, not real effects."
)
pairs_rows = []
HZ = [48, 42, 36, 30, 24, 18, 12, 9, 6, 3, 1]
for i in range(len(HZ) - 1):
    na, nb = HZ[i], HZ[i + 1]
    a = long[long["horizon_h"] == na][["event_id", "BS", "LogLoss", "RPS"]]
    b = long[long["horizon_h"] == nb][["event_id", "BS", "LogLoss", "RPS"]]
    merged = a.merge(b, on="event_id", suffixes=(f"_{na}", f"_{nb}"))
    if len(merged) == 0:
        continue
    pairs_rows.append({
        "pair": f"{na}h -> {nb}h",
        "n": len(merged),
        "dBS": float(merged[f"BS_{nb}"].mean() - merged[f"BS_{na}"].mean()),
        "dLogL": float(merged[f"LogLoss_{nb}"].mean() - merged[f"LogLoss_{na}"].mean()),
        "dRPS": float(merged[f"RPS_{nb}"].mean() - merged[f"RPS_{na}"].mean()),
    })
pairs = pd.DataFrame(pairs_rows)

figp = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Δ Brier", "Δ Log Loss", "Δ RPS"),
    shared_yaxes=False,
)
for i, (key, color) in enumerate([("dBS", "#1f6fb4"), ("dLogL", "#c1121f"),
                                  ("dRPS", "#2e8b57")], start=1):
    figp.add_trace(go.Bar(
        x=pairs["pair"], y=pairs[key], name=key,
        marker_color=color, marker_line_width=0,
        hovertemplate="%{x}: %{y:+.4f}<extra></extra>",
    ), row=1, col=i)
    figp.update_xaxes(
        tickangle=-45, tickfont=dict(size=11, color="#000"),
        row=1, col=i,
        title=dict(text="", font=AXIS_FONT),
    )
    figp.update_yaxes(
        row=1, col=i,
        title=dict(text=key, font=AXIS_FONT),
        tickfont=TICK_FONT,
    )
figp.update_layout(
    height=420,
    showlegend=False,
    margin=dict(l=70, r=30, t=50, b=90),
    plot_bgcolor="white",
)
st.plotly_chart(figp, use_container_width=True)
st.caption(
    "Delta = (nearer horizon) - (farther horizon) on the SAME events. "
    "All bars are negative: every step toward the high improves all three "
    "scores. Pair sample sizes: " + ", ".join(
        f"{r['pair']}(n={r['n']})" for _, r in pairs.iterrows()
    )
)
st.dataframe(pairs.round(4), use_container_width=True, hide_index=True)

st.subheader("Summary table")
disp = summ.copy()
disp["horizon"] = [f"{int(h)}h" for h in disp["horizon_h"]]
disp = disp[["horizon", "n_events", "S_mean", "S_median",
             "BS", "LogLoss", "RPS",
             "BS_ci_lo", "BS_ci_hi", "LogL_ci_lo", "LogL_ci_hi",
             "RPS_ci_lo", "RPS_ci_hi"]].copy()
disp.columns = [
    "Horizon", "Events", "S mean", "S median",
    "Brier", "LogLoss", "RPS",
    "Brier lo", "Brier hi", "LogL lo", "LogL hi", "RPS lo", "RPS hi",
]
st.dataframe(disp.round(4), use_container_width=True, hide_index=True)
st.caption(
    "S = sum of raw bucket prices before normalization (median 1.006-1.027, "
    "i.e. small market friction). CI = 2000-draw bootstrap over whole events."
)

# --------------------------------------------------------------------------
# Recalibration comparison (price3: 3 price segments, roadmap Section 6)
# --------------------------------------------------------------------------
PRICE3_CSV = Path(r"d:\NN\PolyWeatherData\docs\icdm\output\recalibration_price3_results.csv")
if PRICE3_CSV.exists():
    st.subheader("Recalibration comparison (price3)")
    st.caption(
        "Logit recalibration fitted per PRICE segment on the TRAIN set "
        "(roadmap Section 6, 3 segments: p<=0.15, 0.15<p<0.75, p>=0.75), "
        "then applied to each split. Parameters (train): "
        "p1: a=+0.111 b=+1.073 | p2: a=+0.015 b=+1.021 | "
        "p3: a=-0.593 b=+1.182. "
        "After per-bucket recalibration the event is re-normalized "
        "(q_hat = p_hat / sum p_hat) before scoring."
    )
    pr = pd.read_csv(PRICE3_CSV)
    pr["label"] = pr["split"] + " / " + pr["version"]
    pr = pr.sort_values(["split", "version"], ascending=[True, False])

    # comparison table
    disp_pr = pr[["split", "version", "n_events", "BS", "LogLoss", "RPS",
                  "BS_lo", "BS_hi", "LogLoss_lo", "LogLoss_hi",
                  "RPS_lo", "RPS_hi"]].copy()
    disp_pr.columns = ["Split", "Version", "Events", "Brier", "LogLoss", "RPS",
                       "Brier lo", "Brier hi", "LogL lo", "LogL hi",
                       "RPS lo", "RPS hi"]
    st.dataframe(disp_pr.round(4), use_container_width=True, hide_index=True)

    # delta (raw - price3) per split: positive = recalibration improves
    piv = pr.pivot(index="split", columns="version",
                   values=["BS", "LogLoss", "RPS"])
    deltas = []
    for sn in ["train", "val", "test"]:
        deltas.append({
            "split": sn,
            "dBS": float(piv.loc[sn, ("BS", "raw")] - piv.loc[sn, ("BS", "price3")]),
            "dLogL": float(piv.loc[sn, ("LogLoss", "raw")] - piv.loc[sn, ("LogLoss", "price3")]),
            "dRPS": float(piv.loc[sn, ("RPS", "raw")] - piv.loc[sn, ("RPS", "price3")]),
        })
    dd = pd.DataFrame(deltas)

    figp = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Δ Brier (raw - recal)", "Δ LogLoss (raw - recal)",
                        "Δ RPS (raw - recal)"),
        shared_yaxes=False,
    )
    for i, (key, color) in enumerate(
            [("dBS", "#1f6fb4"), ("dLogL", "#c1121f"), ("dRPS", "#2e8b57")],
            start=1):
        figp.add_trace(go.Bar(
            x=dd["split"], y=dd[key], name=key,
            marker_color=color, marker_line_width=0,
            text=[f"{v:+.4f}" for v in dd[key]], textposition="outside",
            textfont=dict(size=13, color="#000"),
            hovertemplate="%{x}: %{y:+.4f}<extra></extra>",
        ), row=1, col=i)
        figp.update_xaxes(
            tickfont=dict(size=13, color="#000"),
            row=1, col=i, title=dict(text="", font=AXIS_FONT),
        )
        figp.update_yaxes(row=1, col=i, title=dict(text=key, font=AXIS_FONT),
                          tickfont=TICK_FONT)
    figp.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=70, r=30, t=50, b=60),
        plot_bgcolor="white",
    )
    st.plotly_chart(figp, use_container_width=True)
    st.caption(
        "Delta > 0 means recalibration IMPROVES the score (lower is better). "
        "On train/val the price3 recalibration helps slightly (dRPS "
        "+0.001~+0.0012); on the 7-month TEST set the gain is +0.0004 - "
        "inside the bootstrap CI, i.e. not significant. This matches the "
        "roadmap Section 6 verdict: the bias is real but too small to be "
        "exploited out-of-sample with a simple calibration function."
    )

st.subheader("Random-uniform baseline")
st.markdown(
    "For context, an uninformative uniform distribution over n buckets "
    "scores: BS = 1 - 1/n, LogLoss = log(n), RPS = (n^2-1)/(12n). "
    "With median n=11 buckets that is BS ≈ 0.909, LogLoss ≈ 2.40, "
    "RPS ≈ 0.833. The market beats this at every horizon (even 48h)."
)

st.subheader("Method notes")
st.markdown(
    """
    - **Full-coverage events only**: a horizon is only scored on events
      where every bucket has a price in that +/-30min window (10-27% of
      events per horizon, 225-599 events). This is required for a
      well-defined normalization; sparse prices would bias S downward.
    - **Horizon subsets differ**: the full-coverage event set differs per
      horizon (adjacent horizons overlap only ~50%). Events that stay
      full-covered at 9h are biased toward 'already-priced' events (higher
      winner price), which mechanically lowers their scores. Compare only
      the overall trend, or use paired/intersection analysis.
    - **Bucket guard**: events with <7 buckets are excluded (a few events
      lost unresolved markets and have a broken bucket structure).
    - **Subset caveat**: full-coverage events have slightly fewer buckets
      (median 9 vs 11) and slightly lower winner prices than partial
      events, so per-horizon means reflect that subset, not all events.
    - **Selection**: D_h(0.10) on the full-coverage subset is -0.0014 vs
      +0.0017 on partial events at 6h - small but worth reporting.
    """
)
