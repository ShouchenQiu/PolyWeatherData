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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUT_DIR  # noqa: E402

CSV_SUM = OUT_DIR / "event_layer_scores.csv"
CSV_LONG = OUT_DIR / "event_layer_scores_long.csv"

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
    st.error("Run pipeline/2_scores/20_event_level_scores.py first.")
    st.stop()

summ = pd.read_csv(CSV_SUM).sort_values("horizon_h")
long = pd.read_csv(CSV_LONG)

st.subheader("Scores vs time to high (combined, normalized to 48h = 1)")
# The three metrics live on very different scales (Brier ~0.70, LogLoss
# ~1.40, RPS ~0.50), so plotting them on one shared y-axis would squash
# two of them. Normalize each metric by its 48h value: every line then
# starts at 1.0 and shows the RELATIVE improvement as the high approaches.
base = summ[summ["horizon_h"] == 48].iloc[0]
norm = summ.copy()
norm_labels = {}
for key, ci_key in [("BS", "BS"), ("LogLoss", "LogL"), ("RPS", "RPS")]:
    b = base[key]
    norm[f"{key}_n"] = summ[key] / b
    norm[f"{key}_lo_n"] = summ[f"{ci_key}_ci_lo"] / b
    norm[f"{key}_hi_n"] = summ[f"{ci_key}_ci_hi"] / b
    # relative improvement 48h -> 1h, for the legend
    v1 = summ[summ["horizon_h"] == 1][key].iloc[0]
    norm_labels[key] = (1.0 - v1 / b) * 100

fig = go.Figure()
metric_lines = [
    ("BS", "Brier", "#1f6fb4", "BS"),
    ("LogLoss", "Log Loss", "#c1121f", "LogL"),
    ("RPS", "RPS", "#2e8b57", "RPS"),
]
for key, title, color, ci_key in metric_lines:
    y = norm[f"{key}_n"]
    fig.add_trace(go.Scatter(
        x=norm["horizon_h"], y=y, mode="lines+markers",
        name=title,
        line=dict(color=color, width=3),
        marker=dict(size=9),
        error_y=dict(
            type="data", symmetric=False,
            array=norm[f"{key}_hi_n"] - y,
            arrayminus=y - norm[f"{key}_lo_n"],
            visible=True, thickness=1.2, color=color,
        ),
        hovertemplate="%{x:.0f}h: %{y:.4f}<extra></extra>",
    ))
fig.add_hline(y=1.0, line_dash="dash", line_color="#888888",
              line_width=1.2, annotation_text="48h baseline",
              annotation_font=dict(size=12, color="#666666"))
fig.update_xaxes(
    tickmode="array", tickvals=norm["horizon_h"],
    ticktext=[f"{int(h)}h" for h in norm["horizon_h"]],
    title=dict(text="Hours Before The Highest Temperature", font=AXIS_FONT),
    tickfont=TICK_FONT,
)
fig.update_yaxes(title=dict(text="Relative Score", font=AXIS_FONT),
                 tickfont=TICK_FONT)
fig.update_layout(
    height=430,
    margin=dict(l=80, r=40, t=50, b=60),
    plot_bgcolor="white",
    legend=dict(font=dict(size=14, color="#000000",
                          family="Arial, sans-serif", weight=700),
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The three metrics are combined on ONE chart by normalizing each to "
    "its 48h value (48h = 1.0); the y-axis shows relative score, so all "
    "three lines are directly comparable. Legend shows the total relative "
    "improvement from 48h to 1h. WARNING: per-horizon means are NOT "
    "directly comparable across horizons - each horizon scores a DIFFERENT "
    "set of full-coverage events (adjacent horizons overlap only ~50%). "
    "The 24h and 9h 'dips' are sample-composition artifacts, not real "
    "effects (see the Paired comparison section). Error bars = bootstrap "
    "95% CI over whole events, normalized by the same 48h value. Raw "
    "metric values are in the Summary table below."
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
    horizontal_spacing=0.16,
)
for i, (key, color) in enumerate([("dBS", "#1f6fb4"), ("dLogL", "#c1121f"),
                                  ("dRPS", "#2e8b57")], start=1):
    figp.add_trace(go.Bar(
        x=pairs["pair"], y=pairs[key], name=key,
        marker_color=color, marker_line_width=0,
        hovertemplate="%{x}: %{y:+.4f}<extra></extra>",
    ), row=1, col=i)
    figp.update_xaxes(
        tickangle=-45, tickfont=dict(size=11, color="#000",
                                      family="Arial, sans-serif",
                                      weight=700),
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
figp.update_annotations(font=dict(color="#000000", size=16,
                                  family="Arial, sans-serif", weight=700))
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
# Recalibration comparison (B-spline: 7 / 8 / 9 total parameters)
# --------------------------------------------------------------------------
SPLINE_CSV = OUT_DIR / "recalibration_spline_results.csv"
if SPLINE_CSV.exists():
    st.subheader("Recalibration comparison (B-spline)")
    st.caption(
        "Cubic B-spline calibration on logit(p), one GLOBAL curve for all "
        "horizons, fitted on the TRAIN set only. Three capacities shown: "
        "7 / 8 / 9 total parameters = 1 intercept + spline basis of "
        "2 / 3 / 4 inner knots (basis size = knots + degree + 1 = knots + 4). "
        "Knots sit at TRAIN quantiles of logit(p). After per-bucket "
        "recalibration the event is re-normalized before scoring, as in "
        "the price3 section. Delta = raw - recalibrated, positive means "
        "the recalibration improves the score."
    )
    sp_df = pd.read_csv(SPLINE_CSV)
    delta_rows = []
    for n_params in [7, 8, 9]:
        sub = sp_df[sp_df["params"] == n_params]
        deltas = []
        for sn in ["train", "val", "test"]:
            raw = sub[(sub["split"] == sn)
                      & (sub["version"] == "raw")].iloc[0]
            cal = sub[(sub["split"] == sn)
                      & (sub["version"].str.startswith("spline"))].iloc[0]
            deltas.append({
                "split": sn,
                "dBS": float(raw["BS"] - cal["BS"]),
                "dLogL": float(raw["LogLoss"] - cal["LogLoss"]),
                "dRPS": float(raw["RPS"] - cal["RPS"]),
            })
        dd = pd.DataFrame(deltas)
        for row in dd.itertuples():
            delta_rows.append({
                "params": n_params, "split": row.split,
                "dBS": row.dBS, "dLogL": row.dLogL, "dRPS": row.dRPS,
            })

        st.markdown(
            f"**B-spline, {n_params} total parameters "
            f"({n_params - 5} inner knots)**"
        )
        if n_params == 7:
            # 7-parameter version: TEST only, concrete raw vs recalibrated
            # scores (not deltas), uppercase, bold black, centered.
            cell_style = (
                "border:1px solid #d0d0d0;padding:7px 8px;"
                "font-weight:700;color:#000000;"
            )
            head_style = (
                "border:1px solid #d0d0d0;padding:7px 8px;"
                "background:#f3f3f3;font-weight:700;color:#000000;"
            )
            t_raw = sub[(sub["split"] == "test")
                        & (sub["version"] == "raw")].iloc[0]
            t_cal = sub[(sub["split"] == "test")
                        & (sub["version"].str.startswith("spline"))].iloc[0]
            # (name, raw, cal, raw CI lo, raw CI hi, cal CI lo, cal CI hi)
            metric_rows = [
                ("BRIER", t_raw["BS"], t_cal["BS"],
                 t_raw["BS_lo"], t_raw["BS_hi"],
                 t_cal["BS_lo"], t_cal["BS_hi"]),
                ("LOGLOSS", t_raw["LogLoss"], t_cal["LogLoss"],
                 t_raw["LogLoss_lo"], t_raw["LogLoss_hi"],
                 t_cal["LogLoss_lo"], t_cal["LogLoss_hi"]),
                ("RPS", t_raw["RPS"], t_cal["RPS"],
                 t_raw["RPS_lo"], t_raw["RPS_hi"],
                 t_cal["RPS_lo"], t_cal["RPS_hi"]),
            ]
            rows_html = []
            for name, rv, cv, rlo, rhi, clo, chi in metric_rows:
                rows_html.append(
                    f'<tr>'
                    f'<td style="{cell_style}">{name}</td>'
                    f'<td style="{cell_style}">{rv:.4f}</td>'
                    f'<td style="{cell_style}">{rlo:.4f} ~ {rhi:.4f}</td>'
                    f'<td style="{cell_style}">{cv:.4f}</td>'
                    f'<td style="{cell_style}">{clo:.4f} ~ {chi:.4f}</td>'
                    f'<td style="{cell_style}">{rv - cv:+.4f}</td>'
                    f'</tr>'
                )
            table_html = (
                '<table style="width:66%;margin:0 auto;border-collapse:'
                'collapse;text-align:center;font-family:Arial,sans-serif;"'
                ' cellpadding="0" cellspacing="0">'
                '<thead><tr>'
                f'<th style="{head_style}">METRIC</th>'
                f'<th style="{head_style}">RAW</th>'
                f'<th style="{head_style}">RAW 95% CI</th>'
                f'<th style="{head_style}">CALIB.</th>'
                f'<th style="{head_style}">CALIB. 95% CI</th>'
                f'<th style="{head_style}">&Delta;</th>'
                '</tr></thead><tbody>'
                + "".join(rows_html)
                + '</tbody></table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)
            st.caption(
                "Test set (2026-07), event-level full-coverage scoring. "
                "RAW = original market prices; CALIB. = 7-parameter cubic "
                "B-spline (2 inner knots) recalibrated prices. "
                "CIs are event-level bootstrap 95% intervals (2,000 "
                "resamples of whole events). Lower is better. "
                "Delta = RAW - CALIB. (>0 means recalibration improves the "
                "score); the deltas are tiny - the recalibration cannot "
                "improve on the raw market."
            )
            continue
        figs = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Δ Brier (raw - spline)",
                            "Δ LogLoss (raw - spline)",
                            "Δ RPS (raw - spline)"),
            shared_yaxes=False,
            horizontal_spacing=0.16,
        )
        for i, (key, color) in enumerate(
                [("dBS", "#1f6fb4"), ("dLogL", "#c1121f"),
                 ("dRPS", "#2e8b57")], start=1):
            figs.add_trace(go.Bar(
                x=dd["split"], y=dd[key], name=key,
                marker_color=color, marker_line_width=0,
                text=[f"{v:+.4f}" for v in dd[key]],
                textposition="outside",
                textfont=dict(size=13, color="#000"),
                hovertemplate="%{x}: %{y:+.4f}<extra></extra>",
            ), row=1, col=i)
            figs.update_xaxes(
                tickfont=dict(size=13, color="#000",
                              family="Arial, sans-serif", weight=700),
                row=1, col=i, title=dict(text="", font=AXIS_FONT),
            )
            figs.update_yaxes(row=1, col=i,
                              title=dict(text=key, font=AXIS_FONT),
                              tickfont=TICK_FONT)
        figs.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=70, r=30, t=50, b=60),
            plot_bgcolor="white",
        )
        figs.update_annotations(font=dict(color="#000000", size=16,
                                          family="Arial, sans-serif",
                                          weight=700))
        st.plotly_chart(figs, use_container_width=True)

    dsum = pd.DataFrame(delta_rows)
    st.dataframe(dsum.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "All three capacities are statistically indistinguishable: test-set "
        "dRPS = +0.00055 (7), +0.00064 (8), +0.00041 (9) - all inside the "
        "bootstrap CI. Validation improves monotonically with capacity "
        "(overfitting toward the train-adjacent June data) while the "
        "July test set peaks at 8 parameters and falls back - the "
        "bias-variance turning point. Function form and parameter count "
        "are not the bottleneck; the bias is too small to exploit "
        "out-of-sample."
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
