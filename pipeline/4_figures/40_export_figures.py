"""Generate high-res PNGs of the three key UI figures for the paper.

Figures (standalone, NOT replacing the UI):
  1. Scores vs time to high (combined, normalized to 48h = 1)
     (Event-Distribution Layer page)
  2. Combined D_h(0.10) vs D_h(0.75~0.95) (dual axes)
     (First-Layer Aligned-Dates page)
  3. Reliability diagram: all horizons overlaid
     (First-Layer Aligned-Dates page)

Logic mirrors ui/pages/*.py exactly (event-weighted, excludes
DENVER/DALLAS, 30-min argmax panel). Output PNGs are written to
docs/icdm/output/.
"""
from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR, PANEL  # noqa: E402
PANEL = OUT_DIR / "first_layer_panel_argmax_30min.parquet"
CSV_SUM = OUT_DIR / "event_layer_scores.csv"
HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
OUTLIERS = ["DENVER", "DALLAS"]

AXIS_FONT = dict(size=18, color="#000000", family="Arial, sans-serif",
                 weight=700)
TICK_FONT = dict(size=16, color="#000000", family="Arial, sans-serif",
                 weight=700)
LEGEND_FONT = dict(size=16, color="#000000", family="Arial, sans-serif",
                   weight=700)
ZERO_LINE = dict(color="#666666", width=1.2, dash="dot")


def event_avg(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    return sub.groupby("event_id").agg(mp=(col, "mean"), my=("y", "mean"))


def dh_one(col: str, sub: pd.DataFrame, equal_min_events: int = 5) -> dict:
    if len(sub) == 0:
        return {"event": np.nan, "equal": np.nan, "n_events": 0}
    ev = event_avg(sub, col)
    ev_dh = float(ev["my"].mean() - ev["mp"].mean())
    eq_vals, st_names = [], []
    for _, csub in sub.groupby("city"):
        e2 = event_avg(csub, col)
        if len(e2) < equal_min_events:
            continue
        eq_vals.append(float(e2["my"].mean() - e2["mp"].mean()))
        st_names.append(csub["city"].iloc[0])
    eq_dh = float(np.mean(eq_vals)) if eq_vals else np.nan
    return {"event": ev_dh, "equal": eq_dh, "n_events": int(len(ev)),
            "n_stations": len(st_names)}


def dh_range(panel: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        col = f"p_{int(h)}h"
        sub = panel[(panel[col].notna()) & (panel[col] >= lo)
                    & (panel[col] <= hi)].copy()
        r = dh_one(col, sub)
        rows.append({"horizon_h": int(h), **r})
    return pd.DataFrame(rows)


def reliability_points(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    BINS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.01]
    BIN_LABELS = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-30%",
                  "30-50%", "50-70%", "70-90%", "90-100%"]
    df = df[df[pcol].notna()].copy()
    df["bin"] = pd.cut(df[pcol], bins=BINS, labels=BIN_LABELS,
                       include_lowest=True)
    ev = df.groupby(["event_id", "bin"], observed=True).agg(
        p=(pcol, "mean"), y=("y", "mean")).reset_index()
    agg = ev.groupby("bin", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean"),
        n_events=("event_id", "nunique")).reset_index()
    return agg


def main() -> int:
    panel = pd.read_parquet(PANEL)
    panel = panel[~panel["city"].isin(OUTLIERS)].reset_index(drop=True)
    print(f"panel: {panel['event_id'].nunique()} events | "
          f"{panel['city'].nunique()} cities", flush=True)

    dh = dh_range(panel, 0.0, 0.10)
    dh_hi = dh_range(panel, 0.75, 0.95)

    # ---- Figure 2: combined D_h dual axes -----------------------------
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=dh["horizon_h"], y=dh["event"], mode="lines+markers",
        name="D_h(0.10) event-wtd", line=dict(color="#1f6fb4", width=3),
        marker=dict(size=9), yaxis="y",
        hovertemplate="D_h(0.10) event: %{y:+.4f}<extra></extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=dh["horizon_h"], y=dh["equal"], mode="lines+markers",
        name="D_h(0.10) station-eq", line=dict(color="#8e44ad", width=3,
                                               dash="dash"),
        marker=dict(size=9), yaxis="y",
        hovertemplate="D_h(0.10) equal: %{y:+.4f}<extra></extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=dh_hi["horizon_h"], y=dh_hi["event"], mode="lines+markers",
        name="D_h(0.75~0.95) event-wtd", line=dict(color="#c1121f", width=3),
        marker=dict(size=9), yaxis="y2",
        hovertemplate="D_h(0.75~0.95) event: %{y:+.4f}<extra></extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=dh_hi["horizon_h"], y=dh_hi["equal"], mode="lines+markers",
        name="D_h(0.75~0.95) station-eq", line=dict(color="#f39c12", width=3,
                                                    dash="dash"),
        marker=dict(size=9), yaxis="y2",
        hovertemplate="D_h(0.75~0.95) equal: %{y:+.4f}<extra></extra>",
    ))
    fig4.add_hline(y=0, line_dash=ZERO_LINE["dash"], line_color=ZERO_LINE["color"],
                   line_width=ZERO_LINE["width"], yref="y")
    fig4.add_hline(y=0, line_dash=ZERO_LINE["dash"], line_color="#999999",
                   line_width=1, yref="y2")
    fig4.update_layout(
        height=560,
        xaxis=dict(
            tickmode="array", tickvals=HORIZONS,
            ticktext=[f"{int(h)}h" for h in HORIZONS],
            title=dict(text="Hours Before The Highest Temperature",
                       font=AXIS_FONT),
            tickfont=TICK_FONT, showgrid=True, gridcolor="#E0E0E0",
        ),
        yaxis=dict(title=dict(text="D_h(0.10)", font=AXIS_FONT),
                   tickfont=TICK_FONT, showgrid=True, gridcolor="#E0E0E0"),
        yaxis2=dict(title=dict(text="D_h(0.75~0.95)", font=AXIS_FONT),
                    tickfont=TICK_FONT, overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0,
                    font=LEGEND_FONT),
        margin=dict(l=90, r=90, t=70, b=70),
        plot_bgcolor="white",
    )
    fig4.write_image(OUT_DIR / "fig2_combined_dh_dual_axes.png",
                     scale=3, width=1400, height=560)
    print("[done] fig2_combined_dh_dual_axes.png", flush=True)

    # ---- Figure 3: reliability diagram --------------------------------
    fig3 = go.Figure()
    rel_colors = sample_colorscale("Blues", len(HORIZONS))
    for i, h in enumerate(HORIZONS):
        col = f"p_{int(h)}h"
        rp = reliability_points(panel, col)
        rp = rp[rp["n_events"] >= 30]
        fig3.add_trace(go.Scatter(
            x=rp["mean_p"], y=rp["mean_y"], mode="lines+markers",
            name=f"{int(h)}h", line=dict(color=rel_colors[i], width=1.6),
            marker=dict(size=6),
            hovertemplate="%{x:.3f} -> %{y:.3f}<extra></extra>",
        ))
    fig3.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#E63946", dash="dash", width=3.5),
        name="Perfect calibration", hoverinfo="skip",
    ))
    fig3.update_layout(
        height=640,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(size=13, color="#000000", family="Arial, sans-serif",
                      weight=700),
        ),
        xaxis=dict(
            title=dict(text="Quoted price (bin mean)", font=AXIS_FONT),
            range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            tickfont=TICK_FONT, showgrid=True, gridcolor="#E0E0E0",
        ),
        yaxis=dict(
            title=dict(text="Realized frequency (bin mean)", font=AXIS_FONT),
            range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            tickfont=TICK_FONT, showgrid=True, gridcolor="#E0E0E0",
        ),
        margin=dict(l=90, r=50, t=90, b=70),
        plot_bgcolor="white",
    )
    fig3.write_image(OUT_DIR / "fig3_reliability_all_horizons.png",
                     scale=3, width=1400, height=640)
    print("[done] fig3_reliability_all_horizons.png", flush=True)

    # ---- Figure 1: combined normalized event-layer scores --------------
    summ = pd.read_csv(CSV_SUM).sort_values("horizon_h")
    base = summ[summ["horizon_h"] == 48].iloc[0]
    norm = summ.copy()
    for key, ci_key in [("BS", "BS"), ("LogLoss", "LogL"), ("RPS", "RPS")]:
        b = base[key]
        norm[f"{key}_n"] = summ[key] / b
        norm[f"{key}_lo_n"] = summ[f"{ci_key}_ci_lo"] / b
        norm[f"{key}_hi_n"] = summ[f"{ci_key}_ci_hi"] / b

    fig1 = go.Figure()
    for key, title, color, ci_key in [
        ("BS", "Brier", "#1f6fb4", "BS"),
        ("LogLoss", "Log Loss", "#c1121f", "LogL"),
        ("RPS", "RPS", "#2e8b57", "RPS"),
    ]:
        y = norm[f"{key}_n"]
        fig1.add_trace(go.Scatter(
            x=norm["horizon_h"], y=y, mode="lines+markers", name=title,
            line=dict(color=color, width=3), marker=dict(size=10),
            error_y=dict(
                type="data", symmetric=False,
                array=norm[f"{key}_hi_n"] - y,
                arrayminus=y - norm[f"{key}_lo_n"],
                visible=True, thickness=1.5, color=color,
            ),
            hovertemplate="%{x:.0f}h: %{y:.4f}<extra></extra>",
        ))
    fig1.add_hline(y=1.0, line_dash="dash", line_color="#888888",
                   line_width=1.2, annotation_text="48h baseline",
                   annotation_font=dict(size=14, color="#666666"))
    fig1.update_xaxes(
        tickmode="array", tickvals=norm["horizon_h"],
        ticktext=[f"{int(h)}h" for h in norm["horizon_h"]],
        title=dict(text="Hours Before The Highest Temperature",
                   font=AXIS_FONT),
        tickfont=TICK_FONT,
    )
    fig1.update_yaxes(title=dict(text="Relative Score", font=AXIS_FONT),
                      tickfont=TICK_FONT)
    fig1.update_layout(
        height=500,
        margin=dict(l=90, r=50, t=70, b=70),
        plot_bgcolor="white",
        legend=dict(font=dict(size=16, color="#000000",
                              family="Arial, sans-serif", weight=700),
                    orientation="h", yanchor="bottom", y=1.05,
                    xanchor="right", x=1),
    )
    fig1.write_image(OUT_DIR / "fig1_scores_combined_normalized_48h.png",
                     scale=3, width=1400, height=500)
    print("[done] fig1_scores_combined_normalized_48h.png", flush=True)

    # ---- Figure 4: bias heatmap (D per price bin x horizon) ------------
    BINS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.01]
    BIN_LABELS = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-30%",
                  "30-50%", "50-70%", "70-90%", "90-100%"]
    Dm = np.full((len(BIN_LABELS), len(HORIZONS)), np.nan)
    Nm = np.zeros((len(BIN_LABELS), len(HORIZONS)), dtype=int)
    Sm = np.full((len(BIN_LABELS), len(HORIZONS)), np.nan)
    for j, h in enumerate(HORIZONS):
        col = f"p_{int(h)}h"
        dd = panel[panel[col].notna()].copy()
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

    vmax = float(np.nanmax(np.abs(Dm)))
    ann = np.empty_like(Dm, dtype=object)
    for i in range(Dm.shape[0]):
        for j in range(Dm.shape[1]):
            if np.isnan(Dm[i, j]):
                ann[i, j] = ""
                continue
            star = ("*" if (np.isfinite(Sm[i, j])
                            and abs(Dm[i, j]) > 2 * Sm[i, j]) else "")
            ann[i, j] = f"{Dm[i, j]:+.1f}{star}"
    fig4 = go.Figure(go.Heatmap(
        z=Dm, x=[f"{int(h)}h" for h in HORIZONS], y=BIN_LABELS,
        zmin=-vmax, zmax=vmax, zmid=0,
        colorscale=[[0.00, "#4575b4"], [0.25, "#abd0e6"],
                    [0.50, "#ffffff"], [0.75, "#fdae61"],
                    [1.00, "#d73027"]],
        text=ann, texttemplate="%{text}",
        textfont=dict(size=11, color="#111111",
                      family="Arial, sans-serif", weight=700),
        colorbar=dict(title=dict(text="D (pp)", font=AXIS_FONT),
                      tickfont=TICK_FONT,
                      tickvals=[-vmax, 0, vmax],
                      ticktext=[f"{-vmax:.1f}", "0", f"{vmax:.1f}"]),
        hovertemplate=("bin %{y}, %{x}<br>D = %{z:+.2f} pp<extra></extra>"),
    ))
    fig4.update_layout(
        height=760, width=1400,
        xaxis=dict(title=dict(text="Hours Before The Highest Temperature",
                              font=AXIS_FONT), tickfont=TICK_FONT),
        yaxis=dict(title=dict(text="Quoted price bin", font=AXIS_FONT),
                   tickfont=TICK_FONT, autorange="reversed"),
        margin=dict(l=110, r=50, t=50, b=170),
        plot_bgcolor="white",
        annotations=[
            dict(x=0.5, y=-0.185, xref="paper", yref="paper",
                 xanchor="center", yanchor="top", showarrow=False,
                 align="center",
                 text=("Cell text = D in percentage points (realized "
                       "frequency - quoted price).<br>"
                       "* = |D| exceeds 2 standard errors across events "
                       "(~95%): distinguishable from 0.  Unmarked cells "
                       "are only CONSISTENT with zero.<br>"
                       "BLUE = negative D (market OVERprices)  |  WHITE "
                       "= no bias  |  RED = positive D (market "
                       "UNDERprices);  colour depth = |D|."),
                 font=dict(size=14, color="#111111",
                           family="Arial, sans-serif", weight=700)),
        ],
    )
    fig4.write_image(OUT_DIR / "fig4_bias_heatmap_bin_x_horizon.png",
                     scale=3, width=1400, height=620)
    print("[done] fig4_bias_heatmap_bin_x_horizon.png", flush=True)

    print("\nAll four PNGs written to", OUT_DIR, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
