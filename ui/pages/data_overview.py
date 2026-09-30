"""Data Overview page - show current CLOB data status across stations.

Shows only the canonical artifacts (excludes KLGA legacy versions and
Phoenix which has only 2 events). Includes per-station trade counts,
date coverage, resolution status, and download progress for pending
stations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import CLOB_ROOT, OUT_DIR  # noqa: E402

BASE = CLOB_ROOT

# Canonical artifacts to display (exclude legacy KLGA versions and Phoenix).
# All nodes share the integrated window artifact 20251205_20260810.
CANONICAL = {
    "NYC": ("KLGA", "20251205_20260810"),
    "LONDON": ("LONDON", "20251205_20260810"),
    "CHICAGO": ("CHICAGO", "20251205_20260810"),
    "DALLAS": ("DALLAS", "20251205_20260810"),
    "MIAMI": ("MIAMI", "20251205_20260810"),
    "DENVER": ("DENVER", "20251205_20260810"),
    "SEATTLE": ("SEATTLE", "20251205_20260810"),
    "ATLANTA": ("ATLANTA", "20251205_20260810"),
    "SEOUL": ("SEOUL", "20251205_20260810"),
    "TORONTO": ("TORONTO", "20251205_20260810"),
    "TOKYO": ("TOKYO", "20251205_20260810"),
    "SHANGHAI": ("SHANGHAI", "20251205_20260810"),
}

st.title(":bar_chart: Data Overview")

st.markdown(
    """
    Current CLOB dataset status across stations.
    *Excludes KLGA legacy versions and Phoenix (only 2 events).*
    """
)


def load_station_manifest(station: str, artifact: str) -> dict | None:
    mf = BASE / station / artifact / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


rows: list[dict] = []
for label, (station, artifact) in CANONICAL.items():
    m = load_station_manifest(station, artifact)
    if m is None:
        # check raw-cache for download progress
        cache = BASE / station / f"{artifact}.raw-cache" / "trades"
        n_cache = len(list(cache.glob("*.json"))) if cache.exists() else 0
        rows.append({
            "label": label,
            "station": station,
            "events": None,
            "markets": None,
            "trades": None,
            "start": None,
            "end": None,
            "resolved_pct": None,
            "status": f"downloading ({n_cache} markets cached)",
        })
        continue
    rc = m.get("row_counts", {})
    rng = m.get("requested_range", [None, None])
    rows.append({
        "label": label,
        "station": station,
        "events": rc.get("events"),
        "markets": rc.get("markets"),
        "trades": rc.get("trades"),
        "start": rng[0] if rng else None,
        "end": rng[1] if rng else None,
        "resolved_pct": None,
        "status": "complete",
    })

df = pd.DataFrame(rows)

# fill resolved% from markets if available
for i, r in df.iterrows():
    if r["status"] != "complete":
        continue
    mk = BASE / r["station"] / CANONICAL[r["label"]][1] / "markets.parquet"
    if mk.exists():
        try:
            cols = pd.read_parquet(mk, columns=None).columns.tolist()
            if "resolved_yes" in cols:
                mdf = pd.read_parquet(mk, columns=["resolved_yes"])
                df.at[i, "resolved_pct"] = round(mdf["resolved_yes"].notna().mean() * 100, 1)
            else:
                df.at[i, "resolved_pct"] = None
        except Exception:
            df.at[i, "resolved_pct"] = None

st.subheader("Station summary")
# Exclude DALLAS / DENVER (excluded from the study), drop the date range,
# resolved% and status columns. ONE merged table, two cities per row
# (8 columns in pairs), original st.dataframe format, text centered.
active = df[~df["station"].isin(["DALLAS", "DENVER"])].copy()
active["label"] = active["label"].replace({"NYC": "New York City"})
disp = active[["label", "events", "markets", "trades"]].copy()
disp.columns = ["City", "Events", "Markets", "Trades"]
disp["Trades"] = disp["Trades"].map(
    lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
# uppercase every label and sort alphabetically by city name
disp["City"] = disp["City"].str.upper()
disp = disp.sort_values("City").reset_index(drop=True)

left_df = disp.iloc[0::2].reset_index(drop=True)
right_df = disp.iloc[1::2].reset_index(drop=True)
if len(left_df) > len(right_df):
    right_df = pd.concat([right_df, pd.DataFrame(
        [["", "", "", ""]], columns=disp.columns)], ignore_index=True)
merged = pd.concat([left_df, right_df], axis=1, ignore_index=True)
# single-level flat header (drop the City A / City B group row).
# Column names must be unique for pyarrow; the second city block uses a
# trailing space (renders visually identical to the first block).
merged.columns = [
    "CITY", "EVENTS", "MARKETS", "TRADES",
    "CITY\u200b", "EVENTS\u200b", "MARKETS\u200b", "TRADES\u200b",
]
# st.dataframe renders via Glide Data Grid (canvas), which ignores CSS
# text-align; st.table renders a real HTML table, so centering works.
# Streamlit forces inline text-align:left on cells, so we override with
# !important CSS on the table's own selector.
styler = merged.style.set_table_styles([{
    "selector": "th, td",
    "props": [("border", "1px solid #d0d0d0")],
}, {
    "selector": "th",
    "props": [("background-color", "#f3f3f3"),
              ("font-weight", "700"),
              ("color", "#000000"),
              ("font-family", "Arial, sans-serif")],
}, {
    "selector": "td",
    "props": [("text-align", "center !important"),
              ("padding", "6px 10px"),
              ("font-weight", "700 !important"),
              ("color", "#000000 !important"),
              ("font-family", "Arial, sans-serif")],
}, {
    "selector": "th",
    "props": [("text-align", "center !important"),
              ("padding", "6px 10px")],
}])
st.table(styler)

# --- raw order statistics (from step-8 CSV if present) ---
raw_csv = OUT_DIR / "raw_trade_stats.csv"
if raw_csv.exists():
    st.subheader("Raw order statistics")
    raw = pd.read_csv(raw_csv)
    raw_display = raw[[
        "city", "events", "conditions", "trades",
        "date_min", "date_max",
    ]].copy()
    # resolved % from the artifact markets directly (same as station summary)
    resolved = []
    for _, r in raw.iterrows():
        mk = BASE / r["station"] / CANONICAL[r["city"]][1] / "markets.parquet"
        try:
            cols = pd.read_parquet(mk, columns=None).columns.tolist()
            if "resolved_yes" in cols:
                mdf = pd.read_parquet(mk, columns=["resolved_yes"])
                resolved.append(round(mdf["resolved_yes"].notna().mean() * 100, 1))
            else:
                resolved.append(None)
        except Exception:
            resolved.append(None)
    raw_display["resolved_pct"] = resolved
    raw_display.columns = [
        "City", "Events", "Markets", "Trades", "From", "To", "Resolved %",
    ]
    raw_display["Trades"] = raw_display["Trades"].map(lambda x: f"{x:,.0f}")
    raw_display["Resolved %"] = raw_display["Resolved %"].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    raw_display["Time range"] = (
        raw_display["From"].astype(str) + " ~ " + raw_display["To"].astype(str)
    )
    raw_display = raw_display[["City", "Events", "Markets", "Trades", "Time range", "Resolved %"]]
    st.dataframe(raw_display, use_container_width=True, hide_index=True)
    tot = raw[["events", "conditions", "trades"]].sum()
    st.caption(
        f"TOTAL: {int(tot['events']):,} events | {int(tot['conditions']):,} markets | "
        f"{int(tot['trades']):,} trades | Source: {raw_csv.name}"
    )

# --- charts ---
done = df[df["status"] == "complete"].copy()
if len(done) > 0:
    st.subheader("Trades per station")
    fig = px.bar(
        done.sort_values("trades"),
        x="trades", y="label", orientation="h",
        labels={"trades": "Trades", "label": ""},
        color="trades", color_continuous_scale="Blues",
    )
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Coverage timeline")
    timeline_rows = []
    for _, r in done.iterrows():
        timeline_rows.append({
            "label": r["label"],
            "start": pd.to_datetime(r["start"]),
            "end": pd.to_datetime(r["end"]),
            "events": r["events"],
        })
    tdf = pd.DataFrame(timeline_rows)
    fig2 = go.Figure()
    for _, r in tdf.iterrows():
        fig2.add_trace(go.Bar(
            name=r["label"], x=[r["end"]],
            y=[r["label"]], orientation="h", base=r["start"],
            text=f"{r['events']} events",
            textposition="outside",
        ))
    fig2.update_layout(
        height=450,
        title="Data coverage by station",
        xaxis_title="Date",
        barmode="overlay",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # events per station
    st.subheader("Events per station")
    fig3 = px.bar(
        done.sort_values("events"),
        x="events", y="label", orientation="h",
        labels={"events": "Events", "label": ""},
        color="events", color_continuous_scale="Greens",
    )
    fig3.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Event lead time distribution ----
    st.subheader("Event lead time (open -> target date)")
    lead_rows = []
    for _, r in done.iterrows():
        root = BASE / r["station"] / CANONICAL[r["label"]][1]
        try:
            mk = pd.read_parquet(root / "markets.parquet", columns=["event_id", "market_start_at"])
            ev = pd.read_parquet(root / "events.parquet", columns=["event_id", "event_end_at"])
            m = mk.merge(ev, on="event_id")
            ev_start = m.groupby("event_id")["market_start_at"].min()
            ev_end = m.drop_duplicates("event_id").set_index("event_id")["event_end_at"]
            lead_h = (ev_end - ev_start).dt.total_seconds() / 3600
            for _, v in lead_h.items():
                lead_rows.append({"label": r["label"], "lead_hours": v})
        except Exception:
            pass
    if lead_rows:
        ldf = pd.DataFrame(lead_rows)
        st.markdown(
            "Hours between the market opening and the event target time "
            "(how far in advance these daily temperature events are listed)."
        )
        # histogram by station (box + violin)
        fig4 = px.box(
            ldf, x="label", y="lead_hours", color="label",
            labels={"lead_hours": "Lead time (hours)", "label": ""},
            points="outliers",
        )
        fig4.update_layout(height=500, showlegend=False,
                           xaxis_title="", yaxis_title="Lead time (hours)")
        st.plotly_chart(fig4, use_container_width=True)

        # overall histogram
        fig5 = px.histogram(
            ldf, x="lead_hours", nbins=40,
            labels={"lead_hours": "Lead time (hours)", "count": "Events"},
            color_discrete_sequence=["#4C72B0"],
        )
        fig5.add_vline(x=ldf["lead_hours"].median(), line_dash="dash",
                       annotation_text=f"median {ldf['lead_hours'].median():.0f}h")
        fig5.update_layout(height=400, showlegend=False,
                           yaxis_title="Events")
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown(
            f"**Summary:** median lead time **{ldf['lead_hours'].median():.0f}h** "
            f"across {len(ldf)} events | "
            f"P25 **{ldf['lead_hours'].quantile(0.25):.0f}h** | "
            f"P75 **{ldf['lead_hours'].quantile(0.75):.0f}h** | "
            f"max **{ldf['lead_hours'].max():.0f}h**"
        )

st.caption(f"Generated from {BASE}")
