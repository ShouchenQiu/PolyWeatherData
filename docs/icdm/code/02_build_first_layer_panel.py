"""Step 2: build the first-layer (binary contract) price panel.

For each node, load trades/markets/events, keep events with
target_date <= 2026-06-30 (training+validation history only), exclude
unresolved markets, and extract a YES-view price at fixed horizons before
settlement: 48h, 36h, 24h, 12h, 6h, 1h.

Price definition: nearest trade in window [h, h+3h] before close, YES-view
price (price if outcome==YES else 1-price). NaN if no trade in window.

Outputs (all inside docs/icdm/output/):
  first_layer_panel.parquet   -- one row per (event, condition)
  panel_quality_report.txt    -- per-node coverage summary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLOB = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob")
OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
HORIZONS = [48.0, 36.0, 24.0, 12.0, 6.0, 1.0]
HORIZON_WIN_H = 3.0  # tolerate sparse trading with a 3h window
CUTOFF = "2026-06-30"

# node -> (station_dir, artifact_dir)
NODES: dict[str, tuple[str, str]] = {
    "NYC": ("KLGA", "20250401_20260331_resolution_v2"),
    "NYC_Q2": ("KLGA", "20260401_20260630"),
    "LONDON": ("LONDON", "20250121_20260810"),
    "CHICAGO": ("CHICAGO", "20251204_20260810"),
    "DALLAS": ("DALLAS", "20250101_20260812"),
    "MIAMI": ("MIAMI", "20250101_20260812"),
    "DENVER": ("DENVER", "20250101_20260812"),
    "SEATTLE": ("SEATTLE", "20250101_20260812"),
    "ATLANTA": ("ATLANTA", "20250101_20260812"),
    "TOKYO": ("TOKYO", "20250101_20260818"),
    "SHANGHAI": ("SHANGHAI", "20250101_20260818"),
    "SEOUL": ("SEOUL", "20250101_20260818"),
    "TORONTO": ("TORONTO", "20250101_20260818"),
}


def load_node(node: str) -> pd.DataFrame:
    station, artifact = NODES[node]
    root = CLOB / station / artifact

    tr = pd.read_parquet(
        root / "trades.parquet",
        columns=["condition_id", "timestamp", "outcome", "side", "price"],
    )
    ev = pd.read_parquet(
        root / "events.parquet",
        columns=["event_id", "target_date"],
    )
    # Read base market columns; resolved_yes handled per node below
    # (NYC_Q2 lacks it in the parquet, so we read it without that column).
    base_mk_cols = [
        "event_id", "condition_id", "bucket_lo_f", "bucket_hi_f", "closed_at",
    ]
    available_mk_cols = pd.read_parquet(
        root / "markets.parquet", columns=None
    ).columns.tolist()
    mk_cols = [c for c in base_mk_cols if c in available_mk_cols]
    if "resolved_yes" in available_mk_cols:
        mk_cols.append("resolved_yes")
    mk = pd.read_parquet(root / "markets.parquet", columns=mk_cols)

    # NYC_Q2 has no resolved_yes in parquet: merge backfilled resolution table
    if node == "NYC_Q2":
        res = pd.read_parquet(OUT_DIR / "klga_q2_resolution.parquet")
        mk = mk.merge(res, on="condition_id", how="left")
        print(
            f"[note] NYC_Q2 resolution merged: "
            f"true={int((mk['resolved_yes']==True).sum())} "  # noqa: E712
            f"false={int((mk['resolved_yes']==False).sum())} "  # noqa: E712
            f"null={int(mk['resolved_yes'].isna().sum())}",
            flush=True,
        )

    # keep history only
    ev = ev[pd.to_datetime(ev["target_date"]) <= pd.to_datetime(CUTOFF)]
    keep_events = set(ev["event_id"])
    mk = mk[mk["event_id"].isin(keep_events)].copy()

    # exclude unresolved
    n_unresolved = int(mk["resolved_yes"].isna().sum())
    mk = mk[mk["resolved_yes"].notna()].copy()

    tr = tr.merge(mk[["condition_id"]], on="condition_id", how="inner")
    tr["yes_view"] = tr["price"].where(
        tr["outcome"] == "YES", 1.0 - tr["price"]
    )
    tr = tr.merge(mk[["condition_id", "closed_at"]], on="condition_id")
    tr["hours_before_close"] = (
        tr["closed_at"] - tr["timestamp"]
    ).dt.total_seconds() / 3600.0

    # build one row per (event, condition) with horizon prices
    # vectorized: sort trades by hours_before_close, then take nearest
    # trade per (condition, horizon window) without per-row scans.
    tr_sorted = tr.sort_values(["condition_id", "hours_before_close"])

    # for each horizon, mark trades falling in [h, h+win] and take first
    # (i.e. smallest hours_before_close) per condition
    price_cols: dict[str, pd.Series] = {}
    for h in HORIZONS:
        lo, hi = h, h + HORIZON_WIN_H
        mask = (
            (tr_sorted["hours_before_close"] >= lo)
            & (tr_sorted["hours_before_close"] <= hi)
        )
        near = tr_sorted.loc[mask].groupby("condition_id")["yes_view"].first()
        price_cols[f"p_{int(h)}h"] = near

    mk = mk.set_index("condition_id")
    panel = mk[["event_id", "bucket_lo_f", "bucket_hi_f", "resolved_yes"]].copy()
    panel["city"] = node
    panel["y"] = panel["resolved_yes"].astype(int)
    panel = panel.drop(columns=["resolved_yes"])
    for col, series in price_cols.items():
        panel[col] = panel.index.map(series.get).astype(float)
    panel = panel.reset_index()

    print(
        f"[loaded] {node}: events={panel['event_id'].nunique()} "
        f"conditions={len(panel)} unresolved_dropped={n_unresolved}",
        flush=True,
    )
    return panel


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for node in NODES:
        frames.append(load_node(node))
    panel = pd.concat(frames, ignore_index=True)

    # sanity: each event must have exactly one winner
    winners = panel[panel["y"] == 1].groupby("event_id").size()
    bad = int((winners != 1).sum())
    print(
        f"\n[sanity] events={panel['event_id'].nunique()} "
        f"conditions={len(panel)} events_without_exactly_one_winner={bad}",
        flush=True,
    )

    panel.to_parquet(OUT_DIR / "first_layer_panel.parquet", index=False)

    # quality report
    rep: list[str] = ["Panel coverage report", "=" * 40]
    for node, sub in panel.groupby("city"):
        n_ev = sub["event_id"].nunique()
        n_cond = len(sub)
        rep.append(f"\n{node}: events={n_ev} conditions={n_cond}")
        for h in HORIZONS:
            col = f"p_{int(h)}h"
            cov = sub[col].notna().mean()
            rep.append(f"  {col}: coverage={cov:.1%}")
    report = "\n".join(rep)
    (OUT_DIR / "panel_quality_report.txt").write_text(
        report, encoding="utf-8"
    )

    print("\n" + report, flush=True)
    print(
        f"\n[done] panel written to {OUT_DIR / 'first_layer_panel.parquet'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
