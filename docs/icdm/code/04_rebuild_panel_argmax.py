"""Step 4: rebuild the first-layer panel using high-temperature argmax time.

Previous panel used hours_before_close (actual settlement). Problem: for
events settling same-evening, the daily high is already known at p_1h/p_6h,
so those prices are not genuine predictions.

This version uses "hours_before_argmax" = time from trade to the moment the
day's high temperature occurred (argmax_utc). Only trades BEFORE the high
occurred are genuine predictions; trades after argmax are "post-reveal" and
excluded from horizon pricing (priced as NaN when the horizon falls after
argmax).

Output: docs/icdm/output/first_layer_panel_argmax.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLOB = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob")
OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
ARGMAX_FILE = OUT_DIR / "daily_high_argmax.parquet"
HORIZONS = [48.0, 36.0, 24.0, 12.0, 6.0, 1.0]
HORIZON_WIN_H = 3.0
CUTOFF = "2026-06-30"

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

# city -> argmax station key
CITY_TO_STATION = {
    "NYC": "KLGA", "NYC_Q2": "KLGA", "LONDON": "LONDON",
    "CHICAGO": "KORD", "DALLAS": "KDFW", "MIAMI": "KMIA",
    "DENVER": "KDEN", "SEATTLE": "KSEA", "ATLANTA": "KATL",
    "TOKYO": "TOKYO", "SHANGHAI": "SHANGHAI",
    "SEOUL": "SEOUL", "TORONTO": "TORONTO",
}


def load_node(node: str, argmax: pd.DataFrame) -> pd.DataFrame:
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
    available_mk_cols = pd.read_parquet(
        root / "markets.parquet", columns=None
    ).columns.tolist()
    mk_cols = ["event_id", "condition_id", "bucket_lo_f", "bucket_hi_f"]
    if "resolved_yes" in available_mk_cols:
        mk_cols.append("resolved_yes")
    mk = pd.read_parquet(root / "markets.parquet", columns=mk_cols)

    if node == "NYC_Q2":
        res = pd.read_parquet(OUT_DIR / "klga_q2_resolution.parquet")
        mk = mk.merge(res, on="condition_id", how="left")

    ev = ev[pd.to_datetime(ev["target_date"]) <= pd.to_datetime(CUTOFF)]
    keep_events = set(ev["event_id"])
    mk = mk[mk["event_id"].isin(keep_events)].copy()
    n_unresolved = int(mk["resolved_yes"].isna().sum())
    mk = mk[mk["resolved_yes"].notna()].copy()

    # join argmax time per event
    station_key = CITY_TO_STATION[node]
    am = argmax[argmax["station"] == station_key][
        ["target_date", "argmax_utc"]
    ].drop_duplicates("target_date")
    ev = ev.merge(am, on="target_date", how="left")
    mk = mk.merge(ev[["event_id", "argmax_utc"]], on="event_id", how="left")

    # only events with argmax data
    mk = mk[mk["argmax_utc"].notna()].copy()

    tr = tr.merge(mk[["condition_id"]], on="condition_id", how="inner")
    tr["yes_view"] = tr["price"].where(tr["outcome"] == "YES", 1.0 - tr["price"])
    tr = tr.merge(mk[["condition_id", "argmax_utc"]], on="condition_id")
    tr["hours_before_argmax"] = (
        tr["argmax_utc"] - tr["timestamp"]
    ).dt.total_seconds() / 3600.0

    # horizon pricing: only trades strictly BEFORE the high occurred count;
    # if the horizon window extends past argmax, price is NaN (post-reveal).
    tr_sorted = tr.sort_values(["condition_id", "hours_before_argmax"])
    price_cols: dict[str, pd.Series] = {}
    for h in HORIZONS:
        lo, hi = h, h + HORIZON_WIN_H
        # window must lie entirely before argmax: hi <= 0 means trade is at
        # least h hours before the high and at most h+3h before it
        mask = (
            (tr_sorted["hours_before_argmax"] >= lo)
            & (tr_sorted["hours_before_argmax"] <= hi)
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
        f"conditions={len(panel)} unresolved={n_unresolved} "
        f"with_argmax={panel['event_id'].nunique()}",
        flush=True,
    )
    return panel


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    argmax = pd.read_parquet(ARGMAX_FILE)
    frames = [load_node(node, argmax) for node in NODES]
    panel = pd.concat(frames, ignore_index=True)

    winners = panel[panel["y"] == 1].groupby("event_id").size()
    bad = int((winners != 1).sum())
    print(
        f"\n[sanity] events={panel['event_id'].nunique()} "
        f"conditions={len(panel)} bad_winner={bad}",
        flush=True,
    )
    panel.to_parquet(OUT_DIR / "first_layer_panel_argmax.parquet", index=False)

    # quality report
    rep = ["Panel (argmax-based) coverage report", "=" * 45]
    for node, sub in panel.groupby("city"):
        n_ev = sub["event_id"].nunique()
        rep.append(f"\n{node}: events={n_ev} conditions={len(sub)}")
        for h in HORIZONS:
            col = f"p_{int(h)}h"
            rep.append(f"  {col}: coverage={sub[col].notna().mean():.1%}")
    report = "\n".join(rep)
    (OUT_DIR / "panel_argmax_quality_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(
        f"\n[done] wrote {OUT_DIR / 'first_layer_panel_argmax.parquet'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
