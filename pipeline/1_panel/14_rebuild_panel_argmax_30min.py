"""Build the canonical price panel: MEAN price over a +/-30min window.

This is the panel every downstream step consumes.  For each
(event, temperature bucket, horizon) it stores the mean YES-view price of all
taker trades inside ``[h - 0.5h, h + 0.5h]`` before the daily temperature peak
(the argmax time), plus the realized outcome ``y``.

Averaging all trades inside the window (rather than taking the single nearest
trade) increases per-bucket sample size and smooths single-trade noise.

Input:  <CLOB_ROOT>/<STATION>/<ARTIFACT>/ and daily_high_argmax.parquet
Output: first_layer_panel_argmax_30min.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ARGMAX_FILE, CLOB_ROOT, OUT_DIR, ensure_out_dir  # noqa: E402
from constants import ARTIFACT, CUTOFF, HORIZONS, NODES, START_DATE, WIN_HALF_H  # noqa: E402

CLOB = CLOB_ROOT

# (city, station) pairs for this build step.
NODE_DIRS: dict[str, tuple[str, str]] = {
    city: (station, ARTIFACT) for city, station in NODES.items()
}

CITY_TO_STATION = {
    "NYC": "KLGA", "LONDON": "LONDON",
    "CHICAGO": "KORD", "DALLAS": "KDFW", "MIAMI": "KMIA",
    "DENVER": "KDEN", "SEATTLE": "KSEA", "ATLANTA": "KATL",
    "TOKYO": "TOKYO", "SHANGHAI": "SHANGHAI",
    "SEOUL": "SEOUL", "TORONTO": "TORONTO",
}


def load_node(node: str, argmax: pd.DataFrame) -> pd.DataFrame:
    station, artifact = NODE_DIRS[node]
    root = CLOB / station / artifact

    tr = pd.read_parquet(
        root / "trades.parquet",
        columns=["condition_id", "timestamp", "outcome", "side", "price"],
    )
    ev = pd.read_parquet(
        root / "events.parquet",
        columns=["event_id", "target_date"],
    )
    available = pd.read_parquet(root / "markets.parquet", columns=None).columns.tolist()
    mk_cols = ["event_id", "condition_id", "bucket_lo_f", "bucket_hi_f"]
    if "resolved_yes" in available:
        mk_cols.append("resolved_yes")
    mk = pd.read_parquet(root / "markets.parquet", columns=mk_cols)

    ev = ev[
        (pd.to_datetime(ev["target_date"]) >= pd.to_datetime(START_DATE))
        & (pd.to_datetime(ev["target_date"]) <= pd.to_datetime(CUTOFF))
    ]
    ev["target_date"] = pd.to_datetime(ev["target_date"]).dt.strftime("%Y-%m-%d")
    keep_events = set(ev["event_id"])
    mk = mk[mk["event_id"].isin(keep_events)].copy()
    mk = mk[mk["resolved_yes"].notna()].copy()

    am = argmax[argmax["station"] == CITY_TO_STATION[node]][
        ["target_date", "argmax_utc"]
    ].drop_duplicates("target_date")
    ev = ev.merge(am, on="target_date", how="left")
    mk = mk.merge(ev[["event_id", "argmax_utc"]], on="event_id", how="left")
    mk = mk[mk["argmax_utc"].notna()].copy()

    tr = tr.merge(mk[["condition_id"]], on="condition_id", how="inner")
    tr["yes_view"] = tr["price"].where(tr["outcome"] == "YES", 1.0 - tr["price"])
    tr = tr.merge(mk[["condition_id", "argmax_utc"]], on="condition_id")
    tr["hours_before_argmax"] = (
        tr["argmax_utc"] - tr["timestamp"]
    ).dt.total_seconds() / 3600.0

    # mean price over [h - 0.5h, h + 0.5h]; require the whole window to be
    # before argmax (h - 0.5 > 0) so prices are strictly ex-ante.
    price_cols: dict[str, pd.Series] = {}
    for h in HORIZONS:
        lo, hi = h - WIN_HALF_H, h + WIN_HALF_H
        mask = (
            (tr["hours_before_argmax"] >= lo)
            & (tr["hours_before_argmax"] <= hi)
        )
        near = (
            tr.loc[mask]
            .groupby("condition_id")["yes_view"]
            .mean()  # average ALL trades in the +/-30min window
        )
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
        f"conditions={len(panel)}",
        flush=True,
    )
    return panel


def main() -> int:
    ensure_out_dir()
    argmax = pd.read_parquet(ARGMAX_FILE)
    frames = [load_node(city, argmax) for city in NODE_DIRS]
    panel = pd.concat(frames, ignore_index=True)

    winners = panel[panel["y"] == 1].groupby("event_id").size()
    bad = int((winners != 1).sum())
    print(
        f"\n[sanity] events={panel['event_id'].nunique()} "
        f"conditions={len(panel)} bad_winner={bad}",
        flush=True,
    )
    panel.to_parquet(OUT_DIR / "first_layer_panel_argmax_30min.parquet", index=False)

    rep = ["Panel (argmax +/-30min mean) coverage report", "=" * 50]
    for node, sub in panel.groupby("city"):
        n_ev = sub["event_id"].nunique()
        rep.append(f"\n{node}: events={n_ev} conditions={len(sub)}")
        for h in HORIZONS:
            col = f"p_{int(h)}h"
            rep.append(f"  {col}: coverage={sub[col].notna().mean():.1%}")
    report = "\n".join(rep)
    (OUT_DIR / "panel_argmax_30min_quality_report.txt").write_text(
        report, encoding="utf-8"
    )
    print("\n" + report, flush=True)
    print(
        f"\n[done] wrote {OUT_DIR / 'first_layer_panel_argmax_30min.parquet'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
