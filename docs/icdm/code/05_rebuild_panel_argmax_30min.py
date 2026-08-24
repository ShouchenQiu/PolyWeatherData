"""Step 5: rebuild first-layer panel using MEAN price over +/-30min window.

Differs from step 4 (nearest trade) by averaging ALL trades within
[h - 0.5h, h + 0.5h] of each horizon point (hours before the argmax time).
This increases per-bucket sample size and smooths single-trade noise.

Output: docs/icdm/output/first_layer_panel_argmax_30min.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLOB = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob")
OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
ARGMAX_FILE = OUT_DIR / "daily_high_argmax.parquet"
HORIZONS = [
    48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0,
]
WIN_HALF_H = 0.5  # +/- 30 minutes
CUTOFF = "2026-08-10"
# Integrated dataset window: every node starts 2025-12-05 (Tokyo/Shanghai
# naturally start 2026-03, unaffected) and ends 2026-08-10.
START_DATE = "2025-12-05"

NODES: dict[str, tuple[str, str]] = {
    "NYC": ("KLGA", "20251205_20260810"),
    "LONDON": ("LONDON", "20251205_20260810"),
    "CHICAGO": ("CHICAGO", "20251205_20260810"),
    "DALLAS": ("DALLAS", "20251205_20260810"),
    "MIAMI": ("MIAMI", "20251205_20260810"),
    "DENVER": ("DENVER", "20251205_20260810"),
    "SEATTLE": ("SEATTLE", "20251205_20260810"),
    "ATLANTA": ("ATLANTA", "20251205_20260810"),
    "TOKYO": ("TOKYO", "20251205_20260810"),
    "SHANGHAI": ("SHANGHAI", "20251205_20260810"),
    "SEOUL": ("SEOUL", "20251205_20260810"),
    "TORONTO": ("TORONTO", "20251205_20260810"),
}

CITY_TO_STATION = {
    "NYC": "KLGA", "LONDON": "LONDON",
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
