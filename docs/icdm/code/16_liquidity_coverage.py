"""Step 16: liquidity / coverage of the +/-0.5h pricing window.

Motivation
----------
The canonical panel prices each (event, bucket, horizon) as the MEAN of all
taker trades inside [h-0.5h, h+0.5h] before the daily-high argmax. The panel
stores only that mean, not how many trades produced it. To justify the
choice of a +/-30min window (vs a single nearest trade), we must show that
the window actually captures enough trades, and in particular enough in the
low-price (long-shot) region that the study focuses on.

This step recomputes, from the raw CLOB artifacts:
  - trades per (event, bucket, horizon) window
  - coverage: share of event buckets that have >=1 trade in the window
  - depth: median / mean trades per priced cell
  - per-horizon x price-bin counts (how the sample is distributed)

Outputs:
  docs/icdm/output/liquidity_by_horizon.csv
  docs/icdm/output/liquidity_by_bin.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLOB = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob")
OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
ARGMAX_FILE = OUT_DIR / "daily_high_argmax.parquet"
HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
WIN_HALF_H = 0.5
CUTOFF = "2026-08-10"
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
    "NYC": "KLGA", "LONDON": "LONDON", "CHICAGO": "KORD",
    "DALLAS": "KDFW", "MIAMI": "KMIA", "DENVER": "KDEN",
    "SEATTLE": "KSEA", "ATLANTA": "KATL", "TOKYO": "TOKYO",
    "SHANGHAI": "SHANGHAI", "SEOUL": "SEOUL", "TORONTO": "TORONTO",
}

BINS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.01]
BIN_LABELS = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-30%",
              "30-50%", "50-70%", "70-90%", "90-100%"]


def per_node(node: str, argmax: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, bucket, horizon) with trade count in the window."""
    station, artifact = NODES[node]
    root = CLOB / station / artifact

    tr = pd.read_parquet(
        root / "trades.parquet",
        columns=["condition_id", "timestamp", "outcome", "price"])
    ev = pd.read_parquet(root / "events.parquet",
                         columns=["event_id", "target_date"])
    mk_cols = ["event_id", "condition_id", "bucket_lo_f", "bucket_hi_f"]
    avail = pd.read_parquet(root / "markets.parquet", columns=None).columns
    if "resolved_yes" in avail:
        mk_cols.append("resolved_yes")
    mk = pd.read_parquet(root / "markets.parquet", columns=mk_cols)

    ev = ev[(pd.to_datetime(ev["target_date"]) >= pd.Timestamp(START_DATE))
            & (pd.to_datetime(ev["target_date"]) <= pd.Timestamp(CUTOFF))]
    ev["target_date"] = pd.to_datetime(ev["target_date"]).dt.strftime("%Y-%m-%d")
    mk = mk[mk["event_id"].isin(set(ev["event_id"]))]
    mk = mk[mk["resolved_yes"].notna()].copy()

    am = argmax[argmax["station"] == CITY_TO_STATION[node]][
        ["target_date", "argmax_utc"]].drop_duplicates("target_date")
    ev = ev.merge(am, on="target_date", how="left")
    mk = mk.merge(ev[["event_id", "argmax_utc"]], on="event_id", how="left")
    mk = mk[mk["argmax_utc"].notna()].copy()

    tr = tr[tr["condition_id"].isin(set(mk["condition_id"]))].copy()
    tr["yes_view"] = tr["price"].where(tr["outcome"] == "YES",
                                      1.0 - tr["price"])
    tr = tr.merge(mk[["condition_id", "argmax_utc", "event_id"]],
                  on="condition_id")
    tr["hrs"] = (tr["argmax_utc"] - tr["timestamp"]).dt.total_seconds() / 3600.0

    frames = []
    for h in HORIZONS:
        m = (tr["hrs"] >= h - WIN_HALF_H) & (tr["hrs"] <= h + WIN_HALF_H)
        w = tr.loc[m]
        if w.empty:
            continue
        g = w.groupby("condition_id").agg(
            n_trades=("yes_view", "size"),
            p_mean=("yes_view", "mean"))
        g["h"] = int(h)
        frames.append(g.reset_index())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.merge(
        mk[["condition_id", "event_id", "bucket_lo_f", "resolved_yes"]],
        on="condition_id", how="left")
    out["y"] = out["resolved_yes"].astype(int)
    out["city"] = node
    return out.drop(columns=["resolved_yes"])


def main() -> int:
    argmax = pd.read_parquet(ARGMAX_FILE)
    parts = []
    for node in NODES:
        df = per_node(node, argmax)
        if df.empty:
            print(f"[skip] {node}: no rows", flush=True)
            continue
        parts.append(df)
        print(f"[loaded] {node}: {len(df)} event-bucket-horizon cells",
              flush=True)

    cells = pd.concat(parts, ignore_index=True)
    print(f"\n[total] {len(cells)} priced cells across "
          f"{cells['event_id'].nunique()} events", flush=True)

    # ---- denominator: every bucket of every event (priced or not) -------
    tot = (cells.groupby(["event_id", "h"])["bucket_lo_f"].nunique()
           .reset_index(name="n_priced"))
    allb = (cells.groupby("event_id")["bucket_lo_f"].nunique()
            .reset_index(name="n_total"))
    cov = tot.merge(allb, on="event_id")
    cov["coverage"] = cov["n_priced"] / cov["n_total"]

    # ---- per-horizon summary -------------------------------------------
    rows = []
    for h in sorted(cells["h"].unique(), reverse=True):
        c = cells[cells["h"] == h]
        cv = cov[cov["h"] == h]
        rows.append({
            "horizon_h": int(h),
            "n_events": int(c["event_id"].nunique()),
            "n_cells_priced": int(len(c)),
            "n_trades_total": int(c["n_trades"].sum()),
            "trades_per_cell_mean": round(float(c["n_trades"].mean()), 2),
            "trades_per_cell_median": int(c["n_trades"].median()),
            "trades_per_cell_p25": int(c["n_trades"].quantile(0.25)),
            "cell_coverage_pct": round(float(cv["coverage"].mean()) * 100, 1),
            "evt_coverage_pct": round(float(
                (cv.groupby("event_id")["coverage"].mean() == 1.0).mean()
            ) * 100, 1),
            "share_cells_ge5_trades": round(
                float((c["n_trades"] >= 5).mean()) * 100, 1),
        })
    by_h = pd.DataFrame(rows)
    by_h.to_csv(OUT_DIR / "liquidity_by_horizon.csv", index=False)

    # ---- per (horizon, price bin) --------------------------------------
    cells = cells.copy()
    cells["bin"] = pd.cut(cells["p_mean"], bins=BINS, labels=BIN_LABELS,
                          include_lowest=True)
    bg = (cells.groupby(["h", "bin"], observed=True)
          .agg(n_cells=("n_trades", "size"),
               n_events=("event_id", "nunique"),
               trades=("n_trades", "sum"),
               trades_med=("n_trades", "median")).reset_index())
    bg.to_csv(OUT_DIR / "liquidity_by_bin.csv", index=False)

    # ---- print --------------------------------------------------------
    print("\n" + "=" * 96)
    print("A. LIQUIDITY BY HORIZON (+/-0.5h window)")
    print("=" * 96)
    print(f"{'h':>4s} {'events':>7s} {'cells':>7s} {'trades':>9s} "
          f"{'tr/cell mean':>13s} {'median':>7s} {'P25':>5s} "
          f"{'cell cov%':>10s} {'evt cov%':>9s} {'>=5 tr%':>8s}")
    for _, r in by_h.iterrows():
        print(f"{int(r['horizon_h']):4d} {int(r['n_events']):7d} "
              f"{int(r['n_cells_priced']):7d} {int(r['n_trades_total']):9d} "
              f"{r['trades_per_cell_mean']:13.2f} "
              f"{int(r['trades_per_cell_median']):7d} "
              f"{int(r['trades_per_cell_p25']):5d} "
              f"{r['cell_coverage_pct']:10.1f} "
              f"{r['evt_coverage_pct']:9.1f} "
              f"{r['share_cells_ge5_trades']:8.1f}")

    print("\n" + "=" * 96)
    print("B. CELLS AND TRADES PER (horizon x price bin) - long-shot bins "
          "first")
    print("=" * 96)
    piv_c = bg.pivot(index="bin", columns="h", values="n_cells")
    piv_t = bg.pivot(index="bin", columns="h", values="trades")
    print("\n-- cells (event-bucket-horizon rows) --")
    print(piv_c.reindex(BIN_LABELS).fillna(0).astype(int).to_string())
    print("\n-- trades --")
    print(piv_t.reindex(BIN_LABELS).fillna(0).astype(int).to_string())

    print(f"\n[write] {OUT_DIR / 'liquidity_by_horizon.csv'}")
    print(f"[write] {OUT_DIR / 'liquidity_by_bin.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
