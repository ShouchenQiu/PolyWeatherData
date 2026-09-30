"""Favorite buy-NO by FIXED signal hour h, 10 cities (excl DENVER/DALLAS).

For each fixed h in {1..11}: take the LAST trade in window [h, h+1h) before
close (idxmax on hrs, no full sort); if its YES-view price in [0.75, 0.95],
buy NO at (1 - yes_view), $10, hold to settlement. Fee = Theta * C * p * (1-p)
(real Polymarket schedule), C = shares, p = traded token price. Per-h pooled
PNL/ROI with event-block bootstrap CI (500 draws).
"""
from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLOB_ROOT, OUT_DIR  # noqa: E402
ROOT = CLOB_ROOT.parent.parent
CLOB = CLOB_ROOT
OUT = OUT_DIR

CUTOFF = "2026-06-30"
NOTIONAL = 10.0
FEE_RATE = 0.05  # Theta in Fee = Theta * C * p * (1 - p)
P_LO, P_HI = 0.75, 0.95
HS = list(range(1, 12))
NBOOT = 500

NODES: dict[str, tuple[str, str]] = {
    "NYC": ("KLGA", "20250401_20260331_resolution_v2"),
    "LONDON": ("LONDON", "20250121_20260810"),
    "CHICAGO": ("CHICAGO", "20251204_20260810"),
    "MIAMI": ("MIAMI", "20250101_20260812"),
    "SEATTLE": ("SEATTLE", "20250101_20260812"),
    "ATLANTA": ("ATLANTA", "20250101_20260812"),
    "TOKYO": ("TOKYO", "20250101_20260818"),
    "SHANGHAI": ("SHANGHAI", "20250101_20260818"),
    "SEOUL": ("SEOUL", "20250101_20260818"),
    "TORONTO": ("TORONTO", "20250101_20260818"),
}


def load_node(station: str, art: str) -> pd.DataFrame:
    root = CLOB / station / art
    tr = pd.read_parquet(root / "trades.parquet",
                         columns=["condition_id", "timestamp", "outcome", "side", "price"])
    mk = pd.read_parquet(root / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    ev = pd.read_parquet(root / "events.parquet",
                         columns=["event_id", "target_date", "event_end_at"])
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    df = tr.merge(mk, on="condition_id", how="left").merge(ev, on="event_id", how="inner")
    df = df[df["resolved_yes"].notna()].copy()
    df["yes_view"] = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    df["hrs"] = (df["event_end_at"] - df["timestamp"]).dt.total_seconds() / 3600.0
    return df


def signal_at(df: pd.DataFrame, h: float) -> pd.DataFrame:
    """Last trade in [h, h+1) per condition, favorite-priced only."""
    w = df[(df["hrs"] >= h) & (df["hrs"] < h + 1.0)]
    if not len(w):
        return w.iloc[0:0]
    idx = w.groupby(["event_id", "condition_id"], sort=False)["hrs"].idxmax()
    s = w.loc[idx].copy()
    s = s[(s["yes_view"] >= P_LO) & (s["yes_view"] <= P_HI)]
    if not len(s):
        return s
    s["no_price"] = 1.0 - s["yes_view"]
    s["shares"] = NOTIONAL / s["no_price"]
    s["cost"] = NOTIONAL
    s["fee"] = FEE_RATE * s["shares"] * s["no_price"] * (1.0 - s["no_price"])
    s["won"] = ~s["resolved_yes"].astype(bool)
    s["payout"] = s["shares"].where(s["won"], 0.0)
    s["pnl_pre"] = s["payout"] - s["cost"]
    s["pnl_post"] = s["payout"] - s["cost"] - s["fee"]
    s["y"] = s["resolved_yes"].astype(int)
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: dict[int, list[pd.DataFrame]] = {h: [] for h in HS}
    for city, (station, art) in NODES.items():
        df = load_node(station, art)
        df = df[pd.to_datetime(df["target_date"], errors="coerce") <= CUTOFF].copy()
        for h in HS:
            s = signal_at(df, float(h))
            if len(s):
                s["city"] = city
                frames[h].append(s)
        del df
        print(f"[{city}] loaded", flush=True)

    rows = []
    print(f"\n{'h':>3} | {'n':>4} {'events':>6} {'cities':>6} {'win':>7} "
          f"{'pnl_post':>9} {'per1_post':>9} {'ROI%':>8} {'CI(2.5-97.5)':>20}")
    print("-" * 84)
    rng = np.random.default_rng(42)
    for h in HS:
        if not frames[h]:
            print(f"{h:>3} |     0      0      0    n/a {'-':>9} {'-':>9} {'-':>8} {'-':>20}")
            continue
        t = pd.concat(frames[h], ignore_index=True)
        n = len(t)
        pnl = t["pnl_post"].sum()
        per1 = pnl / (n * NOTIONAL)
        eids = t["event_id"].unique()
        dist = np.empty(NBOOT)
        for b in range(NBOOT):
            sub = t[t["event_id"].isin(rng.choice(eids, size=len(eids), replace=True))]
            dist[b] = sub["pnl_post"].sum() / (len(sub) * NOTIONAL)
        ci = f"[{np.percentile(dist,2.5):+.3f}, {np.percentile(dist,97.5):+.3f}]"
        print(f"{h:>3} | {n:>4} {t['event_id'].nunique():>6} {t['city'].nunique():>6} "
              f"{t['won'].mean():>7.3f} {pnl:>9.2f} {per1:>9.4f} {per1*100:>8.1f} {ci:>20}")
        rows.append({"h": h, "n_trades": n, "n_events": int(t["event_id"].nunique()),
                     "n_cities": int(t["city"].nunique()), "win_rate": round(float(t["won"].mean()), 4),
                     "pnl_post": round(float(pnl), 2), "per1_post": round(float(per1), 4),
                     "roi_post_pct": round(float(per1 * 100), 2),
                     "ci_low": round(float(np.percentile(dist, 2.5)), 4),
                     "ci_high": round(float(np.percentile(dist, 97.5)), 4)})

    out = pd.DataFrame(rows)
    OUT.joinpath("favorite_fixed_h_bins.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'favorite_fixed_h_bins.csv'}")


if __name__ == "__main__":
    main()
