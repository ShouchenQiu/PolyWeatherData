"""Cross-city historical buy-NO strategy PNL (10 nodes, excluding DENVER/DALLAS).

Strategies (both use trade prices, pre-July settled history):
  FAV  : window [1h, 12h] before close; YES-view price in [0.75, 0.95]
         (favorite overpriced) -> buy NO.
  LS   : window [30h, 48h] before close; YES-view price <= 0.10
         (long-shot overpriced) -> buy NO.
Per (event, condition) take the LAST trade in window as signal price; buy NO at
(1 - yes_view), $10 notional, hold to settlement. Fee follows the real
Polymarket schedule: Fee = Theta * C * p * (1 - p), where C = shares bought,
p = traded token price, Theta = fee rate (0.05). Per-city and pooled summary.
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

# 10 nodes (12 minus DENVER, DALLAS)
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

FAV_LO, FAV_HI = 0.75, 0.95
FAV_WIN = (1.0, 12.0)   # hours before close
LS_MAX = 0.10
LS_WIN = (30.0, 48.0)


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
    print(f"  [load] {station}/{art}: trades={len(df):,} events={df['event_id'].nunique():,}",
          flush=True)
    return df


def run_strategy(df: pd.DataFrame, win: tuple[float, float], plo: float, phi: float) -> pd.DataFrame:
    lo, hi = win
    w = df[(df["hrs"] >= lo) & (df["hrs"] < hi)].copy()
    w = w.sort_values(["event_id", "condition_id", "timestamp"])
    w = w.groupby(["event_id", "condition_id"], as_index=False).last()  # last trade in window
    if phi is None:
        s = w[w["yes_view"] <= plo].copy()
    else:
        s = w[(w["yes_view"] >= plo) & (w["yes_view"] <= phi)].copy()
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


def summarize(s: pd.DataFrame, city: str, strat: str) -> dict:
    if not len(s):
        return {"city": city, "strategy": strat, "n_trades": 0, "n_events": 0,
                "win_rate": np.nan, "pnl_pre": 0.0, "per1_pre": np.nan,
                "pnl_post": 0.0, "per1_post": np.nan, "implied_d": np.nan}
    n = len(s)
    return {
        "city": city, "strategy": strat, "n_trades": n,
        "n_events": int(s["event_id"].nunique()),
        "win_rate": round(float(s["won"].mean()), 4),
        "pnl_pre": round(float(s["pnl_pre"].sum()), 2),
        "per1_pre": round(float(s["pnl_pre"].sum() / (n * NOTIONAL)), 4),
        "pnl_post": round(float(s["pnl_post"].sum()), 2),
        "per1_post": round(float(s["pnl_post"].sum() / (n * NOTIONAL)), 4),
        "implied_d": round(float((s["y"] - s["yes_view"]).mean()), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fav_frames: list[pd.DataFrame] = []
    ls_frames: list[pd.DataFrame] = []
    for city, (station, art) in NODES.items():
        df = load_node(station, art)
        # keep only pre-cutoff settled events (history)
        df = df[pd.to_datetime(df["target_date"], errors="coerce") <= CUTOFF].copy()
        print(f"    history events after cutoff: {df['event_id'].nunique():,}", flush=True)
        fav = run_strategy(df, FAV_WIN, FAV_LO, FAV_HI)
        ls = run_strategy(df, LS_WIN, LS_MAX, None)
        rows.append(summarize(fav, city, "favorite_NO"))
        rows.append(summarize(ls, city, "longshot_NO"))
        if len(fav):
            fav["city"] = city
            fav_frames.append(fav)
        if len(ls):
            ls["city"] = city
            ls_frames.append(ls)
        print(f"    FAV n={len(fav)}  LS n={len(ls)}", flush=True)

    out = pd.DataFrame(rows)
    OUT.joinpath("cross_city_fav_ls_summary.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")

    print("\n=== per city ===")
    print(out.to_string(index=False))

    print("\n=== pooled (10 cities) ===")
    for strat, frames in [("favorite_NO", fav_frames), ("longshot_NO", ls_frames)]:
        if not frames:
            print(f"{strat}: no trades")
            continue
        t = pd.concat(frames, ignore_index=True)
        n = len(t)
        print(f"{strat}: n={n} events={t['event_id'].nunique()} cities={t['city'].nunique()} "
              f"win_rate={t['won'].mean():.4f} "
              f"pnl_pre={t['pnl_pre'].sum():.2f} per1_pre={t['pnl_pre'].sum()/(n*NOTIONAL):.4f} "
              f"pnl_post={t['pnl_post'].sum():.2f} per1_post={t['pnl_post'].sum()/(n*NOTIONAL):.4f}")
        OUT.joinpath(f"cross_city_{strat}_trades.csv").write_text(
            t.to_csv(index=False), encoding="utf-8")

    print(f"\n[write] {OUT / 'cross_city_fav_ls_summary.csv'}")


if __name__ == "__main__":
    main()
