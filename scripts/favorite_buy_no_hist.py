"""KLGA historical favorite (0.75-0.95) buy-NO strategy on trade prices.

Use pre-July resolution_v2 artifact (all settled). For each (event, h),
take the nearest trade in [h, h+3h] before close; if its YES-view price is
in [0.75, 0.95] (market overpricing favorite), BUY NO at price (1 - yes_view)
per share (taker). $10 notional, 5% fee, hold to settlement.

Per-h report: n trades, events, win rate (NO wins), PNL pre/post fee,
per-$1 PNL, and implied D = mean(y - p).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20250401_20260331_resolution_v2"
OUT = ROOT / "analysis_klga_l2_strategy"

HORIZONS = [48.0, 36.0, 24.0, 12.0, 6.0, 1.0]
WIN_H = 3.0
P_LO, P_HI = 0.75, 0.95
NOTIONAL = 10.0
FEE_RATE = 0.05


def load() -> pd.DataFrame:
    tr = pd.read_parquet(ART / "trades.parquet",
                         columns=["condition_id", "timestamp", "outcome", "side", "price"])
    mk = pd.read_parquet(ART / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    ev = pd.read_parquet(ART / "events.parquet",
                         columns=["event_id", "event_end_at"])
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    df = tr.merge(mk, on="condition_id", how="left").merge(ev, on="event_id", how="inner")
    df = df[df["resolved_yes"].notna()].copy()
    df["yes_view"] = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    df["hrs"] = (df["event_end_at"] - df["timestamp"]).dt.total_seconds() / 3600.0
    print(f"[load] settled trades={len(df):,}", flush=True)
    return df


def run_h(df: pd.DataFrame, h: float) -> pd.DataFrame:
    w = df[(df["hrs"] >= h) & (df["hrs"] < h + WIN_H)].copy()
    # nearest (latest) trade in window per (event, condition)
    w = w.sort_values(["event_id", "condition_id", "timestamp"])
    w = w.groupby(["event_id", "condition_id"], as_index=False).last()
    s = w[(w["yes_view"] >= P_LO) & (w["yes_view"] <= P_HI)].copy()
    if not len(s):
        return s
    s["h"] = h
    s["no_price"] = 1.0 - s["yes_view"]  # buy NO at the mirror trade price
    s["shares"] = NOTIONAL / s["no_price"]
    s["cost"] = NOTIONAL
    s["fee"] = FEE_RATE * s["cost"]
    s["won"] = ~s["resolved_yes"].astype(bool)  # NO wins when YES loses
    s["payout"] = s["shares"].where(s["won"], 0.0)
    s["pnl_pre"] = s["payout"] - s["cost"]
    s["pnl_post"] = s["payout"] - s["cost"] - s["fee"]
    s["y"] = s["resolved_yes"].astype(int)
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    all_trades: list[pd.DataFrame] = []
    rows: list[dict] = []
    print(f"\n{'h':>5} | {'n':>4} {'events':>6} {'win':>7} {'pnl_pre':>10} "
          f"{'per1_pre':>9} {'pnl_post':>10} {'per1_post':>9} {'implied_D':>9}")
    print("-" * 84)
    for h in HORIZONS:
        s = run_h(df, h)
        if not len(s):
            print(f"{h:>5.0f} |     0      0     n/a   {'-':>10} {'-':>9} {'-':>10} {'-':>9} {'-':>9}")
            continue
        all_trades.append(s)
        n = len(s)
        per1_pre = s["pnl_pre"].sum() / (n * NOTIONAL)
        per1_post = s["pnl_post"].sum() / (n * NOTIONAL)
        implied_d = (s["y"] - s["yes_view"]).mean()  # event-avg D
        print(f"{h:>5.0f} | {n:>4} {s['event_id'].nunique():>6} {s['won'].mean():>7.3f} "
              f"{s['pnl_pre'].sum():>10.2f} {per1_pre:>9.4f} "
              f"{s['pnl_post'].sum():>10.2f} {per1_post:>9.4f} {implied_d:>9.4f}")
        rows.append({"h": h, "n_trades": n, "n_events": int(s["event_id"].nunique()),
                     "win_rate": round(float(s["won"].mean()), 4),
                     "pnl_pre": round(float(s["pnl_pre"].sum()), 2),
                     "per1_pre": round(float(per1_pre), 4),
                     "pnl_post": round(float(s["pnl_post"].sum()), 2),
                     "per1_post": round(float(per1_post), 4),
                     "implied_d": round(float(implied_d), 4)})

    t = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    OUT.joinpath("favorite_buy_no_hist_trades.csv").write_text(
        t.to_csv(index=False), encoding="utf-8")
    OUT.joinpath("favorite_buy_no_hist_summary.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'favorite_buy_no_hist_trades.csv'}")
    print(f"[write] {OUT / 'favorite_buy_no_hist_summary.csv'}")


if __name__ == "__main__":
    main()
