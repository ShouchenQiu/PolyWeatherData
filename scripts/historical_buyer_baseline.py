"""Historical buyer stats for KLGA long-shot buckets (analysis only).

Use the PRE-JULY KLGA artifact (20250401_20260331_resolution_v2, all settled
before 6/30) with real trades. Select trades in the SAME window as the July
strategy (h in {6,12,24}h +/-2h to settlement, YES-view price <= 0.10).
For every wallet that BOUGHT in these buckets, net their full position on
those conditions to settlement; report average PNL (per-$1 notional and per
wallet) and win rate (per condition and per wallet), pre- and post-fee.

Groups: buy_yes (wallets that bought YES), buy_no (wallets that bought NO),
all_buyers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20250401_20260331_resolution_v2"
OUT = ROOT / "analysis_klga_l2_strategy"

HOURS = [6, 12, 24]
WINDOW = 2.0  # hours, matches strategy snapshot tolerance
YES_MAX = 0.10
FEE_RATE = 0.05


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = pd.read_parquet(ART / "events.parquet",
                         columns=["event_id", "event_end_at"])
    mk = pd.read_parquet(ART / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    tr = pd.read_parquet(ART / "trades.parquet")
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    print(f"[load] hist trades={len(tr):,} events={len(ev):,}", flush=True)
    return tr, mk, ev


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, mk, ev = load()
    df = tr.merge(mk, on="condition_id", how="left").merge(ev, on="event_id", how="inner")
    df["hrs_to_settle"] = (df["event_end_at"] - df["timestamp"]).dt.total_seconds() / 3600.0
    df["yes_view"] = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    df["is_buy"] = df["side"] == "BUY"

    rows: list[dict] = []
    for h in HOURS:
        lo, hi = h - WINDOW, h + WINDOW
        in_window = (df["hrs_to_settle"] >= lo) & (df["hrs_to_settle"] < hi)
        longshot = df["yes_view"] <= YES_MAX
        win = df[in_window & longshot]

        # buyers: wallets that BOUGHT (YES or NO) in these buckets in this window
        buy_yes = win[(win["outcome"] == "YES") & win["is_buy"]]
        buy_no = win[(win["outcome"] == "NO") & win["is_buy"]]

        for label, sub in [("buy_yes", buy_yes), ("buy_no", buy_no),
                           ("all_buyers", win[win["is_buy"]])]:
            if not len(sub):
                continue
            keys = sub[["proxy_wallet", "condition_id"]].drop_duplicates()
            # full position of these wallets on these conditions (all time)
            pos = keys.merge(df, on=["proxy_wallet", "condition_id"], how="inner")
            # net position PNL
            p = pos.copy()
            yes_buy = (p["outcome"] == "YES") & (p["side"] == "BUY")
            no_sell = (p["outcome"] == "NO") & (p["side"] == "SELL")
            no_buy = (p["outcome"] == "NO") & (p["side"] == "BUY")
            yes_sell = (p["outcome"] == "YES") & (p["side"] == "SELL")
            p["signed_size"] = np.select(
                [yes_buy, no_sell, no_buy, yes_sell],
                [p["size"], p["size"], -p["size"], -p["size"]], default=0.0)
            p["cash"] = np.where(p["side"] == "BUY", -1.0, 1.0) * p["price"] * p["size"]
            p["notional"] = (p["price"] * p["size"]).abs()

            g = p.groupby(["proxy_wallet", "condition_id"], sort=False).agg(
                signed_size=("signed_size", "sum"),
                cash=("cash", "sum"),
                notional=("notional", "sum"),
                resolved_yes=("resolved_yes", "first"),
            ).reset_index()
            resolved = g["resolved_yes"].notna()
            won = g["resolved_yes"].astype(bool).fillna(False).values
            ns = g["signed_size"].values
            payout = np.where(won, np.maximum(ns, 0.0), np.maximum(-ns, 0.0))
            payout = np.where(resolved.values, payout, 0.0)
            g["pnl_pre"] = g["cash"] + payout
            g["pnl_post"] = g["pnl_pre"] - FEE_RATE * g["notional"]
            g = g[resolved].copy()
            if not len(g):
                continue
            # per-wallet totals
            w = g.groupby("proxy_wallet").agg(
                pnl_pre=("pnl_pre", "sum"),
                pnl_post=("pnl_post", "sum"),
                notional=("notional", "sum"),
            )
            per1_pre = g["pnl_pre"].sum() / g["notional"].sum()
            per1_post = g["pnl_post"].sum() / g["notional"].sum()
            rows.append({
                "h": h, "group": label,
                "n_trades_in_window": int(len(sub)),
                "n_wallets": int(w.shape[0]),
                "n_positions": int(len(g)),
                "gross_notional": round(g["notional"].sum(), 2),
                "per1_pnl_pre": round(per1_pre, 4),
                "per1_pnl_post": round(per1_post, 4),
                "avg_wallet_pnl_pre": round(w["pnl_pre"].mean(), 2),
                "avg_wallet_pnl_post": round(w["pnl_post"].mean(), 2),
                "wallet_win_rate": round(float((w["pnl_post"] > 0).mean()), 4),
                "position_win_rate": round(float((g["pnl_post"] > 0).mean()), 4),
            })
        print(f"[h={h:>3}] buy_yes wallets={rows[-3 if len(rows)>=3 else 0]['n_wallets']} "
              f"per1_post={rows[-3 if len(rows)>=3 else 0]['per1_pnl_post']}", flush=True)

    out = pd.DataFrame(rows)
    print("\n=== historical buyer stats (pre-July, long-shot buckets) ===")
    print(out.to_string(index=False))
    OUT.joinpath("historical_buyer_baseline.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'historical_buyer_baseline.csv'}")


if __name__ == "__main__":
    main()
