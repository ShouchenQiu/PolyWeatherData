"""User-crowd baseline PNL for KLGA July long-shot buckets, aligned to strategy.

Strategy trades: July KLGA, h in {6,12,24}h before settlement, YES price
<= 0.10, buy NO, $10 notional, taker, 5% fee, hold to settlement.

User-crowd control: real July KLGA trades filtered to the same buckets
(YES price <= 0.10), same time windows (h +/- 2h around settlement), same
(notional-weighted) per-$1 PNL basis. Reported pre-fee and post-fee
(5% of gross notional). Also a buy-NO-only subset aligned to the strategy
direction.

Output: analysis_klga_l2_strategy/user_crowd_baseline.csv + printed table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20260701_20260818"
OUT = ROOT / "analysis_klga_l2_strategy"

HOURS = [6, 12, 24]
WINDOW = 2  # hours; strategy snapshot tolerance
YES_MAX = 0.10
FEE_RATE = 0.05


def load() -> pd.DataFrame:
    ev = pd.read_parquet(ART / "events.parquet",
                         columns=["event_id", "event_end_at"])
    mk = pd.read_parquet(ART / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    tr = pd.read_parquet(ART / "trades.parquet")
    ev = ev[ev["event_end_at"].between("2026-07-01", "2026-07-31")]
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    df = tr.merge(mk, on="condition_id", how="left")
    df = df.merge(ev, on="event_id", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hrs_to_settle"] = (
        df["event_end_at"] - df["timestamp"]
    ).dt.total_seconds() / 3600.0
    # YES-view price of the trade
    df["yes_view"] = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    print(f"[load] july trades={len(df):,}", flush=True)
    return df


def settle_pnl(tr: pd.DataFrame, fee: bool) -> pd.DataFrame:
    """Net-position-to-settlement PNL per (wallet, condition), per-$1 notional."""
    t = tr.copy()
    yes_buy = (t["outcome"] == "YES") & (t["side"] == "BUY")
    no_sell = (t["outcome"] == "NO") & (t["side"] == "SELL")
    no_buy = (t["outcome"] == "NO") & (t["side"] == "BUY")
    yes_sell = (t["outcome"] == "YES") & (t["side"] == "SELL")
    t["signed_size"] = np.select(
        [yes_buy, no_sell, no_buy, yes_sell],
        [t["size"], t["size"], -t["size"], -t["size"]],
        default=0.0,
    )
    t["cash"] = np.where(t["side"] == "BUY", -1.0, 1.0) * t["price"] * t["size"]
    t["notional"] = t["price"] * t["size"].abs()

    key = ["proxy_wallet", "condition_id"]
    net = t.groupby(key, sort=False).agg(
        signed_size=("signed_size", "sum"),
        cash=("cash", "sum"),
        notional=("notional", "sum"),
        resolved_yes=("resolved_yes", "first"),
    ).reset_index()
    resolved = net["resolved_yes"].notna()
    won = net["resolved_yes"].astype(bool).fillna(False).values
    ns = net["signed_size"].values
    payout = np.where(won, np.maximum(ns, 0.0), np.maximum(-ns, 0.0))
    payout = np.where(resolved.values, payout, 0.0)
    net["payout"] = payout
    net["pnl_pre"] = net["cash"] + net["payout"]
    if fee:
        net["pnl_post"] = net["pnl_pre"] - FEE_RATE * net["notional"]
    else:
        net["pnl_post"] = net["pnl_pre"]
    net["is_resolved"] = resolved
    return net


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    rows: list[dict] = []
    for h in HOURS:
        lo = h - WINDOW
        hi = h + WINDOW
        sub = df[
            (df["hrs_to_settle"] >= lo) & (df["hrs_to_settle"] < hi)
            & (df["yes_view"] <= YES_MAX)
        ]
        # all users in these long-shot buckets
        net_all = settle_pnl(sub, fee=False)
        net_all_f = settle_pnl(sub, fee=True)
        # buy-NO-only subset (same action as strategy)
        buy_no = sub[(sub["outcome"] == "NO") & (sub["side"] == "BUY")]
        net_no = settle_pnl(buy_no, fee=False)
        net_no_f = settle_pnl(buy_no, fee=True)

        def agg(net: pd.DataFrame, label: str) -> dict:
            resolved = net[net["is_resolved"]]
            gross = resolved["notional"].sum()
            pnl_pre = resolved["pnl_pre"].sum()
            pnl_post = resolved["pnl_post"].sum()
            return {
                "window_h": h,
                "group": label,
                "n_trades": int(len(sub) if label != "buy_no" else len(buy_no)),
                "n_wallet_positions": int(len(resolved)),
                "n_wallets": int(resolved["proxy_wallet"].nunique()),
                "gross_notional": round(gross, 2),
                "pnl_pre": round(pnl_pre, 2),
                "pnl_post": round(pnl_post, 2),
                "per1_pre": round(pnl_pre / gross, 4) if gross else np.nan,
                "per1_post": round(pnl_post / gross, 4) if gross else np.nan,
            }

        rows.append(agg(net_all, "all_users_pre"))
        rows.append(agg(net_all_f, "all_users_post"))
        rows.append(agg(net_no, "buy_no_only_pre"))
        rows.append(agg(net_no_f, "buy_no_only_post"))
        print(f"[h={h:>3}] trades={len(sub):>7,} "
              f"wallets={rows[-4]['n_wallets']:>6,} "
              f"per1_pre={rows[-4]['per1_pre']} "
              f"per1_post={rows[-4]['per1_post']}", flush=True)

    # dedupe: agg() duplicated labels; keep one row per (window, group, fee)
    out = pd.DataFrame(rows).drop_duplicates(
        subset=["window_h", "group"]
    ).sort_values(["window_h", "group"]).reset_index(drop=True)
    out["fee"] = np.where(out["per1_pre"] != out["per1_post"], "post", "pre")
    print("\n=== user-crowd baseline (long-shot buckets, July) ===")
    print(out.to_string(index=False))
    OUT.joinpath("user_crowd_baseline.csv").write_text(
        out.to_csv(index=False), encoding="utf-8"
    )
    print(f"\n[write] {OUT / 'user_crowd_baseline.csv'}")


if __name__ == "__main__":
    main()
