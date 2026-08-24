"""Compare strategy vs whole crowd in the 24-36h-before-settlement band (KLGA July).

Strategy: at a snapshot within [end-36h, end-24h], if YES mid <= 0.10 buy NO
as taker ($10, 5% fee), hold to settlement.
Crowd: real July trades in the same band (hrs_to_settle in [24,36]) on the
same long-shot buckets (YES-view <= 0.10), all buyers, net to settlement.
Reports per-$1 PNL pre- and post-fee for both.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20260701_20260818"
L2_DB = ROOT / "data_store" / "backtest" / "l2" / "live.sqlite"
OUT = ROOT / "analysis_klga_l2_strategy"

BAND_LO, BAND_HI = 24, 36  # hours before settlement
NOTIONAL = 10.0
FEE_RATE = 0.05
YES_MID_MAX = 0.10


def strategy() -> pd.DataFrame:
    ev = pd.read_parquet(ART / "events.parquet")
    mk = pd.read_parquet(ART / "markets.parquet")
    ev = ev[ev["event_end_at"].between("2026-07-01", "2026-07-31")]
    ev = ev[["event_id", "target_date", "event_end_at"]]
    mk = mk[["event_id", "bucket_lo_f", "yes_token_id", "no_token_id",
             "resolved_yes"]]
    df = mk.merge(ev, on="event_id", how="inner")
    df["event_end_at"] = pd.to_datetime(df["event_end_at"], utc=True)

    con = sqlite3.connect(f"file:{L2_DB.resolve()}?mode=ro", uri=True)
    snap = pd.read_sql_query(
        "SELECT token_id, captured_at, best_bid, best_ask FROM snapshots", con)
    con.close()
    snap["captured_at"] = pd.to_datetime(snap["captured_at"], utc=True)

    # band window per token
    t = df.copy()
    t["lo"] = t["event_end_at"] - pd.Timedelta(hours=BAND_HI)
    t["hi"] = t["event_end_at"] - pd.Timedelta(hours=BAND_LO)

    yes = t.merge(snap, left_on="yes_token_id", right_on="token_id", how="left")
    yes = yes[(yes["captured_at"] >= yes["lo"]) & (yes["captured_at"] <= yes["hi"])].copy()
    if not len(yes):
        return pd.DataFrame()
    yes = yes.sort_values(["event_id", "yes_token_id", "captured_at"])
    yes = yes.groupby(["event_id", "yes_token_id"], as_index=False).last()
    yes["yes_mid"] = (yes["best_bid"] + yes["best_ask"]) / 2.0

    no = t.merge(snap, left_on="no_token_id", right_on="token_id", how="left")
    no = no[(no["captured_at"] >= no["lo"]) & (no["captured_at"] <= no["hi"])].copy()
    if len(no):
        no = no.sort_values(["event_id", "no_token_id", "captured_at"])
        no = no.groupby(["event_id", "no_token_id"], as_index=False).last()
        no = no[["event_id", "no_token_id", "captured_at", "best_bid", "best_ask"]]

    tr = yes[yes["yes_mid"] <= YES_MID_MAX].merge(
        no, on=["event_id", "no_token_id"], how="left", suffixes=("_y", "_n"))
    tr = tr[tr["best_ask_n"].notna()]
    tr["no_ask"] = tr["best_ask_n"]
    tr["shares"] = NOTIONAL / tr["no_ask"]
    tr["cost"] = NOTIONAL
    tr["fee"] = FEE_RATE * tr["cost"]
    tr["won"] = ~tr["resolved_yes"].astype(bool)
    tr["payout"] = tr["shares"].where(tr["won"], 0.0)
    tr["pnl_pre"] = tr["payout"] - tr["cost"]
    tr["pnl_post"] = tr["payout"] - tr["cost"] - tr["fee"]
    return tr


def crowd() -> pd.DataFrame:
    ev = pd.read_parquet(ART / "events.parquet",
                         columns=["event_id", "event_end_at"])
    mk = pd.read_parquet(ART / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    tr = pd.read_parquet(ART / "trades.parquet")
    ev = ev[ev["event_end_at"].between("2026-07-01", "2026-07-31")]
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    df = tr.merge(mk, on="condition_id", how="left").merge(ev, on="event_id", how="inner")
    df["hrs"] = (df["event_end_at"] - df["timestamp"]).dt.total_seconds() / 3600.0
    df["yes_view"] = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    win = df[(df["hrs"] >= BAND_LO) & (df["hrs"] < BAND_HI) & (df["yes_view"] <= YES_MID_MAX)]
    buyers = win[win["side"] == "BUY"]
    keys = buyers[["proxy_wallet", "condition_id"]].drop_duplicates()
    pos = keys.merge(df, on=["proxy_wallet", "condition_id"], how="inner")
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
        signed_size=("signed_size", "sum"), cash=("cash", "sum"),
        notional=("notional", "sum"), resolved_yes=("resolved_yes", "first")).reset_index()
    resolved = g["resolved_yes"].notna()
    won = g["resolved_yes"].astype(bool).fillna(False).values
    ns = g["signed_size"].values
    payout = np.where(won, np.maximum(ns, 0.0), np.maximum(-ns, 0.0))
    payout = np.where(resolved.values, payout, 0.0)
    g["pnl_pre"] = g["cash"] + payout
    g["pnl_post"] = g["pnl_pre"] - FEE_RATE * g["notional"]
    g = g[resolved].copy()
    return g, buyers


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = strategy()
    print(f"[strategy] trades={len(s)} events={s['event_id'].nunique() if len(s) else 0}")
    if len(s):
        print(f"  pnl_pre={s['pnl_pre'].sum():.2f} per1_pre={s['pnl_pre'].sum()/(len(s)*NOTIONAL):.4f}")
        print(f"  pnl_post={s['pnl_post'].sum():.2f} per1_post={s['pnl_post'].sum()/(len(s)*NOTIONAL):.4f}")
        print(f"  win_rate={s['won'].mean():.4f}")

    g, buyers = crowd()
    print(f"\n[crowd 24-36h long-shot] buyer trades={len(buyers):,} wallets={buyers['proxy_wallet'].nunique():,} positions={len(g):,}")
    gross = g["notional"].sum()
    print(f"  per1_pre={g['pnl_pre'].sum()/gross:.4f}  per1_post={g['pnl_post'].sum()/gross:.4f}")
    print(f"  total pre={g['pnl_pre'].sum():.2f} post={g['pnl_post'].sum():.2f} notional={gross:.2f}")
    print(f"  position_win_rate_pre={(g['pnl_pre']>0).mean():.4f} post={(g['pnl_post']>0).mean():.4f}")

    OUT.joinpath("strategy_band_24_36h.csv").write_text(
        s.to_csv(index=False), encoding="utf-8") if len(s) else None
    OUT.joinpath("crowd_band_24_36h.csv").write_text(
        g.to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'strategy_band_24_36h.csv'} / {OUT / 'crowd_band_24_36h.csv'}")


if __name__ == "__main__":
    main()
