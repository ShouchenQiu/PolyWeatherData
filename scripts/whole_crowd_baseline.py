"""Whole-crowd buyer PNL for KLGA 2026-07-02..08-02 (analysis only).

No strategy: ALL buyers (every condition, every time) who traded in the L2
coverage window 2026-07-02T00:00 .. 2026-08-02T23:59 UTC. Net their full
position on each condition to settlement; per-wallet and per-position PNL,
win rate, per-$1 notional. Pre- and post-fee (5% of gross notional).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20260701_20260818"
OUT = ROOT / "analysis_klga_l2_strategy"

WINDOW_LO = pd.Timestamp("2026-07-02T00:00:00Z")
WINDOW_HI = pd.Timestamp("2026-08-02T23:59:59Z")
FEE_RATE = 0.05


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    mk = pd.read_parquet(ART / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    tr = pd.read_parquet(ART / "trades.parquet")
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    print(f"[load] trades={len(tr):,}", flush=True)
    return tr, mk


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, mk = load()
    # buyers only, within window
    buyers = tr[
        (tr["side"] == "BUY")
        & (tr["timestamp"] >= WINDOW_LO)
        & (tr["timestamp"] <= WINDOW_HI)
    ]
    print(f"[window] buyer trades={len(buyers):,} wallets={buyers['proxy_wallet'].nunique():,}",
          flush=True)

    keys = buyers[["proxy_wallet", "condition_id"]].drop_duplicates()
    pos = keys.merge(tr, on=["proxy_wallet", "condition_id"], how="inner")
    pos = pos.merge(mk, on="condition_id", how="left")

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
    print(f"[positions] settled (wallet, condition)={len(g):,} "
          f"unresolved={int((~resolved).sum()):,}", flush=True)

    per_wallet = g.groupby("proxy_wallet").agg(
        pnl_pre=("pnl_pre", "sum"),
        pnl_post=("pnl_post", "sum"),
        notional=("notional", "sum"),
    )
    summary = {
        "n_buyer_trades": int(len(buyers)),
        "n_buyer_wallets": int(buyers["proxy_wallet"].nunique()),
        "n_settled_positions": int(len(g)),
        "n_unresolved_positions": int((~resolved).sum()),
        "gross_notional": round(float(g["notional"].sum()), 2),
        "pnl_pre_total": round(float(g["pnl_pre"].sum()), 2),
        "pnl_post_total": round(float(g["pnl_post"].sum()), 2),
        "per1_pnl_pre": round(float(g["pnl_pre"].sum() / g["notional"].sum()), 4),
        "per1_pnl_post": round(float(g["pnl_post"].sum() / g["notional"].sum()), 4),
        "avg_wallet_pnl_pre": round(float(per_wallet["pnl_pre"].mean()), 2),
        "avg_wallet_pnl_post": round(float(per_wallet["pnl_post"].mean()), 2),
        "median_wallet_pnl_pre": round(float(per_wallet["pnl_pre"].median()), 2),
        "median_wallet_pnl_post": round(float(per_wallet["pnl_post"].median()), 2),
        "wallet_win_rate_pre": round(float((per_wallet["pnl_pre"] > 0).mean()), 4),
        "wallet_win_rate_post": round(float((per_wallet["pnl_post"] > 0).mean()), 4),
        "position_win_rate_pre": round(float((g["pnl_pre"] > 0).mean()), 4),
        "position_win_rate_post": round(float((g["pnl_post"] > 0).mean()), 4),
        "window": [str(WINDOW_LO), str(WINDOW_HI)],
    }
    print("\n=== whole-crowd buyer baseline (KLGA, 2026-07-02..08-02) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    OUT.joinpath("whole_crowd_baseline.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    OUT.joinpath("whole_crowd_per_wallet.csv").write_text(
        per_wallet.reset_index().to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'whole_crowd_baseline.json'}")
    print(f"[write] {OUT / 'whole_crowd_per_wallet.csv'}")


if __name__ == "__main__":
    main()
