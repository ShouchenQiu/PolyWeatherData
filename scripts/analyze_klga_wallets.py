"""Per-wallet stats for ALL KLGA CLOB markets (net-position-to-settlement PNL).

Merges the three resolved_yes-capable artifacts (dedup by transaction_hash),
computes per (wallet, condition) net signed size + cash flow, settles against
resolved_yes, and aggregates per wallet:
  - trade counts (total / buys / sells)
  - gross volume (buy_usdc / sell_usdc)
  - conditions & events participated
  - net cash, settlement payout, realized PNL
  - win rate (settled conditions where the held direction won)
  - profitable condition rate
Unresolved conditions (resolved_yes missing) are excluded from payout but
counted separately.

Output: analysis_klga_wallets/klga_wallet_stats.csv + summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob\KLGA")
OUT = Path(r"d:\NN\PolyWeatherData\analysis_klga_wallets")

ARTS = [
    "20250401_20260331_resolution_v2",
    "20251205_20260810",
    "20260701_20260818",
]

TRADE_COLS = [
    "condition_id", "transaction_hash", "timestamp", "outcome", "side",
    "price", "size", "usdc", "proxy_wallet",
]


def load_trades() -> pd.DataFrame:
    frames = []
    for art in ARTS:
        p = BASE / art / "trades.parquet"
        t = pd.read_parquet(p, columns=TRADE_COLS)
        t["_src"] = art
        frames.append(t)
        print(f"[load] {art}: {len(t):,} trades", flush=True)
    tr = pd.concat(frames, ignore_index=True)
    before = len(tr)
    tr = tr.drop_duplicates(subset=["transaction_hash"], keep="first")
    print(f"[dedup] {before:,} -> {len(tr):,} unique tx", flush=True)
    return tr


def load_markets() -> pd.DataFrame:
    frames = []
    for art in ARTS:
        p = BASE / art / "markets.parquet"
        m = pd.read_parquet(
            p, columns=["condition_id", "event_id", "resolved_yes"]
        )
        frames.append(m)
    mk = pd.concat(frames, ignore_index=True)
    mk = mk.drop_duplicates(subset=["condition_id"], keep="first")
    print(f"[markets] {len(mk):,} conditions", flush=True)
    return mk


def load_events() -> pd.DataFrame:
    frames = []
    for art in ARTS:
        p = BASE / art / "events.parquet"
        e = pd.read_parquet(p, columns=["event_id", "target_date"])
        frames.append(e)
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.drop_duplicates(subset=["event_id"], keep="first")
    ev["target_date"] = pd.to_datetime(ev["target_date"], errors="coerce")
    return ev


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_trades()
    mk = load_markets()
    ev = load_events()

    t = tr.merge(
        mk[["condition_id", "event_id", "resolved_yes"]],
        on="condition_id", how="left",
    )
    t = t.merge(ev[["event_id", "target_date"]], on="event_id", how="left")

    # signed size: +size = holding YES, -size = holding NO
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

    # per (wallet, condition) net position
    key = ["proxy_wallet", "condition_id"]
    net = t.groupby(key, sort=False).agg(
        signed_size=("signed_size", "sum"),
        cash=("cash", "sum"),
        n_trades=("signed_size", "size"),
        event_id=("event_id", "first"),
        target_date=("target_date", "first"),
        resolved_yes=("resolved_yes", "first"),
    ).reset_index()
    # split buy/sell volume and counts with vectorized groupby
    t["is_buy"] = t["side"] == "BUY"
    buy = t[t["is_buy"]].groupby(key, sort=False)["usdc"].sum().rename("buy_usdc")
    sell = t[~t["is_buy"]].groupby(key, sort=False)["usdc"].sum().rename("sell_usdc")
    n_buy = t[t["is_buy"]].groupby(key, sort=False).size().rename("n_buys")
    n_sell = t[~t["is_buy"]].groupby(key, sort=False).size().rename("n_sells")
    net = net.merge(buy, on=key, how="left")
    net = net.merge(sell, on=key, how="left")
    net = net.merge(n_buy, on=key, how="left")
    net = net.merge(n_sell, on=key, how="left")
    net[["buy_usdc", "sell_usdc"]] = net[["buy_usdc", "sell_usdc"]].fillna(0.0)
    net[["n_buys", "n_sells"]] = net[["n_buys", "n_sells"]].fillna(0).astype(int)
    net["target_date"] = pd.to_datetime(net["target_date"], errors="coerce")
    print(f"[net] {len(net):,} (wallet, condition) positions", flush=True)

    resolved = net["resolved_yes"].notna()
    won = net["resolved_yes"].astype(bool).fillna(False).values
    net_size = net["signed_size"].values
    payout = np.where(
        won,
        np.maximum(net_size, 0.0),
        np.maximum(-net_size, 0.0),
    )
    payout = np.where(resolved.values, payout, 0.0)
    net["payout"] = payout
    net["pnl"] = net["cash"] + net["payout"]
    net["is_resolved"] = resolved
    net["won"] = won & resolved.values
    net["profitable"] = resolved.values & (net["pnl"].values > 0)

    # per-wallet aggregation
    u = net.groupby("proxy_wallet").agg(
        n_conditions=("condition_id", "size"),
        n_resolved=("is_resolved", "sum"),
        n_unresolved=("is_resolved", lambda x: int((~x).sum())),
        n_events=("event_id", "nunique"),
        n_days=("target_date", "nunique"),
        first_trade_date=("target_date", "min"),
        last_trade_date=("target_date", "max"),
        n_trades=("n_trades", "sum"),
        n_buys=("n_buys", "sum"),
        n_sells=("n_sells", "sum"),
        buy_usdc=("buy_usdc", "sum"),
        sell_usdc=("sell_usdc", "sum"),
        cash=("cash", "sum"),
        payout=("payout", "sum"),
        pnl=("pnl", "sum"),
        n_won=("won", "sum"),
        n_profitable=("profitable", "sum"),
    ).reset_index()
    u["gross_usdc"] = u["buy_usdc"] + u["sell_usdc"]
    u["net_usdc"] = u["buy_usdc"] - u["sell_usdc"]
    u["win_rate"] = u["n_won"] / u["n_resolved"].replace(0, np.nan)
    u["profitable_rate"] = u["n_profitable"] / u["n_resolved"].replace(0, np.nan)
    u["avg_pnl_per_condition"] = u["pnl"] / u["n_resolved"].replace(0, np.nan)

    u = u.sort_values("pnl", ascending=False).reset_index(drop=True)
    OUT.joinpath("klga_wallet_stats.csv").write_text(
        u.to_csv(index=False), encoding="utf-8"
    )
    print(f"[write] {OUT / 'klga_wallet_stats.csv'} ({len(u):,} wallets)")

    summary = {
        "n_artifacts": len(ARTS),
        "n_trades": int(len(tr)),
        "n_conditions": int(len(mk)),
        "n_conditions_with_trades": int(net["condition_id"].nunique()),
        "n_wallet_positions": int(len(net)),
        "n_wallets": int(len(u)),
        "n_events": int(net["event_id"].nunique()),
        "n_resolved_conditions": int(net["is_resolved"].sum()),
        "n_unresolved_conditions": int((~net["is_resolved"]).sum()),
        "gross_usdc_total": float(u["gross_usdc"].sum()),
        "buy_usdc_total": float(u["buy_usdc"].sum()),
        "sell_usdc_total": float(u["sell_usdc"].sum()),
        "pnl_total_all_wallets": float(u["pnl"].sum()),
        "n_profitable_wallets": int((u["pnl"] > 0).sum()),
        "n_losing_wallets": int((u["pnl"] < 0).sum()),
        "date_range": [
            str(net["target_date"].min()),
            str(net["target_date"].max()),
        ],
    }
    OUT.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)

    # top-line preview
    cols = [
        "proxy_wallet", "n_trades", "n_conditions", "gross_usdc",
        "buy_usdc", "sell_usdc", "pnl", "win_rate", "profitable_rate",
    ]
    print("\n== top 15 by PNL ==")
    print(u[cols].head(15).to_string(index=False))
    print("\n== bottom 5 by PNL ==")
    print(u[cols].tail(5).to_string(index=False))
    print("\n== largest volume ==")
    print(u.sort_values("gross_usdc", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
