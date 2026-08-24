"""Track users who traded very-low-probability buckets shortly after open.

Hypothesis to test: some users sell lottery tickets (extreme buckets at
prices < 5c) to speculative buyers and profit from that flow.  We identify
users who, in the first OPEN_WINDOW_H hours after a market opened, traded a
bucket whose YES-view price was < LOTTERY_PRICE.  Then we trace every trade
those users made across all markets in the artifact, net their positions per
(condition, user), settle against the true daily high temperature, and
compare realized P/L vs the whole-population baseline.

NOTE: this is a taker-only view (data-api takerOnly=true).  A 'SELL' row is
the taker actively selling (hitting the bid) - the maker on the other side is
not visible.  Sellers here are therefore takers selling lottery buckets, who
are effectively supplying lottery tickets at the quoted bid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_OPEN_WINDOW_H = 3.0
DEFAULT_LOTTERY_PRICE = 0.05


def load_nyc(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(
        root / "trades.parquet",
        columns=[
            "condition_id",
            "timestamp",
            "outcome",
            "side",
            "price",
            "size",
            "proxy_wallet",
        ],
    )
    markets = pd.read_parquet(
        root / "markets.parquet",
        columns=[
            "event_id",
            "condition_id",
            "bucket_lo_f",
            "bucket_hi_f",
            "market_start_at",
        ],
    )
    events = pd.read_parquet(
        root / "events.parquet", columns=["event_id", "target_date"]
    )
    return trades, markets, events


def settle_flags(markets: pd.DataFrame, events: pd.DataFrame,
                 actuals: pd.DataFrame) -> pd.Series:
    """Return per-condition_id: True if YES resolved (bucket contains truth)."""
    m = markets.merge(events, on="event_id")
    m = m.merge(actuals, on="target_date", how="left")
    won = (
        (m["actual_high_f"] >= m["bucket_lo_f"])
        & (m["actual_high_f"] < m["bucket_hi_f"])
    ).fillna(False)
    return pd.Series(won.values, index=m["condition_id"].values)


def realized_pnl(trades: pd.DataFrame, yes_won: pd.Series) -> pd.DataFrame:
    """Net per (user, condition) cash flow + settlement payout."""
    t = trades.copy()
    # Yes-view signed size: positive = holding YES, negative = holding NO
    yes_buy = (t["outcome"] == "YES") & (t["side"] == "BUY")
    no_sell = (t["outcome"] == "NO") & (t["side"] == "SELL")
    no_buy = (t["outcome"] == "NO") & (t["side"] == "BUY")
    yes_sell = (t["outcome"] == "YES") & (t["side"] == "SELL")
    t["signed_size"] = np.select(
        [yes_buy, no_sell, no_buy, yes_sell],
        [t["size"], t["size"], -t["size"], -t["size"]],
        default=0.0,
    )
    # Cash flow at trade time: pay price*size on BUY, receive on SELL
    t["cash"] = np.where(t["side"] == "BUY", -1.0, 1.0) * t["price"] * t["size"]

    g = t.groupby(["proxy_wallet", "condition_id"])
    net = pd.DataFrame(
        {
            "signed_size": g["signed_size"].sum(),
            "cash": g["cash"].sum(),
        }
    ).reset_index()
    won = net["condition_id"].map(yes_won).fillna(False).values
    net_size = net["signed_size"].values
    payout = np.where(
        won,
        np.maximum(net_size, 0.0),
        np.maximum(-net_size, 0.0),
    )
    net["payout"] = payout
    net["pnl"] = net["cash"] + net["payout"]
    return net


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            r"d:\NN\PolyWeatherData\data_store\backtest\clob\KLGA\20250401_20260331_final"
        ),
    )
    parser.add_argument(
        "--actuals",
        type=Path,
        default=Path(
            r"d:\NN\PolyWeatherData\data_store\backtest\actuals\KLGA\20250401_20260331_final.csv"
        ),
    )
    parser.add_argument(
        "--open-window-h", type=float, default=DEFAULT_OPEN_WINDOW_H
    )
    parser.add_argument(
        "--lottery-price", type=float, default=DEFAULT_LOTTERY_PRICE
    )
    parser.add_argument("--out", type=Path, default=Path(r"d:\NN\PolyWeatherData\analysis_lottery_sellers"))
    args = parser.parse_args(argv)

    trades, markets, events = load_nyc(args.artifact)
    actuals = pd.read_csv(args.actuals)
    yes_won = settle_flags(markets, events, actuals)
    print(f"markets: {len(markets)}, trades: {len(trades)}", flush=True)

    # --- yes-view price ---
    trades["yes_view"] = trades["price"].where(
        trades["outcome"] == "YES", 1.0 - trades["price"]
    )
    # --- opening window: within N hours of market_start_at ---
    t = trades.merge(
        markets[["condition_id", "market_start_at"]], on="condition_id"
    )
    t["since_open_h"] = (
        t["timestamp"] - t["market_start_at"]
    ).dt.total_seconds() / 3600.0

    early = t[
        (t["since_open_h"] >= 0)
        & (t["since_open_h"] <= args.open_window_h)
        & (t["yes_view"] < args.lottery_price)
    ]
    print(
        f"early lottery-window trades: {len(early)} "
        f"(users {early['proxy_wallet'].nunique()})",
        flush=True,
    )

    # seller = taker actively selling a lottery bucket (SELL YES or BUY NO)
    is_sell = ((early["outcome"] == "YES") & (early["side"] == "SELL")) | (
        (early["outcome"] == "NO") & (early["side"] == "BUY")
    )
    sellers = set(early.loc[is_sell, "proxy_wallet"])
    # buyer = taker actively buying a lottery bucket
    is_buy = ((early["outcome"] == "YES") & (early["side"] == "BUY")) | (
        (early["outcome"] == "NO") & (early["side"] == "SELL")
    )
    buyers = set(early.loc[is_buy, "proxy_wallet"])
    print(
        f"sellers in window: {len(sellers)}, buyers in window: {len(buyers)}, "
        f"overlap: {len(sellers & buyers)}",
        flush=True,
    )

    # --- realized P/L over ALL trades (all markets) per user ---
    net = realized_pnl(trades, yes_won)
    user_pnl = (
        net.groupby("proxy_wallet")[["cash", "payout", "pnl"]]
        .sum()
        .reset_index()
    )
    user_pnl["n_conditions"] = net.groupby("proxy_wallet")["condition_id"].nunique().reindex(
        user_pnl["proxy_wallet"]
    ).values
    user_pnl["group"] = user_pnl["proxy_wallet"].map(
        lambda w: (
            "seller"
            if w in sellers and w not in buyers
            else "both" if w in sellers and w in buyers else (
                "buyer" if w in buyers else "other"
            )
        )
    )
    user_pnl["group"] = pd.Categorical(
        user_pnl["group"],
        categories=["seller", "both", "buyer", "other"],
        ordered=True,
    )

    print("\n=== realized P/L by user group (NYC, all markets) ===", flush=True)
    summary_rows = []
    for grp, sub in user_pnl.groupby("group", observed=True):
        row = {
            "group": grp,
            "users": len(sub),
            "total_pnl": round(sub["pnl"].sum(), 0),
            "mean_pnl": round(sub["pnl"].mean(), 2),
            "median_pnl": round(sub["pnl"].median(), 2),
            "win_users": round((sub["pnl"] > 0).mean(), 3),
            "avg_conditions": round(sub["n_conditions"].mean(), 1),
            "total_trades_value": round(-sub["cash"].sum(), 0),
        }
        summary_rows.append(row)
        print(row, flush=True)

    # baseline: all users with any trade
    row = {
        "group": "ALL_USERS",
        "users": len(user_pnl),
        "total_pnl": round(user_pnl["pnl"].sum(), 0),
        "mean_pnl": round(user_pnl["pnl"].mean(), 2),
        "median_pnl": round(user_pnl["pnl"].median(), 2),
        "win_users": round((user_pnl["pnl"] > 0).mean(), 3),
        "avg_conditions": round(user_pnl["n_conditions"].mean(), 1),
        "total_trades_value": round(-user_pnl["cash"].sum(), 0),
    }
    summary_rows.append(row)
    print(row, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    user_pnl.to_csv(args.out / "nyc_user_pnl.csv", index=False)
    early.to_parquet(args.out / "nyc_early_lottery_trades.parquet", index=False)
    (args.out / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote outputs to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
