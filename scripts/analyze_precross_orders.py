"""Analyze user order patterns before the YES-price 0.90 crossing point.

For each city, select up to 30 markets where the YES price first crossed
0.90 (having previously been <= 0.85), sample 50 orders per market from the
window [t_cross - 4h, +inf) (i.e. orders placed 4 hours or more before the
cross), and tabulate order behavior.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

BASE = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob")
CITY_ARTIFACTS = {
    "NYC": r"KLGA\20250401_20260331_final",
    "LONDON": r"LONDON\20250121_20260810",
    "CHICAGO": r"CHICAGO\20251204_20260810",
}
CROSS_PRICE = 0.90
CROSS_PRIOR_FLOOR = 0.85
EARLY_H = 4.0
N_MARKETS = 30
N_ORDERS = 50
SEED = 42


def load_city(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = BASE / CITY_ARTIFACTS[city]
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
            "condition_id",
            "market_slug",
            "bucket_lo_f",
            "bucket_hi_f",
            "market_end_at",
        ],
    )
    return trades, markets


def cross_times(trades: pd.DataFrame) -> pd.Series:
    """Per condition_id: first timestamp with YES price >= 0.90, only for
    markets that were below 0.85 before the cross (a real 'sudden cross')."""
    yes = trades[trades["outcome"] == "YES"]
    crossed = (
        yes[yes["price"] >= CROSS_PRICE]
        .groupby("condition_id")["timestamp"]
        .min()
    )
    was_low = (
        yes[yes["price"] <= CROSS_PRIOR_FLOOR]
        .groupby("condition_id")["timestamp"]
        .min()
    )
    # Only markets that were priced <= 0.85 strictly before their first 0.90.
    valid_index = crossed.index.intersection(was_low.index)
    valid = crossed[valid_index][
        [was_low[c] < crossed[c] for c in valid_index]
    ]
    return valid


def analyze_city(city: str) -> tuple[dict, pd.DataFrame]:
    trades, markets = load_city(city)
    crosses = cross_times(trades)
    print(f"[{city}] markets with sudden 0.90 cross: {len(crosses)}", flush=True)

    cross_map = crosses.to_dict()
    candidates: list[tuple[str, pd.Timestamp, int]] = []
    for cid, t_cross in cross_map.items():
        early = trades[
            (trades["condition_id"] == cid)
            & (trades["timestamp"] <= t_cross - pd.Timedelta(hours=EARLY_H))
        ]
        if len(early) >= N_ORDERS:
            candidates.append((cid, t_cross, len(early)))
    print(f"[{city}] markets with >= {N_ORDERS} early orders: {len(candidates)}", flush=True)

    rng = random.Random(SEED)
    selected = rng.sample(candidates, min(N_MARKETS, len(candidates)))

    sampled_rows: list[dict] = []
    per_market: list[dict] = []
    for cid, t_cross, n_early in selected:
        cond = trades[trades["condition_id"] == cid]
        early = cond[cond["timestamp"] <= t_cross - pd.Timedelta(hours=EARLY_H)]
        sample = early.sample(N_ORDERS, random_state=rng.randrange(2**31))
        mrow = markets[markets["condition_id"] == cid].iloc[0]
        for _, tr in sample.iterrows():
            sampled_rows.append(
                {
                    "city": city,
                    "condition_id": cid,
                    "hours_before_cross": round(
                        (t_cross - tr["timestamp"]).total_seconds() / 3600, 2
                    ),
                    "outcome": tr["outcome"],
                    "side": tr["side"],
                    "price": tr["price"],
                    "size": tr["size"],
                    "usdc": tr["price"] * tr["size"],
                    "proxy_wallet": tr["proxy_wallet"],
                    "bucket": f"{mrow['bucket_lo_f']}-{mrow['bucket_hi_f']}",
                }
            )
        yes_view = sample["price"].where(
            sample["outcome"] == "YES", 1.0 - sample["price"]
        )
        per_market.append(
            {
                "city": city,
                "condition_id": cid[:12],
                "n_early": n_early,
                "t_cross": t_cross,
                "yes_avg_price": round(yes_view.mean(), 3),
                "yes_med_price": round(yes_view.median(), 3),
                "avg_size": round(sample["size"].mean(), 1),
                "buy_frac": round((sample["side"] == "BUY").mean(), 3),
                "hours_before_med": round(
                    (t_cross - sample["timestamp"])
                    .median()
                    .total_seconds()
                    / 3600,
                    2,
                ),
            }
        )

    df = pd.DataFrame(sampled_rows)
    summary = summarize(df, city)
    return summary, pd.DataFrame(per_market)


def summarize(df: pd.DataFrame, city: str) -> dict:
    yes_view = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    orders_per_user = df.groupby("proxy_wallet").size()
    return {
        "city": city,
        "orders": len(df),
        "unique_users": df["proxy_wallet"].nunique(),
        "avg_orders_per_user": round(orders_per_user.mean(), 2),
        "yes_avg_price": round(yes_view.mean(), 3),
        "buy_frac": round((df["side"] == "BUY").mean(), 3),
        "avg_size": round(df["size"].mean(), 1),
        "avg_usdc": round(df["usdc"].mean(), 1),
        "hours_before_cross_med": round(df["hours_before_cross"].median(), 2),
        "hours_before_cross_p75": round(
            df["hours_before_cross"].quantile(0.75), 2
        ),
        "user_max_orders": int(orders_per_user.max()),
    }


if __name__ == "__main__":
    summaries: list[dict] = []
    for city in CITY_ARTIFACTS:
        summary, per_market = analyze_city(city)
        summaries.append(summary)
        out = BASE / ".." / ".." / ".." / "analysis_precross_orders"
        out.mkdir(parents=True, exist_ok=True)
        per_market.to_csv(out / f"precross_per_market_{city}.csv", index=False)
        print(f"[{city}] summary: {summary}", flush=True)

    summ_df = pd.DataFrame(summaries)
    print("\n=== SUMMARY TABLE ===", flush=True)
    print(
        summ_df.to_string(
            index=False,
            columns=[
                "city",
                "orders",
                "unique_users",
                "avg_orders_per_user",
                "yes_avg_price",
                "buy_frac",
                "avg_size",
                "avg_usdc",
                "hours_before_cross_med",
                "hours_before_cross_p75",
                "user_max_orders",
            ],
        ),
        flush=True,
    )
    summ_df.to_csv(
        BASE / ".." / ".." / ".." / "analysis_precross_orders" / "precross_summary.csv",
        index=False,
    )
