"""Find lottery-bucket sellers across all cities and analyze strategies.

Window: first 4h after market open, YES-view price < 5c.
Seller = taker who SELLs YES or BUYs NO on a lottery bucket.
Computes hold-to-settlement P/L per seller using resolved_yes, classifies
strategies (ONE-OFF / REPEAT / SPRAYER / PRO), and saves per-city CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"d:\NN\PolyWeatherData\data_store\backtest\clob")
CITIES = {
    "LONDON": "LONDON/20250121_20260810",
    "CHICAGO": "CHICAGO/20251204_20260810",
    "DALLAS": "DALLAS/20250101_20260812",
    "MIAMI": "MIAMI/20250101_20260812",
    "PHOENIX": "PHOENIX/20250101_20260812",
    "DENVER": "DENVER/20250101_20260812",
    "SEATTLE": "SEATTLE/20250101_20260812",
    "ATLANTA": "ATLANTA/20250101_20260812",
}
OPEN_H = 4.0
LOT = 0.05
OUT = Path(r"d:\NN\PolyWeatherData\analysis_lottery_sellers")


def analyze_city(city: str, rel: str) -> tuple[pd.DataFrame, dict]:
    root = BASE / rel
    tr = pd.read_parquet(
        root / "trades.parquet",
        columns=["condition_id", "timestamp", "outcome", "side", "price", "size",
                 "proxy_wallet", "usdc"],
    )
    mk = pd.read_parquet(
        root / "markets.parquet",
        columns=["event_id", "condition_id", "market_start_at", "resolved_yes"],
    )
    tr = tr.merge(
        mk[["condition_id", "market_start_at", "event_id", "resolved_yes"]],
        on="condition_id", how="left",
    )
    tr["since_open_min"] = (
        tr["timestamp"] - tr["market_start_at"]
    ).dt.total_seconds() / 60.0
    tr["yes_view"] = tr["price"].where(
        tr["outcome"] == "YES", 1.0 - tr["price"]
    )

    early = tr[
        (tr["since_open_min"] >= 0)
        & (tr["since_open_min"] <= OPEN_H * 60)
        & (tr["yes_view"] < LOT)
    ].copy()
    is_sell = ((early["outcome"] == "YES") & (early["side"] == "SELL")) | (
        (early["outcome"] == "NO") & (early["side"] == "BUY")
    )
    s = early[is_sell].copy()
    if len(s) == 0:
        return pd.DataFrame(), {"city": city, "sellers": 0}

    sell_yes = (s["outcome"] == "YES") & (s["side"] == "SELL")
    w = s["resolved_yes"].fillna(False).values
    pnl_sell_yes = np.where(w, -s["size"] + s["price"] * s["size"],
                            s["price"] * s["size"])
    pnl_buy_no = np.where(w, -s["price"] * s["size"],
                          s["size"] - s["price"] * s["size"])
    s["pnl"] = np.where(sell_yes, pnl_sell_yes, pnl_buy_no)
    s["won_bucket"] = s["resolved_yes"].fillna(False).values

    # spray: >=3 distinct lottery conds of one event within 15 min
    spray = s.groupby(["proxy_wallet", "event_id"]).agg(
        n_conds=("condition_id", "nunique"),
        dt_min=("timestamp", lambda x: (x.max() - x.min()).total_seconds() / 60),
    ).reset_index()
    spray["is_spray"] = (spray["n_conds"] >= 3) & (spray["dt_min"] <= 15)
    u_spray = spray.groupby("proxy_wallet")["is_spray"].sum().reset_index()
    u_spray.columns = ["proxy_wallet", "n_spray_events"]

    # roundtrip: bought back same condition at any time
    sells_uc = s.groupby(["proxy_wallet", "condition_id"]).agg(
        sold_usdc=("usdc", "sum")
    ).reset_index()
    buys_all = tr[(tr["outcome"] == "YES") & (tr["side"] == "BUY")].groupby(
        ["proxy_wallet", "condition_id"]
    ).agg(bought_usdc=("usdc", "sum")).reset_index()
    mrg = sells_uc.merge(buys_all, on=["proxy_wallet", "condition_id"], how="left")
    mrg["bought_usdc"] = mrg["bought_usdc"].fillna(0)
    rt = mrg.groupby("proxy_wallet").agg(
        sold_conds=("condition_id", "size"),
        rt_conds=("bought_usdc", lambda x: (x > 0).sum()),
    ).reset_index()
    rt["roundtrip_frac"] = rt["rt_conds"] / rt["sold_conds"]

    u_time = s.groupby("proxy_wallet")["since_open_min"].median().reset_index()
    u_time.columns = ["proxy_wallet", "med_open_min"]

    u = s.groupby("proxy_wallet").agg(
        n_sells=("pnl", "size"),
        sell_usdc=("usdc", "sum"),
        pnl=("pnl", "sum"),
        n_events=("event_id", "nunique"),
        n_won_hits=("won_bucket", "sum"),
        n_win_trades=("pnl", lambda x: (x > 0).sum()),
        avg_size=("size", "mean"),
    ).reset_index()
    u["win_rate_trades"] = u["n_win_trades"] / u["n_sells"]
    u["hit_rate"] = u["n_won_hits"] / u["n_sells"]
    u["sells_per_event"] = u["n_sells"] / u["n_events"]
    u = u.merge(u_spray, on="proxy_wallet", how="left")
    u = u.merge(rt[["proxy_wallet", "roundtrip_frac"]], on="proxy_wallet", how="left")
    u = u.merge(u_time, on="proxy_wallet", how="left")
    u["n_spray_events"] = u["n_spray_events"].fillna(0).astype(int)
    u["roundtrip_frac"] = u["roundtrip_frac"].fillna(0)
    u["city"] = city

    def classify(r: pd.Series) -> str:
        if r["n_sells"] >= 20 and r["n_events"] >= 5:
            return "PRO (scale)"
        if r["n_sells"] >= 5 and (r["n_spray_events"] >= 1
                                  or r["sells_per_event"] >= 3):
            return "SPRAYER"
        if r["n_sells"] >= 5:
            return "REPEAT"
        return "ONE-OFF"

    u["strategy"] = u.apply(classify, axis=1)

    summary = {
        "city": city,
        "sellers": len(u),
        "sold_usdc": float(u["sell_usdc"].sum()),
        "pnl": float(u["pnl"].sum()),
        "pnl_per_usdc": float(u["pnl"].sum() / u["sell_usdc"].sum()),
        "profitable_frac": float((u["pnl"] > 0).mean()),
    }
    return u, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_sellers: list[pd.DataFrame] = []
    for city, rel in CITIES.items():
        u, summ = analyze_city(city, rel)
        if len(u) == 0:
            print(f"[{city}] no sellers", flush=True)
            summaries.append(summ)
            continue
        u.to_csv(OUT / f"sellers_{city}.csv", index=False)
        all_sellers.append(u)
        print(
            f"[{city}] sellers={summ['sellers']} sold=${summ['sold_usdc']:,.0f} "
            f"pnl=${summ['pnl']:,.0f} per$1={summ['pnl_per_usdc']*100:+.2f}c "
            f"profitable={summ['profitable_frac']:.1%}",
            flush=True,
        )
        summaries.append(summ)

    summ_df = pd.DataFrame(summaries)
    summ_df.to_csv(OUT / "sellers_summary_all.csv", index=False)
    combined = pd.concat(all_sellers, ignore_index=True)
    combined.to_csv(OUT / "sellers_all_cities.csv", index=False)
    print("\n=== SUMMARY ===", flush=True)
    print(
        summ_df.to_string(index=False,
                          float_format=lambda x: f"{x:.3f}"),
        flush=True,
    )
    print(f"\ncombined sellers: {len(combined)} saved to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
