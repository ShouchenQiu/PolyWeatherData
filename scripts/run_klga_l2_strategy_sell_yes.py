"""KLGA July L2 SELL-YES-on-longshot strategy (analysis only, no codebase change).

Signal: at h hours before settlement, if a bucket's YES mid <= 0.10,
SELL its YES as taker (fill at YES best_bid). Notional $10 on the YES side.
Fee = 5% of notional. Hold to settlement: sold YES wins nothing if bucket
wins (lose), pays 1/share if bucket loses (win). Compare to buy-NO version.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20260701_20260818"
L2_DB = ROOT / "data_store" / "backtest" / "l2" / "live.sqlite"
OUT = ROOT / "analysis_klga_l2_strategy"

HOURS = [72, 48, 36, 24, 12, 6]
NOTIONAL = 10.0
FEE_RATE = 0.05
YES_MID_MAX = 0.10
TOLERANCE = pd.Timedelta(hours=2)


def load_markets() -> pd.DataFrame:
    ev = pd.read_parquet(ART / "events.parquet")
    mk = pd.read_parquet(ART / "markets.parquet")
    ev = ev[ev["event_end_at"].between("2026-07-01", "2026-07-31")]
    ev = ev[["event_id", "target_date", "event_end_at"]]
    mk = mk[["event_id", "bucket_lo_f", "yes_token_id", "resolved_yes"]]
    df = mk.merge(ev, on="event_id", how="inner")
    df["event_end_at"] = pd.to_datetime(df["event_end_at"], utc=True)
    print(f"[markets] July events={df['event_id'].nunique()} buckets={len(df)}",
          flush=True)
    return df


def load_snapshots(tokens: set[str]) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{L2_DB.resolve()}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=15000")
    df = pd.read_sql_query(
        "SELECT token_id, captured_at, best_bid, best_ask FROM snapshots", con
    )
    con.close()
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df = df[df["token_id"].isin(tokens)]
    print(f"[snapshots] {len(df):,} rows for {df['token_id'].nunique()} tokens",
          flush=True)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mk = load_markets()
    snap = load_snapshots(set(mk["yes_token_id"].dropna())).sort_values(
        ["token_id", "captured_at"]
    )
    rows: list[dict] = []
    for h in HOURS:
        t = mk.copy()
        t["target_time"] = t["event_end_at"] - pd.Timedelta(hours=h)
        t["h"] = h
        y = t.merge(snap, left_on="yes_token_id", right_on="token_id", how="left")
        y = y[y["captured_at"].notna()].copy()
        y["age"] = y["target_time"] - y["captured_at"]
        y = y[(y["age"] >= pd.Timedelta(0)) & (y["age"] <= TOLERANCE)]
        if not len(y):
            print(f"[h={h:>3}] no snapshots", flush=True)
            continue
        y = y.sort_values(["event_id", "h", "yes_token_id", "age"])
        y = y.groupby(["event_id", "h", "yes_token_id"], as_index=False).first()
        y["yes_mid"] = (y["best_bid"] + y["best_ask"]) / 2.0
        tr = y[y["yes_mid"] <= YES_MID_MAX].copy()
        tr = tr[tr["best_bid"].notna()]  # sell YES needs a bid
        tr = tr.rename(columns={"best_bid": "yes_bid"})
        tr["shares"] = NOTIONAL / tr["yes_bid"]
        tr["proceeds"] = tr["shares"] * tr["yes_bid"]  # cash in at sell
        tr["fee"] = FEE_RATE * tr["proceeds"]
        # sold YES: if bucket wins we owe 1/share; if loses we owe 0
        tr["owed"] = tr["shares"].where(tr["resolved_yes"].astype(bool), 0.0)
        tr["won"] = ~tr["resolved_yes"].astype(bool)  # NO wins
        tr["pnl"] = tr["proceeds"] - tr["fee"] - tr["owed"]
        rows.append(tr)
        print(f"[h={h:>3}] trades={len(tr):>3} events={tr['event_id'].nunique():>3} "
              f"pnl={tr['pnl'].sum():>10.2f}", flush=True)

    t = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    t = t.sort_values(["target_date", "h", "bucket_lo_f"]).reset_index(drop=True)
    OUT.joinpath("trades_sell_yes.csv").write_text(
        t.to_csv(index=False), encoding="utf-8"
    )
    print("\n=== sell-YES summary ===")
    if len(t):
        g = t.groupby("h").agg(
            n_trades=("pnl", "size"),
            n_events=("event_id", "nunique"),
            pnl=("pnl", "sum"),
            pnl_per_usd=("pnl", lambda x: x.sum() / (x.size * NOTIONAL)),
            win_rate=("won", "mean"),
        ).reset_index()
        print(g.to_string(index=False))
        print(f"TOTAL trades={len(t)} pnl={t['pnl'].sum():.2f} "
              f"pnl_per_usd={t['pnl'].sum()/(len(t)*NOTIONAL):.4f}")
    else:
        print("NO TRADES")
    print(f"\n[write] {OUT / 'trades_sell_yes.csv'}")


if __name__ == "__main__":
    main()
