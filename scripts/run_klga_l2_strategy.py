"""KLGA July L2 strategy backtest: buy NO on long-shot buckets (YES mid <= 0.10).

Signal: at h hours before settlement, read the L2 snapshot; if a bucket's
YES mid price (bid+ask)/2 <= YES_MID_MAX, buy its NO token as taker.
Execution: fill at NO best ask (taker), Polymarket fee = 0.05 * price * shares
(= 5% of notional since fee_exponent=1). Hold to settlement; payout via
resolved_yes. Reports per-trade detail + per-h and per-band aggregates.
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
BANDS = {"far": (36, 73), "mid": (12, 36), "near": (1, 12)}
NOTIONAL = 10.0
FEE_RATE = 0.05
YES_MID_MAX = 0.10
TOLERANCE = pd.Timedelta(hours=2)  # max age of a snapshot before target


def load_markets() -> pd.DataFrame:
    ev = pd.read_parquet(ART / "events.parquet")
    mk = pd.read_parquet(ART / "markets.parquet")
    ev = ev[ev["event_end_at"].between("2026-07-01", "2026-07-31")]
    ev = ev[["event_id", "target_date", "event_end_at"]]
    mk = mk[["event_id", "market_slug", "bucket_lo_f", "bucket_hi_f",
             "yes_token_id", "no_token_id", "resolved_yes"]]
    df = mk.merge(ev, on="event_id", how="inner")
    df["event_end_at"] = pd.to_datetime(df["event_end_at"], utc=True)
    df["target_date"] = pd.to_datetime(df["target_date"])
    print(f"[markets] July events={df['event_id'].nunique()} buckets={len(df)}",
          flush=True)
    return df


def load_snapshots(tokens: set[str]) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{L2_DB.resolve()}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=15000")
    df = pd.read_sql_query(
        "SELECT token_id, captured_at, best_bid, best_ask FROM snapshots",
        con,
    )
    con.close()
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df = df[df["token_id"].isin(tokens)]
    print(f"[snapshots] loaded {len(df):,} rows for {df['token_id'].nunique()} tokens",
          flush=True)
    return df


def band_of(h: int) -> str:
    for name, (lo, hi) in BANDS.items():
        if lo <= h < hi:
            return name
    return "far"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mk = load_markets()
    tokens = set(mk["yes_token_id"].dropna()) | set(mk["no_token_id"].dropna())
    snap = load_snapshots(tokens).sort_values(["token_id", "captured_at"])

    rows: list[dict] = []
    for h in HOURS:
        target = mk[["event_id", "target_date", "event_end_at", "yes_token_id",
                     "no_token_id", "resolved_yes", "bucket_lo_f"]].copy()
        target["target_time"] = target["event_end_at"] - pd.Timedelta(hours=h)
        target["h"] = h
        # nearest snapshot at or before target_time per token (YES side)
        yes = target.merge(
            snap, left_on="yes_token_id", right_on="token_id", how="left"
        )
        yes = yes[yes["captured_at"].notna()].copy()
        yes["age"] = yes["target_time"] - yes["captured_at"]
        yes = yes[(yes["age"] >= pd.Timedelta(0)) & (yes["age"] <= TOLERANCE)]
        if len(yes):
            yes = yes.sort_values(["event_id", "h", "yes_token_id", "age"])
            yes = yes.groupby(
                ["event_id", "h", "yes_token_id"], as_index=False
            ).first()
        # nearest snapshot for the NO side
        no = target.merge(
            snap, left_on="no_token_id", right_on="token_id", how="left"
        )
        no = no[no["captured_at"].notna()].copy()
        no["age"] = no["target_time"] - no["captured_at"]
        no = no[(no["age"] >= pd.Timedelta(0)) & (no["age"] <= TOLERANCE)]
        if len(no):
            no = no.sort_values(["event_id", "h", "no_token_id", "age"])
            no = no.groupby(
                ["event_id", "h", "no_token_id"], as_index=False
            ).first()

        if not len(yes) or not len(no):
            print(f"[h={h:>3}] no snapshots within tolerance", flush=True)
            continue
        merged = yes.merge(
            no[["event_id", "h", "no_token_id", "captured_at", "best_bid",
                "best_ask"]],
            on=["event_id", "h", "no_token_id"],
            how="inner",
            suffixes=("_yes", "_no"),
        )
        merged["yes_mid"] = (
            merged["best_bid_yes"] + merged["best_ask_yes"]
        ) / 2.0
        trade = merged[merged["yes_mid"] <= YES_MID_MAX].copy()
        trade["no_ask"] = trade["best_ask_no"]
        trade = trade[trade["no_ask"].notna()]
        # notional sizing on the NO side
        trade["shares"] = NOTIONAL / trade["no_ask"]
        trade["cost"] = trade["shares"] * trade["no_ask"]
        trade["fee"] = FEE_RATE * trade["cost"]
        trade["won"] = ~trade["resolved_yes"].astype(bool)  # NO wins
        trade["payout"] = trade["shares"].where(trade["won"], 0.0)
        trade["pnl"] = trade["payout"] - trade["cost"] - trade["fee"]
        rows.append(trade)
        print(f"[h={h:>3}] trades={len(trade):>3} "
              f"events={trade['event_id'].nunique():>3} "
              f"pnl={trade['pnl'].sum():>10.2f}", flush=True)

    t = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    t["band"] = t["h"].map(band_of)
    t["target_date"] = pd.to_datetime(t["target_date"])
    t = t.sort_values(["target_date", "h", "bucket_lo_f"]).reset_index(drop=True)
    OUT.joinpath("trades.csv").write_text(t.to_csv(index=False), encoding="utf-8")

    print("\n=== per-h summary ===")
    if len(t):
        g = t.groupby("h").agg(
            n_trades=("pnl", "size"),
            n_events=("event_id", "nunique"),
            pnl=("pnl", "sum"),
            pnl_per_usd=("pnl", lambda x: x.sum() / (x.size * NOTIONAL)),
            win_rate=("won", "mean"),
        ).reset_index()
        print(g.to_string(index=False))
        print("\n=== per-band summary ===")
        gb = t.groupby("band").agg(
            n_trades=("pnl", "size"),
            n_events=("event_id", "nunique"),
            pnl=("pnl", "sum"),
            pnl_per_usd=("pnl", lambda x: x.sum() / (x.size * NOTIONAL)),
            win_rate=("won", "mean"),
        ).reset_index()
        print(gb.to_string(index=False))
        print(f"\nTOTAL trades={len(t)} events={t['event_id'].nunique()} "
              f"pnl={t['pnl'].sum():.2f} "
              f"pnl_per_usd={t['pnl'].sum()/(len(t)*NOTIONAL):.4f}")
    else:
        print("NO TRADES")
    OUT.joinpath("summary.txt").write_text(
        t.groupby("h").agg(
            n_trades=("pnl", "size"),
            pnl=("pnl", "sum"),
        ).to_string() if len(t) else "NO TRADES",
        encoding="utf-8",
    )
    print(f"\n[write] {OUT / 'trades.csv'}")


if __name__ == "__main__":
    main()
