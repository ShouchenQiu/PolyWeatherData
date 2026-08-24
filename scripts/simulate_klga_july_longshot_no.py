"""KLGA July L2 long-shot NO strategy simulation (no calibration).

Fixed rule: for each KLGA July 2026 event, at reference hours h before
settlement (h in {72, 48, 36, 24, 12, 6}), take the L2 snapshot whose
captured_at is closest to (event_end_at - h).  For every bucket whose YES
midpoint (bid+ask)/2 is <= 0.10 and both sides are quoted, BUY its NO with
$10 notional at the NO ask (taker).  Fee = fee_rate * price per share
(rate 0.05, exponent 1.0).  Hold to settlement; NO pays $1/share iff the
bucket did NOT win (resolved_yes == False).

Reports per-h and per-band PNL (pre/post fee), an optimistic mid-price
execution variant, win rate, and skipped (event, h) pairs due to missing
snapshots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20260701_20260818"
L2 = ROOT / "data_store" / "backtest" / "l2" / "live.sqlite"
OUT = ROOT / "analysis_klga_july"

NOTIONAL = 10.0
FEE_RATE = 0.05
HOURS = [72, 48, 36, 24, 12, 6]
BANDS = {"far": [72, 48, 36], "mid": [24, 12], "near": [6]}
SNAP_WINDOW_MIN = 90  # look for snapshot within +/- this many minutes of target
LOTTERY_MID = 0.10


def load_july_markets() -> tuple[pd.DataFrame, pd.DataFrame]:
    mk = pd.read_parquet(
        ART / "markets.parquet",
        columns=[
            "event_id", "condition_id", "market_slug",
            "bucket_lo_f", "bucket_hi_f",
            "yes_token_id", "no_token_id",
            "resolved_yes",
        ],
    )
    ev = pd.read_parquet(
        ART / "events.parquet", columns=["event_id", "event_end_at"]
    )
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    jul = ev[
        (ev["event_end_at"] >= "2026-07-01") & (ev["event_end_at"] < "2026-08-01")
    ].copy()
    mk = mk[mk["event_id"].isin(jul["event_id"])].copy()
    # rows per event with token ids
    print(f"[july] events={len(jul)} markets={len(mk)}", flush=True)
    return mk, jul


def load_snapshots(tokens: set[str]) -> pd.DataFrame:
    import sqlite3

    con = sqlite3.connect(f"file:{L2.resolve()}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=10000")
    placeholders = ",".join("?" * len(tokens))
    snap = pd.read_sql_query(
        "SELECT token_id, captured_at, best_bid, best_ask "
        f"FROM snapshots WHERE token_id IN ({placeholders})",
        con,
        params=sorted(tokens),
    )
    con.close()
    snap["captured_at"] = pd.to_datetime(snap["captured_at"], utc=True)
    print(f"[snapshots] loaded {len(snap):,} rows for {len(tokens)} tokens", flush=True)
    return snap


def nearest_snapshot_time(
    snap_times: np.ndarray, target: pd.Timestamp, window_min: int
) -> pd.Timestamp | None:
    diff = np.abs(snap_times - np.datetime64(target))
    best = int(np.argmin(diff))
    if diff[best] <= np.timedelta64(window_min, "m"):
        return pd.Timestamp(snap_times[best])
    return None


def run_strategy(
    mk: pd.DataFrame,
    jul: pd.DataFrame,
    snap: pd.DataFrame,
    *,
    taker: bool,
) -> pd.DataFrame:
    """Execute the fixed long-shot NO rule; return per-trade records.

    taker=True  -> buy NO at NO ask (realistic).
    taker=False -> buy NO at NO midpoint (optimistic).
    """
    # pivot snapshots: best_bid/best_ask per token per captured_at
    snap_bid = snap.pivot_table(
        index="captured_at", columns="token_id", values="best_bid"
    )
    snap_ask = snap.pivot_table(
        index="captured_at", columns="token_id", values="best_ask"
    )
    all_times = np.sort(snap_bid.index.values)
    print(f"[pivot] {len(all_times)} capture times", flush=True)

    trades: list[dict] = []
    skipped: list[dict] = []
    for _, ev in jul.sort_values("event_end_at").iterrows():
        end = ev["event_end_at"]
        m_ev = mk[mk["event_id"] == ev["event_id"]]
        for h in HOURS:
            target = end - pd.Timedelta(hours=h)
            t0 = nearest_snapshot_time(all_times, target, SNAP_WINDOW_MIN)
            if t0 is None:
                skipped.append(
                    {"event_id": ev["event_id"], "h": h, "reason": "no_snapshot"}
                )
                continue
            row_bid = snap_bid.loc[t0]
            row_ask = snap_ask.loc[t0]
            for _, market in m_ev.iterrows():
                y_tok = market["yes_token_id"]
                n_tok = market["no_token_id"]
                if not y_tok or not n_tok:
                    continue
                y_bid, y_ask = row_bid.get(y_tok), row_ask.get(y_tok)
                n_bid, n_ask = row_bid.get(n_tok), row_ask.get(n_tok)
                if pd.isna(y_bid) or pd.isna(y_ask):
                    continue  # one-sided quote -> missing, per design
                y_mid = (y_bid + y_ask) / 2.0
                if y_mid > LOTTERY_MID:
                    continue
                if pd.isna(n_bid) or pd.isna(n_ask):
                    skipped.append(
                        {
                            "event_id": ev["event_id"],
                            "h": h,
                            "reason": "no_no_quote",
                        }
                    )
                    continue
                n_mid = (n_bid + n_ask) / 2.0
                exec_price = n_ask if taker else n_mid
                shares = NOTIONAL / exec_price
                fee = FEE_RATE * exec_price * shares  # = 5% of notional
                no_won = not bool(market["resolved_yes"])
                payout = shares if no_won else 0.0
                trades.append(
                    {
                        "event_id": ev["event_id"],
                        "target_date": end.normalize().date(),
                        "h": h,
                        "condition_id": market["condition_id"],
                        "bucket": f"{market['bucket_lo_f']}..{market['bucket_hi_f']}",
                        "yes_mid": y_mid,
                        "no_ask": n_ask,
                        "no_mid": n_mid,
                        "exec_price": exec_price,
                        "shares": shares,
                        "fee": fee,
                        "notional": NOTIONAL,
                        "no_won": no_won,
                        "payout": payout,
                        "pnl_pre": payout - NOTIONAL,
                        "pnl_post": payout - NOTIONAL - fee,
                    }
                )
    return pd.DataFrame(trades), skipped


def summarize(trades: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"STRATEGY: long-shot NO (YES mid <= {LOTTERY_MID}), $10/trade, taker at ask, 5% fee")
    print("=" * 78)
    for label, hours in [("ALL", HOURS)] + list(BANDS.items()):
        sub = trades[trades["h"].isin(hours)] if label != "ALL" else trades
        if len(sub) == 0:
            print(f"\n[{label}] no trades")
            continue
        n_ev = sub["event_id"].nunique()
        pnl = sub["pnl_post"].sum()
        pnl_pre = sub["pnl_pre"].sum()
        per_dollar = pnl / sub["notional"].sum()
        per_dollar_pre = pnl_pre / sub["notional"].sum()
        win = sub["no_won"].mean()
        cost = sub["fee"].sum()
        print(
            f"\n[{label}] trades={len(sub):3d} events={n_ev:2d} "
            f"pnl_post=${pnl:9.2f} pnl_pre=${pnl_pre:9.2f} "
            f"fees=${cost:8.2f}"
        )
        print(
            f"         per-$1 post={per_dollar:+.4f} pre={per_dollar_pre:+.4f} "
            f"win_rate={win:.3f} gross_notional=${sub['notional'].sum():,.0f}"
        )

    # per-h breakdown
    print("\n[per-h]")
    for h in HOURS:
        sub = trades[trades["h"] == h]
        if len(sub) == 0:
            print(f"  h={h:2d}: no trades")
            continue
        pnl = sub["pnl_post"].sum()
        print(
            f"  h={h:2d}: trades={len(sub):3d} events={sub['event_id'].nunique():2d} "
            f"pnl_post=${pnl:9.2f} per-$1={pnl/sub['notional'].sum():+.4f}"
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mk, jul = load_july_markets()
    tokens = set(
        mk["yes_token_id"].dropna().tolist() + mk["no_token_id"].dropna().tolist()
    )
    snap = load_snapshots(tokens)

    trades, skipped = run_strategy(mk, jul, snap, taker=True)
    print(f"\n[taker] trades={len(trades)} skipped={len(skipped)}")
    summarize(trades)

    trades_opt, _ = run_strategy(mk, jul, snap, taker=False)
    print(f"\n[optimistic mid] trades={len(trades_opt)}")
    summarize(trades_opt)

    trades.to_csv(OUT / "july_longshot_no_trades.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "july_longshot_no_skipped.csv", index=False)
    print(f"\n[saved] {OUT / 'july_longshot_no_trades.csv'} "
          f"({len(trades)} trades), skipped {len(skipped)}")


if __name__ == "__main__":
    main()
