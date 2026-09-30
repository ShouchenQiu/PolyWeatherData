"""Step 8: raw order-data statistics per station.

Computes order-level statistics for every canonical CLOB artifact and
writes docs/icdm/output/raw_trade_stats.csv (also shown on the Data
Overview UI page).

Metrics per station:
  - date range of target dates
  - events, conditions (markets), trades
  - trades per event, YES/NO splits, BUY/SELL splits
  - total size, total USDC, median price, unique wallets
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLOB_ROOT, OUT_DIR, ensure_out_dir  # noqa: E402
CLOB = CLOB_ROOT
OUT = OUT_DIR / "raw_trade_stats.csv"

NODES: dict[str, tuple[str, str]] = {
    "NYC": ("KLGA", "20251205_20260810"),
    "LONDON": ("LONDON", "20251205_20260810"),
    "CHICAGO": ("CHICAGO", "20251205_20260810"),
    "DALLAS": ("DALLAS", "20251205_20260810"),
    "MIAMI": ("MIAMI", "20251205_20260810"),
    "DENVER": ("DENVER", "20251205_20260810"),
    "SEATTLE": ("SEATTLE", "20251205_20260810"),
    "ATLANTA": ("ATLANTA", "20251205_20260810"),
    "TOKYO": ("TOKYO", "20251205_20260810"),
    "SHANGHAI": ("SHANGHAI", "20251205_20260810"),
    "SEOUL": ("SEOUL", "20251205_20260810"),
    "TORONTO": ("TORONTO", "20251205_20260810"),
}


def main() -> int:
    rows = []
    for city, (st, art) in NODES.items():
        root = CLOB / st / art
        ev = pd.read_parquet(root / "events.parquet", columns=["event_id", "target_date"])
        ev["target_date"] = pd.to_datetime(ev["target_date"])
        mk = pd.read_parquet(root / "markets.parquet")
        tr = pd.read_parquet(root / "trades.parquet")
        ts = pd.to_datetime(tr["timestamp"])
        side = tr["side"].str.upper()
        n_ev = ev["event_id"].nunique()
        n_mk = mk["condition_id"].nunique()
        n_tr = len(tr)
        rows.append({
            "city": city,
            "station": st,
            "date_min": ev["target_date"].min().date(),
            "date_max": ev["target_date"].max().date(),
            "events": n_ev,
            "conditions": n_mk,
            "trades": n_tr,
            "trades_per_event": round(n_tr / n_ev, 1),
            "yes_trades": int((tr["outcome"] == "YES").sum()),
            "no_trades": int((tr["outcome"] == "NO").sum()),
            "buy": int((side == "BUY").sum()),
            "sell": int((side == "SELL").sum()),
            "size_total": round(float(tr["size"].sum()), 0),
            "usdc_total": round(float(tr["usdc"].sum()), 0),
            "price_median": round(float(tr["price"].median()), 4),
            "unique_wallets": int(tr["proxy_wallet"].nunique()),
            "t_first": ts.min(),
            "t_last": ts.max(),
        })
        print(
            f"[stats] {city}: events={n_ev} conditions={n_mk} "
            f"trades={n_tr:,} usdc={rows[-1]['usdc_total']:,.0f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    tot = df[["events", "conditions", "trades", "usdc_total"]].sum()
    print(
        f"\n[done] wrote {OUT}\n"
        f"TOTAL: events={int(tot['events'])} conditions={int(tot['conditions'])} "
        f"trades={int(tot['trades']):,} usdc={tot['usdc_total']:,.0f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
