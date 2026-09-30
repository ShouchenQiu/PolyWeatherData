"""Favorite Buy-NO by fixed signal horizon h (10 cities, paper Fig. S1 support).

Same trade model as 50_cross_city_buy_no.py, but instead of pooling the whole
1-12 h window it reports one row per horizon so the reader can see *where*
inside the window the favorite edge sits.

SIGNAL DEFINITION
-----------------
Horizon h is hours before the daily temperature peak (argmax time), so prices
come from the canonical panel (`p_{h}h` columns) exactly as in 50_*.py.

For each h in {1, 3, 6, 9, 12}: buckets whose price at that horizon lies in
[0.75, 0.95] are bought on the NO side at (1 - p), $10 notional, held to
settlement.  Restricting to h <= 12 keeps this inside the favorite window.

Fee = Theta * C * p * (1 - p), Theta = 0.05, C = shares, p = traded price.
Event-block bootstrap CI (500 draws) on post-fee PnL per $1 notional.

VERIFICATION STATUS
-------------------
Not yet run on the real panel (raw artifacts are not in the repository);
exercised only on a synthetic fixture.  Treat the output as a structural
companion to Table III, not as a re-derivation of it.

Output: <OUT_DIR>/favorite_fixed_h_bins.csv
"""
from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR, PANEL, SPLIT  # noqa: E402

OUT = OUT_DIR
NOTIONAL = 10.0
FEE_RATE = 0.05  # Theta in Fee = Theta * C * p * (1 - p)
P_LO, P_HI = 0.75, 0.95
HS = [1, 3, 6, 9, 12]     # horizons before the peak, inside the 1-12 h window
NBOOT = 500
SEED = 20260822
TEST_SPLIT = "test"


def trades_at(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Favorite-priced buckets at exactly horizon h, with NO-leg economics."""
    col = f"p_{h}h"
    s = panel[panel[col].notna()].copy()
    s = s[(s[col] >= P_LO) & (s[col] <= P_HI)]
    if not len(s):
        return s
    s["signal_price"] = s[col]
    s["no_price"] = 1.0 - s["signal_price"]
    s["shares"] = NOTIONAL / s["no_price"]
    s["cost"] = NOTIONAL
    s["fee"] = FEE_RATE * s["shares"] * s["no_price"] * (1.0 - s["no_price"])
    s["won"] = ~s["y"].astype(bool)
    s["payout"] = s["shares"].where(s["won"], 0.0)
    s["pnl_pre"] = s["payout"] - s["cost"]
    s["pnl_post"] = s["payout"] - s["cost"] - s["fee"]
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL)
    split = pd.read_parquet(SPLIT)
    panel = panel.merge(split[["event_id", "split"]], on="event_id",
                        how="left")
    test = panel[panel["split"] == TEST_SPLIT].copy()
    print(f"[panel] {PANEL.name}: {test['event_id'].nunique()} test events",
          flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"\n{'h':>3} | {'n':>4} {'events':>6} {'cities':>6} {'win':>7} "
          f"{'pnl_post':>9} {'per1_post':>9} {'ROI%':>8} {'CI(2.5-97.5)':>20}")
    print("-" * 84)

    for h in HS:
        t = trades_at(test, h)
        if not len(t):
            print(f"{h:>3} |     0      0      0    n/a "
                  f"{'-':>9} {'-':>9} {'-':>8} {'-':>20}")
            continue
        n = len(t)
        pnl = float(t["pnl_post"].sum())
        per1 = pnl / (n * NOTIONAL)
        eids = t["event_id"].unique()
        dist = np.array([
            (lambda sub: sub["pnl_post"].sum() / (len(sub) * NOTIONAL))(
                t[t["event_id"].isin(
                    rng.choice(eids, size=len(eids), replace=True))])
            for _ in range(NBOOT)
        ])
        lo, hi = np.percentile(dist, [2.5, 97.5])
        print(f"{h:>3} | {n:>4} {t['event_id'].nunique():>6} "
              f"{t['city'].nunique():>6} {t['won'].mean():>7.3f} "
              f"{pnl:>9.2f} {per1:>9.4f} {per1*100:>8.1f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}")
        rows.append({
            "h": h, "n_trades": n,
            "n_events": int(t["event_id"].nunique()),
            "n_cities": int(t["city"].nunique()),
            "win_rate": round(float(t["won"].mean()), 4),
            "pnl_post": round(pnl, 2),
            "per1_post": round(per1, 4),
            "roi_post_pct": round(per1 * 100, 2),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
        })

    out = pd.DataFrame(rows)
    OUT.joinpath("favorite_fixed_h_bins.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'favorite_fixed_h_bins.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
