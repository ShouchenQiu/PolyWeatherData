"""Cross-city Buy-NO diagnostics (10 cities, DENVER/DALLAS excluded).

Reproduces paper Table III: buying NO against overpriced favorites and
against overpriced longshots, in the two windows where each error is
strongest, evaluated on the July test sample.

SIGNAL DEFINITION (paper Section IV-E)
--------------------------------------
The paper defines the horizon h as hours before the *daily temperature peak*
(the argmax time), not before settlement.  Prices therefore come from the
canonical panel built by pipeline/1_panel/14_rebuild_panel_argmax_30min.py,
whose `p_{h}h` columns already hold the mean taker price in the
[h - 0.5 h, h + 0.5 h] window before the peak.

    favorite  1-12 h   before peak, YES price in [0.75, 0.95]  -> buy NO
    longshot  30-48 h  before peak, YES price <= 0.10          -> buy NO

Each row of the panel is one (event, temperature bucket) at one horizon, so
the panel already provides the signal price; no trade-level re-windowing is
needed and the numbers stay on the same price definition as Figs. 1-4.

TRADE MODEL
-----------
$10 notional per trade, held to settlement.  Fee follows the Polymarket
schedule  Fee = Theta * C * p * (1 - p), where C = shares bought,
p = traded token price, Theta = fee rate (0.05).  Per-city and pooled
summary, with an event-block bootstrap CI on post-fee PnL per $1 notional.

VERIFICATION STATUS
-------------------
The raw Polymarket artifacts are not in the repository, so this script has
been exercised on a synthetic panel (correct schema, wrong numbers) rather
than against the real data.  The paper's Table III point estimates
(+50.71% / -0.23%) have NOT been re-derived from this exact code path.
Before relying on the numbers, run it on the real panel and compare against
Table III.  If they differ, the historical implementation used the last
trade inside the window measured from *settlement* (`event_end_at`) rather
than the panel price measured from the peak, which is the one substantive
difference between this version and the original one.

Outputs (written to <OUT_DIR>):
    cross_city_fav_ls_summary.csv         per-city + pooled summary rows
    cross_city_{favorite,longshot}_NO_trades.csv   trade-level detail
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
NBOOT = 500
SEED = 20260822

FAV_LO, FAV_HI = 0.75, 0.95
FAV_WIN = (1, 12)      # horizons (hours before peak) inside the window
LS_MAX = 0.10
LS_WIN = (30, 48)

# Only the July test sample is evaluated (paper Section IV-E).
TEST_SPLIT = "test"


def signals(panel: pd.DataFrame, win: tuple[int, int],
            plo: float, phi: float | None) -> pd.DataFrame:
    """Rows whose price at a horizon inside `win` falls in [plo, phi].

    When several horizons of the same (event, bucket) qualify, the one
    closest to the peak (smallest h) is kept: traders acting on the
    strongest signal would use the latest available price.
    """
    lo, hi = win
    hs = [h for h in (1, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48) if lo <= h <= hi]
    recs = []
    for h in hs:
        col = f"p_{h}h"
        sub = panel[panel[col].notna()].copy()
        sub = sub[(sub[col] >= plo) & (sub[col] <= phi)]
        if not len(sub):
            continue
        sub["h"] = h
        sub["signal_price"] = sub[col]
        recs.append(sub[["event_id", "city", "condition_id", "h",
                         "signal_price", "y"]])
    if not recs:
        return pd.DataFrame(columns=["event_id", "city", "condition_id",
                                     "h", "signal_price", "y"])
    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["event_id", "condition_id", "h"])
    out = out.groupby(["event_id", "condition_id"], as_index=False).first()
    return out


def evaluate(s: pd.DataFrame) -> pd.DataFrame:
    """Attach NO-leg economics and PnL to a signal frame."""
    s = s.copy()
    s["no_price"] = 1.0 - s["signal_price"]
    s["shares"] = NOTIONAL / s["no_price"]
    s["cost"] = NOTIONAL
    s["fee"] = FEE_RATE * s["shares"] * s["no_price"] * (1.0 - s["no_price"])
    s["won"] = ~s["y"].astype(bool)          # NO wins when YES did not resolve
    s["payout"] = s["shares"].where(s["won"], 0.0)
    s["pnl_pre"] = s["payout"] - s["cost"]
    s["pnl_post"] = s["payout"] - s["cost"] - s["fee"]
    return s


def summarize(s: pd.DataFrame, city: str, strat: str) -> dict:
    if not len(s):
        return {"city": city, "strategy": strat, "n_trades": 0, "n_events": 0,
                "win_rate": np.nan, "pnl_pre": 0.0, "per1_pre": np.nan,
                "pnl_post": 0.0, "per1_post": np.nan, "implied_d": np.nan}
    n = len(s)
    return {
        "city": city, "strategy": strat, "n_trades": n,
        "n_events": int(s["event_id"].nunique()),
        "win_rate": round(float(s["won"].mean()), 4),
        "pnl_pre": round(float(s["pnl_pre"].sum()), 2),
        "per1_pre": round(float(s["pnl_pre"].sum() / (n * NOTIONAL)), 4),
        "pnl_post": round(float(s["pnl_post"].sum()), 2),
        "per1_post": round(float(s["pnl_post"].sum() / (n * NOTIONAL)), 4),
        "implied_d": round(float((s["y"] - s["signal_price"]).mean()), 4),
    }


def boot_ci_p1(t: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """Event-block bootstrap CI for post-fee PnL per $1 notional."""
    eids = t["event_id"].unique()
    if not len(eids):
        return np.nan, np.nan
    dist = np.empty(NBOOT)
    for b in range(NBOOT):
        sub = t[t["event_id"].isin(
            rng.choice(eids, size=len(eids), replace=True))]
        dist[b] = sub["pnl_post"].sum() / (len(sub) * NOTIONAL)
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL)
    split = pd.read_parquet(SPLIT)
    panel = panel.merge(split[["event_id", "split"]], on="event_id",
                        how="left")
    test = panel[panel["split"] == TEST_SPLIT].copy()
    print(f"[panel] {PANEL.name}: {panel['event_id'].nunique()} events | "
          f"{test['event_id'].nunique()} in the {TEST_SPLIT} split",
          flush=True)

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    fav_frames, ls_frames = [], []

    for city, csub in test.groupby("city"):
        fav = evaluate(signals(csub, FAV_WIN, FAV_LO, FAV_HI))
        ls = evaluate(signals(csub, LS_WIN, 0.0, LS_MAX))
        rows.append(summarize(fav, city, "favorite_NO"))
        rows.append(summarize(ls, city, "longshot_NO"))
        if len(fav):
            fav_frames.append(fav)
        if len(ls):
            ls_frames.append(ls)
        print(f"  [city] {city}: FAV n={len(fav)}  LS n={len(ls)}", flush=True)

    out = pd.DataFrame(rows)
    OUT.joinpath("cross_city_fav_ls_summary.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")

    print("\n=== per city ===")
    print(out.to_string(index=False))

    print("\n=== pooled (10 cities) ===")
    for strat, frames in [("favorite_NO", fav_frames),
                          ("longshot_NO", ls_frames)]:
        if not frames:
            print(f"{strat}: no trades")
            continue
        t = pd.concat(frames, ignore_index=True)
        n = len(t)
        lo, hi = boot_ci_p1(t, rng)
        print(f"{strat}: n={n} events={t['event_id'].nunique()} "
              f"cities={t['city'].nunique()} "
              f"win_rate={t['won'].mean():.4f} "
              f"pnl_pre={t['pnl_pre'].sum():.2f} "
              f"per1_pre={t['pnl_pre'].sum()/(n*NOTIONAL):.4f} "
              f"pnl_post={t['pnl_post'].sum():.2f} "
              f"per1_post={t['pnl_post'].sum()/(n*NOTIONAL):.4f} "
              f"CI=[{lo:+.4f}, {hi:+.4f}]", flush=True)
        OUT.joinpath(f"cross_city_{strat}_trades.csv").write_text(
            t.to_csv(index=False), encoding="utf-8")

    print(f"\n[write] {OUT / 'cross_city_fav_ls_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
