"""Recompute D_h(0.10) for KLGA with the panel's exact definition, July vs history.

Panel definition (docs/icdm/code/02_build_first_layer_panel.py):
  price = YES-view price of nearest trade in [h, h+3h] before close
  y = resolved outcome of the condition
  D_h(0.10) = mean(y - p) over contracts with p <= 0.10, event-averaged
  (first average within event, then across events).

Compare: KLGA history (resolution_v2, before 2026-07-01) vs KLGA July
(20260701_20260818 artifact, event_end_at in July). Both use trades.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\NN\PolyWeatherData")
HIST_ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20250401_20260331_resolution_v2"
JULY_ART = ROOT / "data_store" / "backtest" / "clob" / "KLGA" / "20260701_20260818"
OUT = ROOT / "analysis_klga_l2_strategy"

HORIZONS = [48.0, 36.0, 24.0, 12.0, 6.0, 1.0]
WIN_H = 3.0
CUT = 0.10


def load(art: Path) -> pd.DataFrame:
    tr = pd.read_parquet(art / "trades.parquet",
                         columns=["condition_id", "timestamp", "outcome", "side", "price"])
    mk = pd.read_parquet(art / "markets.parquet",
                         columns=["condition_id", "event_id", "resolved_yes"])
    ev = pd.read_parquet(art / "events.parquet",
                         columns=["event_id", "event_end_at"])
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    ev["event_end_at"] = pd.to_datetime(ev["event_end_at"], utc=True)
    df = tr.merge(mk, on="condition_id", how="left").merge(ev, on="event_id", how="inner")
    df = df[df["resolved_yes"].notna()].copy()
    df["yes_view"] = df["price"].where(df["outcome"] == "YES", 1.0 - df["price"])
    df["hrs_to_close"] = (df["event_end_at"] - df["timestamp"]).dt.total_seconds() / 3600.0
    return df


def d_h(df: pd.DataFrame, h: float) -> tuple[float, int, int]:
    lo, hi = h, h + WIN_H
    w = df[(df["hrs_to_close"] >= lo) & (df["hrs_to_close"] < hi) & (df["yes_view"] <= CUT)]
    if not len(w):
        return np.nan, 0, 0
    ev = w.groupby(["event_id", "condition_id"], sort=False).agg(
        p=("yes_view", "mean"), y=("resolved_yes", "first")
    ).reset_index()
    # event-average then across events
    ev_agg = ev.groupby("event_id").agg(
        dp=("p", "mean"), dy=("y", "mean")
    ).reset_index()
    d = (ev_agg["dy"] - ev_agg["dp"]).mean()
    return d, ev["event_id"].nunique(), len(ev)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist = load(HIST_ART)
    july = load(JULY_ART)
    july = july[july["event_end_at"].between("2026-07-01", "2026-07-31")]
    print(f"[load] hist trades={len(hist):,}  july trades={len(july):,}")

    rows = []
    print(f"\n{'h':>5} | {'D_h hist':>10} {'n_ev':>5} {'n_bkt':>6} | "
          f"{'D_h july':>10} {'n_ev':>5} {'n_bkt':>6}")
    print("-" * 62)
    for h in HORIZONS:
        dh_hist, nev_h, nb_h = d_h(hist, h)
        dh_jul, nev_j, nb_j = d_h(july, h)
        print(f"{h:>5.0f} | {dh_hist:>10.4f} {nev_h:>5} {nb_h:>6} | "
              f"{dh_jul:>10.4f} {nev_j:>5} {nb_j:>6}")
        rows.append({"h": h, "d_h_hist": dh_hist, "n_events_hist": nev_h,
                     "n_buckets_hist": nb_h, "d_h_july": dh_jul,
                     "n_events_july": nev_j, "n_buckets_july": nb_j})

    out = pd.DataFrame(rows)
    OUT.joinpath("d_h_hist_vs_july.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'd_h_hist_vs_july.csv'}")


if __name__ == "__main__":
    main()
