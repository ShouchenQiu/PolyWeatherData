"""D_h(0.75-0.95) favorite-side bias for KLGA: history (resolution_v2) vs July.

Same panel definition: nearest trade in [h, h+3h] before close, YES-view
price; D_h(a-b) = mean(y - p) over contracts priced in [a, b],
event-averaged. Histories: all pre-July events; July: event_end_at in July.
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
LO, HI = 0.75, 0.95


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
    w = df[(df["hrs_to_close"] >= h) & (df["hrs_to_close"] < h + WIN_H)
           & (df["yes_view"] >= LO) & (df["yes_view"] <= HI)]
    if not len(w):
        return np.nan, 0, 0
    ev = w.groupby(["event_id", "condition_id"], sort=False).agg(
        p=("yes_view", "mean"), y=("resolved_yes", "first")).reset_index()
    ev_agg = ev.groupby("event_id").agg(dp=("p", "mean"), dy=("y", "mean")).reset_index()
    d = (ev_agg["dy"] - ev_agg["dp"]).mean()
    return d, ev["event_id"].nunique(), len(ev)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist = load(HIST_ART)
    july = load(JULY_ART)
    july = july[july["event_end_at"].between("2026-07-01", "2026-07-31")]
    print(f"[load] hist={len(hist):,} july={len(july):,}")

    # also report how many bucket-conditions EVER reach 0.75+ in each
    for name, df in [("HIST", hist), ("JULY", july)]:
        hi_price = df[df["yes_view"] >= 0.75]
        print(f"{name}: trades with yes_view>=0.75 = {len(hi_price):,} "
              f"({hi_price['condition_id'].nunique()} conditions / {hi_price['event_id'].nunique()} events)")

    rows = []
    print(f"\n{'h':>5} | {'D_h(0.75-0.95) hist':>20} {'n_ev':>5} {'n_bkt':>6} | "
          f"{'D_h(0.75-0.95) july':>20} {'n_ev':>5} {'n_bkt':>6}")
    print("-" * 76)
    for h in HORIZONS:
        dh_hist, nev_h, nb_h = d_h(hist, h)
        dh_jul, nev_j, nb_j = d_h(july, h)
        print(f"{h:>5.0f} | {dh_hist:>20.4f} {nev_h:>5} {nb_h:>6} | "
              f"{dh_jul:>20.4f} {nev_j:>5} {nb_j:>6}")
        rows.append({"h": h, "d_h_hist": dh_hist, "n_events_hist": nev_h,
                     "n_buckets_hist": nb_h, "d_h_july": dh_jul,
                     "n_events_july": nev_j, "n_buckets_july": nb_j})

    OUT.joinpath("d_h_075_095_hist_vs_july.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    print(f"\n[write] {OUT / 'd_h_075_095_hist_vs_july.csv'}")


if __name__ == "__main__":
    main()
