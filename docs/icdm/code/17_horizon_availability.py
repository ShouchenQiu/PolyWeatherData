"""Step 17: availability and band-composition diagnostics by horizon.

Motivation
----------
Two mechanical questions about the horizon grid:

  Q1 (far end) Does the +/-0.5h window capture trades at far horizons?
     Markets are often listed late, so at 48h many buckets (and some whole
     events) have no trade at all in [47.5h, 48.5h]. If so, the far-horizon
     sample is not only smaller but a SELECTED subset of events.

  Q2 (near end) Do the two price bands (long-shot p<=0.10, favorite
     p in [0.75,0.95]) hold the same kind of bucket as the peak approaches?
     As the outcome becomes nearly known, losing buckets collapse toward 0
     and the winning bucket drifts up, so buckets MIGRATE between bands.
     The band is then not the same population at 48h and at 1h.

This step quantifies both, using the canonical panel (no new downloads):
  - per horizon: priced buckets / total buckets, events with >=1 priced
    bucket, events with ZERO priced buckets (dropped from that horizon)
  - the "common event set": events priced at ALL horizons (what a balanced
    panel would leave)
  - per horizon x band: buckets in band, winners in band, winner density
    (which equals mean(y) and therefore drives D_h mechanically)
  - winner-bucket migration across price bands

Outputs:
  docs/icdm/output/horizon_availability.csv
  docs/icdm/output/band_composition.csv
  docs/icdm/output/winner_migration.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
PANEL = OUT_DIR / "first_layer_panel_argmax_30min.parquet"
HORIZONS = [48, 42, 36, 30, 24, 18, 12, 9, 6, 3, 1]
OUTLIERS = ["DENVER", "DALLAS"]
BANDS = [("long-shot", 0.0, 0.10), ("favorite", 0.75, 0.95)]
# winner-migration bins: below/inside the two bands, above, and the middle
MIG_BINS = [(-np.inf, 0.10, "<=0.10"), (0.10, 0.75, "(0.10,0.75)"),
            (0.75, 0.95, "[0.75,0.95]"), (0.95, np.inf, ">0.95")]


def main() -> int:
    panel = pd.read_parquet(PANEL)
    panel = panel[~panel["city"].isin(OUTLIERS)].reset_index(drop=True)
    cols = {h: f"p_{h}h" for h in HORIZONS}

    n_bucket_total = len(panel)
    ev_total = panel["event_id"].nunique()

    # ---- Q1: availability -------------------------------------------------
    rows = []
    for h in HORIZONS:
        col = cols[h]
        priced = panel[col].notna()
        per_ev = panel.groupby("event_id")[col].apply(
            lambda s: int(s.notna().sum()))
        n_ev_priced = int((per_ev > 0).sum())
        full = panel.groupby("event_id")[col].apply(
            lambda s: bool(s.notna().all()))
        rows.append({
            "horizon_h": h,
            "buckets_priced": int(priced.sum()),
            "buckets_total": n_bucket_total,
            "bucket_coverage_pct": round(100 * priced.sum() / n_bucket_total, 1),
            "buckets_unpriced_pct": round(
                100 * (1 - priced.sum() / n_bucket_total), 1),
            "events_priced": n_ev_priced,
            "events_total": ev_total,
            "events_missing_pct": round(
                100 * (ev_total - n_ev_priced) / ev_total, 1),
            "events_full_coverage": int(full.sum()),
            "events_full_coverage_pct": round(100 * full.sum() / ev_total, 1),
        })
    avail = pd.DataFrame(rows)
    avail.to_csv(OUT_DIR / "horizon_availability.csv", index=False)

    # the balanced / common panel: events priced at EVERY horizon
    common_ev = None
    for h in HORIZONS:
        s = set(panel.loc[panel[cols[h]].notna(), "event_id"])
        common_ev = s if common_ev is None else (common_ev & s)
    print(f"[common] events priced at ALL {len(HORIZONS)} horizons: "
          f"{len(common_ev)} of {ev_total} "
          f"({100*len(common_ev)/ev_total:.1f}%)", flush=True)

    # ---- Q2: band composition --------------------------------------------
    rows = []
    for h in HORIZONS:
        col = cols[h]
        d = panel[panel[col].notna()]
        for name, lo, hi in BANDS:
            b = d[(d[col] >= lo) & (d[col] <= hi)]
            n = len(b)
            nw = int((b["y"] == 1).sum())
            rows.append({
                "horizon_h": h, "band": name,
                "n_buckets": n,
                "n_events": b["event_id"].nunique(),
                "n_winners": nw,
                "winner_density_pct": round(100 * nw / n, 2) if n else np.nan,
                "mean_p": round(float(b[col].mean()), 4) if n else np.nan,
                "mean_y": round(float(b["y"].mean()), 4) if n else np.nan,
                "D": round(float(b["y"].mean() - b[col].mean()), 4) if n else np.nan,
            })
    band = pd.DataFrame(rows)
    band.to_csv(OUT_DIR / "band_composition.csv", index=False)

    # ---- Q2b: winner migration -------------------------------------------
    rows = []
    for h in HORIZONS:
        col = cols[h]
        w = panel[(panel[col].notna()) & (panel["y"] == 1)]
        n = len(w)
        rec = {"horizon_h": h, "n_winners": n}
        for lo, hi, lab in MIG_BINS:
            cnt = int(((w[col] > lo) & (w[col] <= hi)).sum())
            rec[f"share_{lab}_pct"] = round(100 * cnt / n, 1) if n else np.nan
            rec[f"n_{lab}"] = cnt
        rec["winner_price_median"] = round(float(w[col].median()), 4)
        rows.append(rec)
    mig = pd.DataFrame(rows)
    mig.to_csv(OUT_DIR / "winner_migration.csv", index=False)

    # ---- print ------------------------------------------------------------
    print()
    print("=" * 104)
    print("Q1. AVAILABILITY BY HORIZON (does the window capture trades?)")
    print("=" * 104)
    print(f"{'h':>4s} {'buckets':>9s} {'unpriced%':>10s} {'events':>8s} "
          f"{'missing%':>9s} {'full cov events':>16s}")
    for _, r in avail.iterrows():
        print(f"{int(r['horizon_h']):4d} {int(r['buckets_priced']):9d} "
              f"{r['buckets_unpriced_pct']:9.1f}% {int(r['events_priced']):8d} "
              f"{r['events_missing_pct']:8.1f}% "
              f"{int(r['events_full_coverage']):16d}")

    print()
    print("=" * 104)
    print("Q2. BAND COMPOSITION BY HORIZON (is the band the same population?)")
    print("=" * 104)
    for name, _, _ in BANDS:
        print(f"\n-- {name} --")
        print(f"{'h':>4s} {'buckets':>9s} {'events':>8s} {'winners':>9s} "
              f"{'winner density%':>16s} {'mean_p':>8s} {'D':>9s}")
        sub = band[band["band"] == name]
        for _, r in sub.iterrows():
            print(f"{int(r['horizon_h']):4d} {int(r['n_buckets']):9d} "
                  f"{int(r['n_events']):8d} {int(r['n_winners']):9d} "
                  f"{r['winner_density_pct']:15.2f}% {r['mean_p']:8.4f} "
                  f"{r['D']:+9.4f}")

    print()
    print("=" * 104)
    print("Q2b. WHERE DO THE WINNER BUCKETS SIT? (migration)")
    print("=" * 104)
    print(f"{'h':>4s} {'winners':>8s} "
          + " ".join(f"{lab:>12s}" for _, _, lab in MIG_BINS)
          + f" {'median price':>13s}")
    for _, r in mig.iterrows():
        print(f"{int(r['horizon_h']):4d} {int(r['n_winners']):8d} "
              + " ".join(f"{r[f'share_{lab}_pct']:11.1f}%"
                         for _, _, lab in MIG_BINS)
              + f" {r['winner_price_median']:13.3f}")

    print(f"\n[write] {OUT_DIR / 'horizon_availability.csv'}")
    print(f"[write] {OUT_DIR / 'band_composition.csv'}")
    print(f"[write] {OUT_DIR / 'winner_migration.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
