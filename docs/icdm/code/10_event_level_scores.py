"""Step 10: event-distribution-layer scores (Brier, Log Loss, RPS).

Implements Section 5 of the technical roadmap on the canonical panel
(every node clamped to [2025-12-05, 2026-08-10]; DENVER/DALLAS excluded
as outliers).

Method (strict per roadmap 5.1-5.4):
  1. For each horizon, keep only FULL-COVERAGE events: every temperature
     bucket of the event has a price at that horizon. Sparse price
     coverage would otherwise bias the normalized distribution.
  2. S = sum of raw prices over all buckets of the event.
  3. q_k = p_k / S  (normalized market distribution, sum = 1).
  4. BS   = sum_k (q_k - y_k)^2
     LogL = -log(q_winner)
     RPS  = sum_{k=1}^{n-1} (F_k - O_k)^2   (F = cumsum q, O = cumsum y)
  5. Average each score across events; report n_events and a per-event
     bootstrap 95% CI (resample whole events, roadmap 7.3).

Outputs:
  docs/icdm/output/event_layer_scores.csv
  docs/icdm/output/event_layer_scores_bootstrap.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
PANEL = OUT_DIR / "first_layer_panel_argmax_30min.parquet"
HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
OUTLIERS = ["DENVER", "DALLAS"]
N_BOOT = 2000
SEED = 20260822


def event_scores(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-event normalized scores at one horizon (full-coverage events only)."""
    # Keep ALL buckets of every event (priced or not) so full-coverage
    # detection is correct: n_total = all buckets, n_priced = priced ones.
    sub = panel[["event_id", "city", "bucket_lo_f", col, "y"]].copy()
    sub["has_p"] = sub[col].notna()
    per_ev = sub.groupby("event_id").agg(
        n_total=("has_p", "size"),
        n_priced=("has_p", "sum"),
    )
    full = set(per_ev[per_ev["n_priced"] == per_ev["n_total"]].index)
    sub = sub[sub["event_id"].isin(full)].copy()
    sub = sub[sub[col].notna()].copy()

    # Guard: normal temperature events have 7-11 buckets. Events with fewer
    # buckets (e.g. ATLANTA 106072 whose unresolved buckets were dropped)
    # have a broken bucket structure and must not enter event-level scoring.
    sub = sub[sub.groupby("event_id")["y"].transform("size") >= 7]

    rows = []
    for ev_id, g in sub.groupby("event_id"):
        g = g.sort_values("bucket_lo_f")
        p = g[col].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=int)
        S = p.sum()
        q = p / S
        F = np.cumsum(q)
        O = np.cumsum(y)
        rps = float(((F - O) ** 2).sum())
        bs = float(((q - y) ** 2).sum())
        w = int(np.argmax(y))
        logl = float(-np.log(q[w])) if q[w] > 0 else np.inf
        rows.append({
            "event_id": ev_id,
            "city": g["city"].iloc[0],
            "n_buckets": int(len(g)),
            "S": float(S),
            "q_winner": float(q[w]),
            "BS": bs,
            "LogLoss": logl,
            "RPS": rps,
        })
    return pd.DataFrame(rows)


def main() -> int:
    panel = pd.read_parquet(PANEL)
    panel = panel[~panel["city"].isin(OUTLIERS)].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    summ, boot = [], []
    for h in HORIZONS:
        col = f"p_{int(h)}h"
        df = event_scores(panel, col)
        n = len(df)
        if n == 0:
            print(f"[skip] {int(h)}h: no full-coverage events", flush=True)
            continue
        m = df[["BS", "LogLoss", "RPS"]].mean()

        # bootstrap 95% CI, resampling whole events
        cis = {}
        for metric in ["BS", "LogLoss", "RPS"]:
            vals = df[metric].to_numpy()
            draws = vals[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            cis[metric] = (lo, hi)

        summ.append({
            "horizon_h": int(h),
            "n_events": n,
            "S_mean": float(df["S"].mean()),
            "S_median": float(df["S"].median()),
            "BS": m["BS"],
            "LogLoss": m["LogLoss"],
            "RPS": m["RPS"],
            "BS_ci_lo": cis["BS"][0], "BS_ci_hi": cis["BS"][1],
            "LogL_ci_lo": cis["LogLoss"][0], "LogL_ci_hi": cis["LogLoss"][1],
            "RPS_ci_lo": cis["RPS"][0], "RPS_ci_hi": cis["RPS"][1],
        })
        print(
            f"[done] {int(h)}h: n={n:4d} S={float(df['S'].mean()):.3f} "
            f"BS={m['BS']:.4f} LogL={m['LogLoss']:.3f} RPS={m['RPS']:.4f}",
            flush=True,
        )

    out_s = pd.DataFrame(summ)
    out_s.to_csv(OUT_DIR / "event_layer_scores.csv", index=False)

    # per-event long table (for scatter / reuse)
    long_frames = []
    for h in HORIZONS:
        col = f"p_{int(h)}h"
        df = event_scores(panel, col)
        if len(df):
            df["horizon_h"] = int(h)
            long_frames.append(df)
    out_l = pd.concat(long_frames, ignore_index=True)
    out_l.to_csv(OUT_DIR / "event_layer_scores_long.csv", index=False)

    print(f"\n[done] wrote event_layer_scores.csv + event_layer_scores_long.csv",
          flush=True)
    print(out_s.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
