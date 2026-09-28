"""Step 15: raw scores + EXCESS scores (metric minus its no-skill floor).

Motivation
----------
All three event-level metrics have an inherent bucket-count dependence: a
perfectly calibrated forecaster still scores a positive floor

    E[BS]      = 1 - sum_k q_k^2
    E[LogLoss] = -sum_k q_k log q_k        (entropy H(q))
    E[RPS]     = sum_{k=1}^{n-1} F_k (1 - F_k)

so events with MORE buckets mechanically score worse. Bucket counts differ
across splits (train ~9.5, test ~10.75) and horizons (48h ~10.2, 1h ~9.4),
which contaminates raw cross-split / cross-horizon comparisons.

This step reports side by side:
  - RAW    : the original metric
  - FLOOR  : perfect-calibration expectation given the predicted q
  - EXCESS : RAW - FLOOR (distance from perfect calibration; the bucket
             dependence largely cancels)

for both raw market prices and 7-parameter cubic B-spline recalibrated
prices, aggregated over splits and horizons with event-level bootstrap
95% CIs. Also reports the n-bucket composition and the paired
raw-vs-calibrated change (where the bucket effect cancels exactly).

Outputs:
  docs/icdm/output/excess_scores.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output"
SEED = 20260927
N_BOOT = 2000
EPS = 1e-12
MIN_BUCKETS = 7

METRICS = ["BS", "LogLoss", "RPS"]
# Per-bucket normalizations: raw/n and excess/n. These put the scores on a
# "average per bucket" scale, which is far more intuitive (a single bucket's
# Brier contribution) and is directly comparable to a binary Brier.
NORM_METRICS = ["BS", "LogLoss", "RPS"]
COLS = ([f"{m}_{k}" for m in METRICS for k in ("raw", "floor", "excess")]
        + [f"{m}_{k}_per_bucket"
           for m in NORM_METRICS for k in ("raw", "floor", "excess")])


def _load_step11():
    spec = importlib.util.spec_from_file_location(
        "step11_recalibration", HERE / "11_train_recalibration.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_step11()


def score_with_floor(lng: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Per (event, horizon): raw metrics, no-skill floors, excess."""
    lng = lng[lng[price_col].notna()].copy()
    tot = (lng.groupby("event_id")["bucket_lo_f"].nunique()
           .reset_index(name="n_total"))
    priced = (lng.groupby(["event_id", "h"])[price_col].size()
              .reset_index(name="n_priced"))
    priced = priced.merge(tot, on="event_id", how="left")
    ok = priced[(priced["n_priced"] == priced["n_total"])
                & (priced["n_total"] >= MIN_BUCKETS)][["event_id", "h"]]
    ok["_keep"] = True
    lng = lng.merge(ok, on=["event_id", "h"], how="inner").drop(
        columns=["_keep"])

    split_map = (lng[["event_id", "split"]].drop_duplicates()
                 .set_index("event_id")["split"])
    rows = []
    for (ev, h), g in lng.groupby(["event_id", "h"]):
        g = g.sort_values("bucket_lo_f")
        p = g[price_col].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=int)
        S = p.sum()
        # Guard: the winning bucket must be among the priced buckets (see
        # 11_train_recalibration.py for the rationale).
        if S <= 0 or int(y.sum()) != 1:
            continue
        q = p / S
        n = len(q)
        F = np.cumsum(q)
        O = np.cumsum(y)
        bs = float(((q - y) ** 2).sum())
        ll = float(-np.log(max(q[np.argmax(y)], EPS)))
        rps = float(((F - O) ** 2).sum())
        bs_f = float(1.0 - (q ** 2).sum())
        ll_f = float(-(q * np.log(np.clip(q, EPS, 1.0))).sum())
        rps_f = float((F[:-1] * (1.0 - F[:-1])).sum())
        rows.append({
            "event_id": str(ev), "h": int(h), "n_buckets": n,
            "split": split_map.get(ev),
            "BS_raw": bs, "BS_floor": bs_f, "BS_excess": bs - bs_f,
            "LogLoss_raw": ll, "LogLoss_floor": ll_f,
            "LogLoss_excess": ll - ll_f,
            "RPS_raw": rps, "RPS_floor": rps_f, "RPS_excess": rps - rps_f,
            "BS_raw_per_bucket": bs / n,
            "BS_floor_per_bucket": bs_f / n,
            "BS_excess_per_bucket": (bs - bs_f) / n,
            "LogLoss_raw_per_bucket": ll / n,
            "LogLoss_floor_per_bucket": ll_f / n,
            "LogLoss_excess_per_bucket": (ll - ll_f) / n,
            "RPS_raw_per_bucket": rps / n,
            "RPS_floor_per_bucket": rps_f / n,
            "RPS_excess_per_bucket": (rps - rps_f) / n,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cols: list[str]) -> dict:
    """Mean + event-level bootstrap 95% CI (whole events resampled)."""
    out: dict = {
        "n_events": int(df["event_id"].nunique()),
        "n_rows": int(len(df)),
        "n_buckets_mean": float(df["n_buckets"].mean()),
    }
    ev = df.groupby("event_id")[cols].agg(["sum", "count"])
    n_ev = len(ev)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_ev, size=(N_BOOT, n_ev))
    for c in cols:
        sums = ev[(c, "sum")].to_numpy(dtype=float)
        counts = ev[(c, "count")].to_numpy(dtype=float)
        num = sums[idx].sum(axis=1)
        den = counts[idx].sum(axis=1)
        draws = num / den
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[c] = round(float(df[c].mean()), 5)
        out[f"{c}_lo"] = round(float(lo), 5)
        out[f"{c}_hi"] = round(float(hi), 5)
    return out


def main() -> int:
    panel = pd.read_parquet(M.PANEL)
    split = pd.read_parquet(M.SPLIT)
    panel = panel.merge(split[["event_id", "split"]], on="event_id",
                        how="left")
    panel = panel[panel["split"].notna()].copy()
    lng = M.build_long(panel)

    # ---- fit the final 7-parameter cubic B-spline on TRAIN -------------
    train = lng[lng["split"] == "train"].copy()
    z = M.logit(train["p"].clip(M.EPS, 1 - M.EPS).to_numpy())
    knots = M.make_knots(z, 2, 3)
    X = np.column_stack([np.ones(len(train)),
                         M.bspline_design(z, knots, 3)])
    w = M.fit_logistic(X, train["y"].to_numpy(dtype=float))

    lng = lng.copy()
    zz = M.logit(lng["p"].clip(M.EPS, 1 - M.EPS).to_numpy())
    Xa = np.column_stack([np.ones(len(lng)), M.bspline_design(zz, knots, 3)])
    lng["p_hat"] = M.sigmoid(Xa @ w)

    raws = score_with_floor(lng, "p")
    cals = score_with_floor(lng, "p_hat")

    # ---- n-bucket composition -----------------------------------------
    print("=" * 78)
    print("A. BUCKET-COUNT COMPOSITION (share of rows, by n_buckets)")
    print("=" * 78)
    comp = (pd.concat([raws.assign(variant="raw"),
                       cals.assign(variant="cal")])
            .pivot_table(index="n_buckets", columns="split",
                         values="event_id", aggfunc="count")
            .fillna(0).astype(int))
    print(comp.to_string())
    print()
    for sn in ["train", "val", "test"]:
        s = raws[raws["split"] == sn]
        shares = s["n_buckets"].value_counts(normalize=True).sort_index()
        print(f"  {sn:5s}: mean n = {s['n_buckets'].mean():.2f} | "
              + " | ".join(f"n={int(k)}:{v:.1%}" for k, v in shares.items()))
    print()
    hcomp = raws.groupby("h")["n_buckets"].mean()
    print("  mean n by horizon: " +
          "  ".join(f"{int(h)}h={v:.2f}" for h, v in hcomp.items()))

    # ---- by split: raw vs floor vs excess ------------------------------
    rows_out = []
    for variant, df in [("raw", raws), ("calibrated", cals)]:
        for sn in ["train", "val", "test"]:
            sub = df[df["split"] == sn]
            r = summarize(sub, COLS)
            rows_out.append({"scope_type": "split", "scope_value": sn,
                             "variant": variant, **r})
        r = summarize(df, COLS)
        rows_out.append({"scope_type": "split", "scope_value": "ALL",
                         "variant": variant, **r})
        for h in sorted(df["h"].unique(), reverse=True):
            sub = df[df["h"] == h]
            r = summarize(sub, COLS)
            rows_out.append({"scope_type": "horizon", "scope_value": str(h),
                             "variant": variant, **r})

    # ---- bucket dependence: raw vs excess vs per-bucket ---------------
    dep_rows = []
    for m in METRICS:
        r_raw = float(raws[f"{m}_raw"].corr(raws["n_buckets"]))
        r_exc = float(raws[f"{m}_excess"].corr(raws["n_buckets"]))
        r_nb = float(raws[f"{m}_raw_per_bucket"].corr(raws["n_buckets"]))
        r_ne = float(
            raws[f"{m}_excess_per_bucket"].corr(raws["n_buckets"]))
        dep_rows.append({
            "metric": m,
            "corr_raw": r_raw,
            "slope_raw": float(np.polyfit(raws["n_buckets"],
                                          raws[f"{m}_raw"], 1)[0]),
            "corr_excess": r_exc,
            "slope_excess": float(np.polyfit(raws["n_buckets"],
                                             raws[f"{m}_excess"], 1)[0]),
            "corr_per_bucket": r_nb,
            "corr_excess_per_bucket": r_ne,
        })
    dep = pd.DataFrame(dep_rows)
    dep.to_csv(OUT_DIR / "bucket_dependence.csv", index=False)

    res = pd.DataFrame(rows_out)
    res.to_csv(OUT_DIR / "excess_scores.csv", index=False)

    # ---- print: by split ------------------------------------------------
    for variant in ["raw", "calibrated"]:
        print()
        print("=" * 78)
        print(f"B. BY SPLIT - {variant.upper()} prices")
        print("=" * 78)
        hdr = (f"{'split':6s} {'n_ev':>5s} {'meanN':>6s} | "
               f"{'Brier':>7s} {'floor':>7s} {'excess':>8s} | "
               f"{'LogLoss':>7s} {'floor':>7s} {'excess':>8s} | "
               f"{'RPS':>7s} {'floor':>7s} {'excess':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for _, r in res[(res["scope_type"] == "split")
                        & (res["variant"] == variant)].iterrows():
            print(f"{r['scope_value']:6s} {int(r['n_events']):5d} "
                  f"{r['n_buckets_mean']:6.2f} | "
                  f"{r['BS_raw']:7.4f} {r['BS_floor']:7.4f} "
                  f"{r['BS_excess']:8.4f} | "
                  f"{r['LogLoss_raw']:7.4f} {r['LogLoss_floor']:7.4f} "
                  f"{r['LogLoss_excess']:8.4f} | "
                  f"{r['RPS_raw']:7.4f} {r['RPS_floor']:7.4f} "
                  f"{r['RPS_excess']:8.4f}")

    # ---- print: by horizon (raw) ---------------------------------------
    print()
    print("=" * 78)
    print("C. BY HORIZON - RAW prices (excess columns show the "
          "bucket-free part)")
    print("=" * 78)
    hdr = (f"{'h':>4s} {'n_ev':>5s} {'meanN':>6s} | "
           f"{'Brier':>7s} {'excess':>8s} | "
           f"{'LogLoss':>7s} {'excess':>8s} | "
           f"{'RPS':>7s} {'excess':>8s}")
    print(hdr)
    print("-" * len(hdr))
    hd = res[(res["scope_type"] == "horizon")
             & (res["variant"] == "raw")].copy()
    hd["_h"] = hd["scope_value"].astype(int)
    for _, r in hd.sort_values("_h", ascending=False).iterrows():
        print(f"{int(r['_h']):4d} {int(r['n_events']):5d} "
              f"{r['n_buckets_mean']:6.2f} | "
              f"{r['BS_raw']:7.4f} {r['BS_excess']:8.4f} | "
              f"{r['LogLoss_raw']:7.4f} {r['LogLoss_excess']:8.4f} | "
              f"{r['RPS_raw']:7.4f} {r['RPS_excess']:8.4f}")

    # ---- bucket dependence: raw vs excess -------------------------------
    print()
    print("=" * 78)
    print("D. BUCKET DEPENDENCE: correlation vs n_buckets")
    print("=" * 78)
    print(f"{'metric':10s} {'raw':>9s} {'excess':>9s} {'raw/n':>9s} "
          f"{'excess/n':>9s}")
    for _, r in dep.iterrows():
        print(f"{r['metric']:10s} {r['corr_raw']:+9.3f} "
              f"{r['corr_excess']:+9.3f} {r['corr_per_bucket']:+9.3f} "
              f"{r['corr_excess_per_bucket']:+9.3f}")

    print()
    print("=" * 78)
    print("D2. PER-BUCKET MEANS (market prices) - intuitive scale")
    print("=" * 78)
    print(f"{'split':6s} {'meanN':>6s} | {'Brier/n':>8s} {'floor/n':>8s} "
          f"{'excess/n':>9s} | {'RPS/n':>7s} {'LogLoss/n':>10s}")
    for sn in ["train", "val", "test", "ALL"]:
        s = raws if sn == "ALL" else raws[raws["split"] == sn]
        print(f"{sn:6s} {s['n_buckets'].mean():6.2f} | "
              f"{s['BS_raw_per_bucket'].mean():8.5f} "
              f"{s['BS_floor_per_bucket'].mean():8.5f} "
              f"{s['BS_excess_per_bucket'].mean():+9.5f} | "
              f"{s['RPS_raw_per_bucket'].mean():7.4f} "
              f"{s['LogLoss_raw_per_bucket'].mean():10.4f}")

    # ---- paired raw vs calibrated (bucket effect cancels exactly) -------
    key = ["event_id", "h"]
    mg = raws.merge(cals, on=key, suffixes=("_r", "_c"))
    print()
    print("=" * 78)
    print("E. PAIRED raw -> calibrated (same event-h rows; the bucket "
          "effect cancels exactly)")
    print("=" * 78)
    for sn in ["train", "val", "test"]:
        s = mg[mg["split_r"] == sn]
        parts = []
        for m in METRICS:
            d = (s[f"{m}_raw_r"] - s[f"{m}_raw_c"]).mean()
            de = (s[f"{m}_excess_r"] - s[f"{m}_excess_c"]).mean()
            parts.append(f"{m}: dRaw={d:+.5f} dExcess={de:+.5f}")
        print(f"  {sn:5s} (n={len(s)})  " + " | ".join(parts))
    print()
    print(f"[write] {OUT_DIR / 'excess_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
