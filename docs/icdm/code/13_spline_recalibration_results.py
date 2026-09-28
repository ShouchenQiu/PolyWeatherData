"""Step 13: B-spline recalibration results for the UI comparison.

For each spline capacity (total parameters 7 / 8 / 9, i.e. cubic B-spline
with 2 / 3 / 4 inner knots on logit(p), global group), fit on TRAIN and
score raw vs recalibrated on train / val / test with the event-level
full-coverage scoring from step 10 and bootstrap 95% CI.

Outputs:
  docs/icdm/output/recalibration_spline_results.csv
  (long format: split, version, params, n_events, BS, LogLoss, RPS + CIs)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output"

# capacity ladder: total params -> (n_inner knots), cubic spline + intercept
LADDER = {7: 2, 8: 3, 9: 4}


def _load_step11():
    spec = importlib.util.spec_from_file_location(
        "step11_recalibration", HERE / "11_train_recalibration.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_step11()


def fit_spline(train: pd.DataFrame, n_inner: int) -> dict:
    z = M.logit(train["p"].clip(M.EPS, 1 - M.EPS).to_numpy())
    knots = M.make_knots(z, n_inner, 3)
    X = np.column_stack([np.ones(len(train)),
                         M.bspline_design(z, knots, 3)])
    w = M.fit_logistic(X, train["y"].to_numpy(dtype=float))
    return {"weights": w, "knots": knots}


def apply_spline(df: pd.DataFrame, prm: dict) -> pd.Series:
    z = M.logit(df["p"].clip(M.EPS, 1 - M.EPS).to_numpy())
    X = np.column_stack([np.ones(len(df)),
                         M.bspline_design(z, prm["knots"], 3)])
    return pd.Series(M.sigmoid(X @ prm["weights"]), index=df.index)


def main() -> int:
    panel = pd.read_parquet(M.PANEL)
    split = pd.read_parquet(M.SPLIT)
    panel = panel.merge(split[["event_id", "split"]], on="event_id",
                        how="left")
    panel = panel[panel["split"].notna()].copy()
    lng = M.build_long(panel)
    train = lng[lng["split"] == "train"].copy()

    raw_by_split = {}
    for sn in ["train", "val", "test"]:
        raw_by_split[sn] = M.summary_with_ci(
            M.score_from_long(lng[lng["split"] == sn], "p"))

    rows = []
    for n_params, n_inner in LADDER.items():
        prm = fit_spline(train, n_inner)
        for sn in ["train", "val", "test"]:
            sub = lng[lng["split"] == sn].copy()
            sub["p_hat"] = apply_spline(sub, prm)
            s = M.summary_with_ci(M.score_from_long(sub, "p_hat"))
            r = raw_by_split[sn]
            rows.append({
                "split": sn, "version": f"spline_p{n_params}",
                "params": n_params, "n_events": int(s["n_events"]),
                "BS": round(s["BS"], 5), "LogLoss": round(s["LogLoss"], 5),
                "RPS": round(s["RPS"], 5),
                "BS_lo": round(s["BS_lo"], 5),
                "BS_hi": round(s["BS_hi"], 5),
                "LogLoss_lo": round(s["LogLoss_lo"], 5),
                "LogLoss_hi": round(s["LogLoss_hi"], 5),
                "RPS_lo": round(s["RPS_lo"], 5),
                "RPS_hi": round(s["RPS_hi"], 5),
            })
            print(f"[p{n_params}] {sn:5s} "
                  f"dRPS={r['RPS'] - s['RPS']:+.5f} "
                  f"dBS={r['BS'] - s['BS']:+.5f} "
                  f"dLogL={r['LogLoss'] - s['LogLoss']:+.5f} "
                  f"(n={s['n_events']})", flush=True)
        # raw rows once per split (shared across param counts)
        for sn in ["train", "val", "test"]:
            r = raw_by_split[sn]
            rows.append({
                "split": sn, "version": "raw", "params": n_params,
                "n_events": int(r["n_events"]),
                "BS": round(r["BS"], 5), "LogLoss": round(r["LogLoss"], 5),
                "RPS": round(r["RPS"], 5),
                "BS_lo": round(r["BS_lo"], 5),
                "BS_hi": round(r["BS_hi"], 5),
                "LogLoss_lo": round(r["LogLoss_lo"], 5),
                "LogLoss_hi": round(r["LogLoss_hi"], 5),
                "RPS_lo": round(r["RPS_lo"], 5),
                "RPS_hi": round(r["RPS_hi"], 5),
            })

    res = pd.DataFrame(rows)
    res = res.drop_duplicates(subset=["split", "version", "params"])
    res.to_csv(OUT_DIR / "recalibration_spline_results.csv", index=False)
    print("\n[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
