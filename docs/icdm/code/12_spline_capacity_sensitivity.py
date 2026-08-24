"""Step 12: spline capacity sensitivity - total parameters 2..10, step 2.

Evaluate how the total number of calibration parameters affects the
three event-level metrics (RPS / Brier / LogLoss) on train / val / test.

Capacity ladder (total columns = intercept + spline basis):
  params=2  : global logit linear     alpha + beta * logit(p)
  params=4  : quadratic B-spline,     0 inner knots (global quadratic)
  params=6  : cubic B-spline,         1 inner knot
  params=8  : cubic B-spline,         3 inner knots (== spline_global,
                                         step 11)
  params=10 : cubic B-spline,         5 inner knots

Knots sit at TRAIN logit(p) quantiles; every fit is global (no horizon
grouping). Raw vs recalibrated scores use the same event-level
full-coverage scoring as steps 10-11. Positive delta = improvement
(raw - recalibrated > 0).

Outputs:
  docs/icdm/output/spline_capacity_results.csv  (long format)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output"

# capacity ladder: name -> (kind, degree, n_inner)
LADDER = [
    ("p2_linear", "linear", None, None),
    ("p4_quadratic", "spline", 2, 0),
    ("p6_cubic_k1", "spline", 3, 1),
    ("p8_cubic_k3", "spline", 3, 3),
    ("p10_cubic_k5", "spline", 3, 5),
]


def _load_step11():
    """Load 11_train_recalibration.py as a module to reuse its helpers."""
    spec = importlib.util.spec_from_file_location(
        "step11_recalibration", HERE / "11_train_recalibration.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_step11()


def fit_linear(train: pd.DataFrame) -> dict:
    p = train["p"].clip(M.EPS, 1 - M.EPS).to_numpy()
    x = np.column_stack([np.ones_like(p), M.logit(p)])
    w = M.fit_logistic(x, train["y"].to_numpy(dtype=float))
    return {"alpha": float(w[0]), "beta": float(w[1])}


def apply_linear(df: pd.DataFrame, prm: dict) -> pd.Series:
    p = df["p"].clip(M.EPS, 1 - M.EPS).to_numpy()
    z = prm["alpha"] + prm["beta"] * M.logit(p)
    return pd.Series(M.sigmoid(z), index=df.index)


def fit_spline_global(train: pd.DataFrame, degree: int,
                      n_inner: int) -> dict:
    z = M.logit(train["p"].clip(M.EPS, 1 - M.EPS).to_numpy())
    knots = M.make_knots(z, n_inner, degree)
    X = np.column_stack([np.ones(len(train)),
                         M.bspline_design(z, knots, degree)])
    w = M.fit_logistic(X, train["y"].to_numpy(dtype=float))
    return {"weights": w, "knots": knots, "degree": degree}


def apply_spline_global(df: pd.DataFrame, prm: dict) -> pd.Series:
    p = df["p"].clip(M.EPS, 1 - M.EPS).to_numpy()
    z = M.logit(p)
    X = np.column_stack([np.ones(len(df)),
                         M.bspline_design(z, prm["knots"], prm["degree"])])
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
    for name, kind, degree, n_inner in LADDER:
        n_params = int(name.split("_")[0][1:])
        if kind == "linear":
            prm = fit_linear(train)

            def apply_fn(df, prm=prm):
                return apply_linear(df, prm)
        else:
            prm = fit_spline_global(train, degree, n_inner)

            def apply_fn(df, prm=prm):
                return apply_spline_global(df, prm)

        for sn in ["train", "val", "test"]:
            sub = lng[lng["split"] == sn].copy()
            sub["p_hat"] = apply_fn(sub)
            s = M.summary_with_ci(M.score_from_long(sub, "p_hat"))
            r = raw_by_split[sn]
            rows.append({
                "params": n_params,
                "split": sn,
                "raw_RPS": round(r["RPS"], 5),
                "cal_RPS": round(s["RPS"], 5),
                "dRPS": round(r["RPS"] - s["RPS"], 5),
                "raw_BS": round(r["BS"], 5),
                "cal_BS": round(s["BS"], 5),
                "dBS": round(r["BS"] - s["BS"], 5),
                "raw_LogL": round(r["LogLoss"], 5),
                "cal_LogL": round(s["LogLoss"], 5),
                "dLogL": round(r["LogLoss"] - s["LogLoss"], 5),
                "n_events": int(s["n_events"]),
            })
            print(f"[{name:14s}] {sn:5s} "
                  f"dRPS={r['RPS'] - s['RPS']:+.5f} "
                  f"dBS={r['BS'] - s['BS']:+.5f} "
                  f"dLogL={r['LogLoss'] - s['LogLoss']:+.5f} "
                  f"(n={s['n_events']})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "spline_capacity_results.csv", index=False)

    wide = (res.pivot(index="params", columns="split",
                      values=["dRPS", "dBS", "dLogL"])
            .reorder_levels([1, 0], axis=1)
            .sort_index(axis=1, level=0))
    print("\n[wide delta table] (raw - recalibrated, positive = "
          "improvement)")
    print(wide.round(5).to_string(), flush=True)
    print("\n[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
