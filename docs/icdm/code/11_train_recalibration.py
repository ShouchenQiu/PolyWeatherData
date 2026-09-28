"""Step 11: train the logit recalibration function (roadmap Section 6).

Model (fixed form, per roadmap):  logit(p_hat) = alpha_g + beta_g * logit(p)
where g indexes horizon groups.

Horizon groups (candidate schemes):
  - global: one (alpha, beta) for all horizons
  - 3-group: g1 h>=24 (48,42,36,30,24), g2 6<=h<24 (18,12,9,6), g3 h<6 (3,1)

Protocol:
  1. Fit on TRAIN only (station start .. 2026-05-31, 10 active stations).
  2. Choose grouping on VALIDATION (2026-06-01 .. 06-30) - event-level RPS.
  3. Apply the chosen model ONCE to TEST (2026-07 full month).
  4. Compare raw vs recalibrated q on RPS / Brier / LogLoss, with
     event-level bootstrap 95% CI (roadmap 7.3).

Event-level scoring reuses the full-coverage + n_buckets>=7 guard from
step 10, computed directly from a LONG frame (one row per
condition_id x horizon) to avoid pivot/merge hazards.

Outputs:
  docs/icdm/output/recalibration_params.csv
  docs/icdm/output/recalibration_results.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
PANEL = OUT_DIR / "first_layer_panel_argmax_30min.parquet"
SPLIT = OUT_DIR / "train_val_test_split.parquet"
HORIZONS = [48.0, 42.0, 36.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 1.0]
HOR_COLS = [f"p_{int(h)}h" for h in HORIZONS]
N_BOOT = 2000
SEED = 20260822
EPS = 1e-6


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 60) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        z = x @ w
        mu = sigmoid(z)
        grad = x.T @ (mu - y)
        W = mu * (1.0 - mu)
        H = (x * W[:, None]).T @ x
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(x.shape[1]), grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
    return w


def group_by_h(h: float, scheme: str) -> str:
    """Horizon-based grouping (kept for comparison)."""
    if scheme == "global":
        return "g0"
    if scheme == "3group":
        if h >= 24.0:
            return "g1_h>=24"
        if h >= 6.0:
            return "g2_6-24"
        return "g3_h<6"
    # per-horizon grouping (every ~3h block gets its own alpha, beta)
    return f"h{int(h)}"


def group_by_price(p: float, scheme: str) -> str:
    """Price-based grouping: 3 segments on the market probability."""
    if scheme == "price3":
        if p <= 0.15:
            return "p1<=0.15"
        if p < 0.75:
            return "p2_0.15-0.75"
        return "p3>=0.75"
    raise ValueError(scheme)


def group_time_price(h: float, p: float) -> str:
    """Extreme time x extreme price: 4 cells.

    Near (h < 12) x long-shot (p <= 0.15) / favorite (p >= 0.75);
    Far (h > 30) x long-shot / favorite. Middle horizons and middle
    prices are dropped entirely (no group assigned).
    """
    if h >= 12.0 and h <= 30.0:
        return ""
    if 0.15 < p < 0.75:
        return ""
    if h < 12.0:
        t = "t1_near_h<12"
    else:
        t = "t3_far_h>30"
    if p <= 0.15:
        q = "p1<=0.15"
    else:
        q = "p3>=0.75"
    return f"{t}_{q}"


def train_scheme(df: pd.DataFrame) -> dict:
    params = {}
    for g, sub in df.groupby("group"):
        p = sub["p"].clip(EPS, 1 - EPS).to_numpy()
        y = sub["y"].to_numpy(dtype=float)
        x = np.column_stack([np.ones_like(p), logit(p)])
        w = fit_logistic(x, y)
        params[g] = {"alpha": float(w[0]), "beta": float(w[1]),
                     "n": int(len(sub))}
    return params


# --------------------------------------------------------------------------
# cubic B-spline calibration on logit(p)
# --------------------------------------------------------------------------
def _de_boor(x: float, knots: np.ndarray, k: int, i: int) -> float:
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    c1 = ((x - knots[i]) / (knots[i + k] - knots[i])
          if knots[i + k] > knots[i] else 0.0)
    c2 = ((knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1])
          if knots[i + k + 1] > knots[i + 1] else 0.0)
    return (c1 * _de_boor(x, knots, k - 1, i)
            + c2 * _de_boor(x, knots, k - 1, i + 1))


def bspline_design(z: np.ndarray, knots: np.ndarray,
                   degree: int = 3) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    lo, hi = knots[degree], knots[len(knots) - degree - 1]
    zc = np.clip(z, lo, hi - 1e-12)
    X = np.zeros((len(z), n_basis))
    for j in range(n_basis):
        X[:, j] = [_de_boor(x, knots, degree, j) for x in zc]
    return X


def make_knots(z: np.ndarray, n_inner: int = 3,
               degree: int = 3) -> np.ndarray:
    lo, hi = float(z.min()), float(z.max())
    pad = degree + 1
    if n_inner > 0:
        inner = np.quantile(
            z, np.linspace(0.05, 0.95, n_inner + 2)[1:-1])
    else:
        inner = np.array([])
    return np.concatenate([
        np.repeat(lo, pad), inner, np.repeat(hi, pad)]).astype(float)


def fit_spline_groups(sub: pd.DataFrame, knots: np.ndarray) -> dict:
    params = {}
    for g, gg in sub.groupby("group"):
        z = logit(gg["p"].clip(EPS, 1 - EPS).to_numpy())
        X = np.column_stack([np.ones(len(gg)), bspline_design(z, knots)])
        w = fit_logistic(X, gg["y"].to_numpy(dtype=float))
        params[g] = {"weights": w, "n": int(len(gg))}
    return params


def apply_spline_groups(sub: pd.DataFrame, params: dict,
                        knots: np.ndarray) -> pd.Series:
    out = np.full(len(sub), np.nan)
    pv = sub["p"].to_numpy()
    for g, prm in params.items():
        mask = (sub["group"] == g).to_numpy()
        z = logit(np.clip(pv[mask], EPS, 1 - EPS))
        X = np.column_stack([np.ones(len(z)), bspline_design(z, knots)])
        out[mask] = sigmoid(X @ prm["weights"])
    return pd.Series(out, index=sub.index)


def apply_params(df: pd.DataFrame, params: dict) -> pd.Series:
    p = df["p"].clip(EPS, 1 - EPS).to_numpy()
    out = np.full(len(df), np.nan)
    for g, prm in params.items():
        mask = (df["group"] == g).to_numpy()
        z = prm["alpha"] + prm["beta"] * logit(p[mask])
        out[mask] = sigmoid(z)
    # Direction A (tp4extreme): rows without a fitted group (middle
    # horizons / middle prices) keep the raw market price unchanged
    # (identity mapping). Every bucket then carries a price, so the
    # event-level full-coverage scoring stays intact.
    unfilled = np.isnan(out)
    out[unfilled] = p[unfilled]
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------
# long-frame building & event-level scoring
# --------------------------------------------------------------------------
def build_long(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition_id, horizon) with a non-null price."""
    rows = []
    for h, col in zip(HORIZONS, HOR_COLS):
        sub = panel[panel[col].notna()][
            ["condition_id", "event_id", "city", "bucket_lo_f", "split",
             col, "y"]].copy()
        sub["h"] = h
        sub = sub.rename(columns={col: "p"})
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def score_from_long(lng: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Event-level RPS/BS/LogLoss per (h, event) from a long frame.

    Full-coverage guard: an (event, h) is scored only if every bucket of
    the event has a price at that h, and the event has >= 7 buckets.
    """
    lng = lng[lng[price_col].notna()].copy()
    # bucket count per event (unique buckets across horizons)
    tot = (lng.groupby("event_id")["bucket_lo_f"]
           .nunique()
           .reset_index(name="n_total"))
    priced = lng.groupby(["event_id", "h"])[price_col].size().reset_index(
        name="n_priced")
    priced = priced.merge(tot, on="event_id", how="left")
    ok = priced[(priced["n_priced"] == priced["n_total"])
                & (priced["n_total"] >= 7)][["event_id", "h"]]
    ok["_keep"] = True
    lng = lng.merge(ok, on=["event_id", "h"], how="inner").drop(
        columns=["_keep"])

    rows = []
    for (ev, h), g in lng.groupby(["event_id", "h"]):
        g = g.sort_values("bucket_lo_f")
        p = g[price_col].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=int)
        S = p.sum()
        # Guard: the winning bucket must be among the priced buckets. The
        # coverage check above is computed on the priced long frame, so an
        # event whose winning bucket NEVER traded can pass it falsely; its
        # metrics would then be meaningless (LogLoss would pick an
        # arbitrary bucket via argmax). Drop such rows.
        if S <= 0 or int(y.sum()) != 1:
            continue
        q = p / S
        F = np.cumsum(q)
        O = np.cumsum(y)
        rows.append({
            "event_id": str(ev), "h": int(h),
            "BS": float(((q - y) ** 2).sum()),
            "LogLoss": float(-np.log(max(q[np.argmax(y)], 1e-12))),
            "RPS": float(((F - O) ** 2).sum()),
        })
    return pd.DataFrame(rows)


def summary_with_ci(scores: pd.DataFrame) -> dict:
    n = len(scores)
    rng = np.random.default_rng(SEED)
    out = {"n_events": n}
    for m in ["BS", "LogLoss", "RPS"]:
        vals = scores[m].to_numpy()
        out[m] = float(vals.mean())
        if n >= 10:
            draws = vals[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out[f"{m}_lo"], out[f"{m}_hi"] = float(lo), float(hi)
        else:
            out[f"{m}_lo"], out[f"{m}_hi"] = np.nan, np.nan
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    panel = pd.read_parquet(PANEL)
    split = pd.read_parquet(SPLIT)
    panel = panel.merge(split[["event_id", "split"]], on="event_id",
                        how="left")
    panel = panel[panel["split"].notna()].copy()

    lng = build_long(panel)
    print(f"[long] rows={len(lng)} "
          f"(train={int((lng['split']=='train').sum())}, "
          f"val={int((lng['split']=='val').sum())}, "
          f"test={int((lng['split']=='test').sum())})", flush=True)

    # ---- fit schemes on train ----
    # Each scheme maps a row to a group. Horizon schemes use h; price3 uses
    # the market probability p; tp9 crosses time groups x price segments.
    schemes = {
        "global": ("h", lambda df: df["h"].apply(
            lambda h: group_by_h(h, "global"))),
        "3group": ("h", lambda df: df["h"].apply(
            lambda h: group_by_h(h, "3group"))),
        "per3h": ("h", lambda df: df["h"].apply(
            lambda h: group_by_h(h, "per3h"))),
        "price3": ("p", lambda df: df["p"].apply(
            lambda p: group_by_price(p, "price3"))),
        "tp4extreme": ("hp", lambda df: [
            group_time_price(h, p)
            for h, p in zip(df["h"], df["p"])
        ]),
    }
    params_by_scheme = {}
    for name, (basis, group_fn) in schemes.items():
        tr = lng[lng["split"] == "train"].copy()
        tr["group"] = group_fn(tr)
        if name == "tp4extreme":
            tr = tr[tr["group"] != ""]
        params_by_scheme[name] = train_scheme(tr)
        print(f"[fit] {name}: " + " | ".join(
            f"{g}: a={v['alpha']:+.4f} b={v['beta']:+.4f} (n={v['n']})"
            for g, v in params_by_scheme[name].items()), flush=True)

    # ---- validation: pick scheme by event-level RPS ----
    val_raw = score_from_long(lng[lng["split"] == "val"], "p")
    s_rv = summary_with_ci(val_raw)
    print(f"[val raw] RPS={s_rv['RPS']:.4f} BS={s_rv['BS']:.4f} "
          f"LogL={s_rv['LogLoss']:.3f} (n={s_rv['n_events']})", flush=True)

    val_choice = {}
    for name, (basis, group_fn) in schemes.items():
        va = lng[lng["split"] == "val"].copy()
        va["group"] = group_fn(va)
        va["p_hat"] = apply_params(va, params_by_scheme[name])
        sc = score_from_long(va, "p_hat")
        s = summary_with_ci(sc)
        val_choice[name] = s
        print(f"[val {name}] RPS={s['RPS']:.4f} BS={s['BS']:.4f} "
              f"LogL={s['LogLoss']:.3f} (n={s['n_events']})", flush=True)

    # ---- cubic B-spline calibration on logit(p) ----
    # Knots shared across groups, fixed on TRAIN quantiles of logit(p).
    SPLINE_DEGREE = 3
    SPLINE_INNER = 3
    tr_z = logit(lng.loc[lng["split"] == "train", "p"]
                 .clip(EPS, 1 - EPS).to_numpy())
    spline_knots = make_knots(tr_z, SPLINE_INNER, SPLINE_DEGREE)
    spline_params_by_name: dict = {}
    spline_schemes = {
        "spline_global": lambda df: pd.Series("g0", index=df.index),
        "spline3": lambda df: df["h"].apply(
            lambda h: group_by_h(h, "3group")),
    }
    for sname, sgroup in spline_schemes.items():
        tr = lng[lng["split"] == "train"].copy()
        tr["group"] = sgroup(tr)
        sp = fit_spline_groups(tr, spline_knots)
        spline_params_by_name[sname] = (sgroup, sp)
        print(f"[fit] {sname}: " + " | ".join(
            f"{g}: w={np.round(v['weights'], 4).tolist()} (n={v['n']})"
            for g, v in sp.items()), flush=True)

        va = lng[lng["split"] == "val"].copy()
        va["group"] = sgroup(va)
        va["p_hat"] = apply_spline_groups(va, sp, spline_knots)
        s = summary_with_ci(score_from_long(va, "p_hat"))
        val_choice[sname] = s
        print(f"[val {sname}] RPS={s['RPS']:.4f} BS={s['BS']:.4f} "
              f"LogL={s['LogLoss']:.3f} (n={s['n_events']})", flush=True)

    best = min(val_choice, key=lambda k: val_choice[k]["RPS"])
    print(f"\n[choice] best scheme on validation: {best}", flush=True)

    # ---- score raw vs recalibrated on all splits ----
    results = []
    for split_name in ["train", "val", "test"]:
        sub = lng[lng["split"] == split_name]
        s_raw = summary_with_ci(score_from_long(sub, "p"))
        results.append({"split": split_name, "version": "raw",
                        "scheme": "-", **s_raw})
        print(f"[score] {split_name} raw: RPS={s_raw['RPS']:.4f} "
              f"BS={s_raw['BS']:.4f} LogL={s_raw['LogLoss']:.3f} "
              f"(n={s_raw['n_events']})", flush=True)

        rc = sub.copy()
        if best in spline_params_by_name:
            sgroup, sp = spline_params_by_name[best]
            rc["group"] = sgroup(rc)
            rc["p_hat"] = apply_spline_groups(rc, sp, spline_knots)
        else:
            rc["group"] = schemes[best][1](rc)
            rc["p_hat"] = apply_params(rc, params_by_scheme[best])
        s_rc = summary_with_ci(score_from_long(rc, "p_hat"))
        results.append({"split": split_name, "version": "recalibrated",
                        "scheme": best, **s_rc})
        print(f"[score] {split_name} recal: RPS={s_rc['RPS']:.4f} "
              f"BS={s_rc['BS']:.4f} LogL={s_rc['LogLoss']:.3f} "
              f"(n={s_rc['n_events']})", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "recalibration_results.csv", index=False)

    if best in spline_params_by_name:
        sgroup, sp = spline_params_by_name[best]
        param_rows = []
        for g, v in sp.items():
            w = v["weights"]
            row = {"scheme": best, "group": g, "n": v["n"]}
            for j in range(len(w)):
                row[f"w{j}"] = float(w[j])
            param_rows.append(row)
        pd.DataFrame(param_rows).to_csv(
            OUT_DIR / "recalibration_params.csv", index=False)
    else:
        param_rows = [{"scheme": best, "group": g, **v}
                      for g, v in params_by_scheme[best].items()]
        pd.DataFrame(param_rows).to_csv(
            OUT_DIR / "recalibration_params.csv", index=False)

    print("\n[done]", flush=True)
    print(res.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
