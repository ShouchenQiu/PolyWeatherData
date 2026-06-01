"""BacktestAnalysis: statistics and breakdowns for BacktestEngine results.

Ported from Herbie-main/_sniper_analysis.py.

All monetary figures are in units of $1 per bet (i.e. ROI is a fraction
such that ROI=0.05 means 5 cents profit per dollar risked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
_LEAD_BINS          = [(0, 6), (6, 12), (12, 24), (24, 36), (36, 48)]
_LEAD_LABELS        = ["0-6h", "6-12h", "12-24h", "24-36h", "36-48h"]


def _lead_bin(lh: float) -> str:
    for lo, hi in _LEAD_BINS:
        if lo <= lh < hi:
            return f"{lo}-{hi}h"
    return ">48h"


def _pnl_summary(sub: pd.DataFrame, thr: float) -> dict | None:
    """Return a P&L metrics dict for signals with |edge| > thr.

    Returns None if no signals pass the threshold.
    """
    yes = sub[sub["edge"] >  thr]
    no  = sub[sub["edge"] < -thr]
    n_yes, n_no = len(yes), len(no)
    n_tot = n_yes + n_no
    if n_tot == 0:
        return None

    pnl_y_mid  = yes["pnl_if_yes"].sum()
    pnl_n_mid  = no["pnl_if_no"].sum()
    pnl_y_real = (yes["pnl_if_yes"] - yes["halfspread"]).sum()
    pnl_n_real = (no["pnl_if_no"]  - no["halfspread"]).sum()
    total_mid  = pnl_y_mid  + pnl_n_mid
    total_real = pnl_y_real + pnl_n_real

    wr_yes = float(yes["pnl_if_yes"].gt(0).mean()) if n_yes > 0 else float("nan")
    wr_no  = float(no["pnl_if_no"].gt(0).mean())   if n_no  > 0 else float("nan")

    return {
        "threshold":   thr,
        "N_yes":       n_yes,
        "N_no":        n_no,
        "N_total":     n_tot,
        "PnL_mid":     round(float(total_mid),  4),
        "PnL_real":    round(float(total_real), 4),
        "ROI_mid":     round(float(total_mid)  / n_tot, 5),
        "ROI_real":    round(float(total_real) / n_tot, 5),
        "WinRate_YES": round(wr_yes, 4) if not np.isnan(wr_yes) else float("nan"),
        "WinRate_NO":  round(wr_no,  4) if not np.isnan(wr_no)  else float("nan"),
    }


class BacktestAnalysis:
    """Compute performance metrics from a BacktestEngine result DataFrame.

    Parameters
    ----------
    results_df:
        DataFrame returned by ``BacktestEngine.run()``.  Must contain
        columns: edge, pnl_if_yes, pnl_if_no, halfspread,
        actual_outcome, mc_prob, clob_price, coverage_type,
        lead_hours_to_4pm, init_utc, init_local (UTC-parseable string).
    """

    def __init__(self, results_df: pd.DataFrame) -> None:
        self._df = results_df.copy()
        # Pre-add lead_bin and init_local_hour for repeated calls
        if "lead_bin" not in self._df.columns:
            self._df["lead_bin"] = self._df["lead_hours_to_4pm"].apply(
                _lead_bin
            )
        if "init_local_hour" not in self._df.columns and "init_local" in self._df.columns:
            self._df["init_local_hour"] = pd.to_datetime(
                self._df["init_local"]
            ).dt.hour

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self, threshold: float = 0.05) -> dict:
        """High-level performance metrics at the given edge threshold.

        Returns
        -------
        {
            N_yes, N_no, N_total,
            PnL_mid, PnL_real, ROI_mid, ROI_real,
            WinRate_YES, WinRate_NO,
            brier_market, brier_tft, bss,
            max_drawdown_mid,
        }
        """
        base = _pnl_summary(self._df, threshold) or {}

        # Brier scores (full-coverage rows only where outcome is known)
        valid = self._df[
            (self._df["coverage_type"] == "full")
            & self._df["actual_outcome"].notna()
        ]
        brier_mkt = brier_tft = bss = float("nan")
        if len(valid) > 0:
            # Use L2 VWAP when available, legacy mid otherwise
            if "legacy_mid_price" in valid.columns:
                mkt_price = valid["clob_price"].fillna(valid["legacy_mid_price"])
            else:
                mkt_price = valid["clob_price"]
            brier_mkt = float(
                ((mkt_price - valid["actual_outcome"]) ** 2).mean()
            )
            brier_tft = float(
                ((valid["mc_prob"] - valid["actual_outcome"]) ** 2).mean()
            )
            bss = (
                float(1.0 - brier_tft / brier_mkt)
                if brier_mkt > 0
                else float("nan")
            )

        # Max drawdown on cumulative mid PnL
        max_dd = self._max_drawdown(threshold)

        return {
            **base,
            "brier_market":    round(brier_mkt, 5) if not np.isnan(brier_mkt) else float("nan"),
            "brier_tft":       round(brier_tft, 5) if not np.isnan(brier_tft) else float("nan"),
            "bss":             round(bss, 5)        if not np.isnan(bss)       else float("nan"),
            "max_drawdown_mid": round(max_dd, 5),
        }

    def by_threshold(
        self, thresholds: list[float] = _DEFAULT_THRESHOLDS
    ) -> pd.DataFrame:
        """P&L metrics broken down by edge threshold.

        Returns a DataFrame with one row per threshold, columns:
        threshold, N_yes, N_no, N_total, PnL_mid, PnL_real,
        ROI_mid, ROI_real, WinRate_YES, WinRate_NO.
        """
        rows = []
        for thr in thresholds:
            r = _pnl_summary(self._df, thr)
            if r:
                rows.append(r)
        return pd.DataFrame(rows)

    def by_lead_time(
        self, thresholds: list[float] | None = None
    ) -> pd.DataFrame:
        """P&L metrics grouped by lead-hours-to-4pm bin.

        Returns a DataFrame with one row per (lead_bin, threshold), columns:
        lead_bin, threshold, N_total, PnL_mid, ROI_mid, ROI_real.
        """
        if thresholds is None:
            thresholds = [0.05, 0.10, 0.15, 0.20]
        rows = []
        for lb in _LEAD_LABELS:
            sub = self._df[self._df["lead_bin"] == lb]
            for thr in thresholds:
                r = _pnl_summary(sub, thr)
                if r and r["N_total"] > 0:
                    rows.append({"lead_bin": lb, **r})
        return pd.DataFrame(rows)

    def by_init_hour(
        self, thresholds: list[float] | None = None
    ) -> pd.DataFrame:
        """P&L metrics grouped by init EST hour.

        Returns a DataFrame with one row per (init_local_hour, threshold).
        """
        if thresholds is None:
            thresholds = [0.05, 0.10, 0.15, 0.20]
        if "init_local_hour" not in self._df.columns:
            return pd.DataFrame()
        rows = []
        for hr in sorted(self._df["init_local_hour"].unique()):
            sub = self._df[self._df["init_local_hour"] == hr]
            for thr in thresholds:
                r = _pnl_summary(sub, thr)
                if r and r["N_total"] > 0:
                    rows.append({"init_local_hour": hr, **r})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_drawdown(self, threshold: float) -> float:
        """Maximum peak-to-trough drawdown on cumulative mid PnL."""
        yes = self._df[self._df["edge"] >  threshold]["pnl_if_yes"]
        no  = self._df[self._df["edge"] < -threshold]["pnl_if_no"]
        combined = pd.concat([yes, no]).dropna()
        if combined.empty:
            return 0.0
        cum = combined.cumsum().values
        peak = np.maximum.accumulate(cum)
        dd   = peak - cum
        return float(dd.max())
