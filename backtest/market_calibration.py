"""Historical token-price calibration without executable-edge claims."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from core.pricing.fees import taker_fee_usdc
from training.historical_crossfit import (
    HistoricalCrossfitPlan,
    historical_six_fold_plan,
)


DEFAULT_PRICE_BIN_WIDTHS = (0.10, 0.05, 0.02, 0.01)
LEAD_BAND_LABELS = (
    "0-1h",
    "1-3h",
    "3-6h",
    "6-12h",
    "12-18h",
    "18-24h",
    "24-30h",
    "30-36h",
    "36-42h",
    "42-48h",
    "48-72h",
    "72h+",
)


@dataclass(frozen=True)
class MarketCalibrationAnalysis:
    """All decision artifacts produced by one development-only analysis."""

    frames: dict[str, pd.DataFrame]
    report: dict[str, object]
    question_ledger: dict[str, object]
    html_artifacts: dict[str, str]


def build_transaction_panel(
    *,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    trades: pd.DataFrame,
    actuals: pd.DataFrame,
    timezone: str,
    fold_plan: HistoricalCrossfitPlan | None = None,
) -> pd.DataFrame:
    """Attach point-in-time regime, token truth, and governed fold lineage."""
    required = {
        "events": {"event_id", "target_date"},
        "markets": {
            "event_id",
            "condition_id",
            "yes_token_id",
            "no_token_id",
            "bucket_lo_f",
            "bucket_hi_f",
            "resolved_yes",
            "closed_at",
        },
        "trades": {
            "condition_id",
            "token_id",
            "timestamp",
            "outcome",
            "side",
            "price",
            "size",
            "usdc",
        },
        "actuals": {"target_date", "actual_high_f"},
    }
    frames = {
        "events": events,
        "markets": markets,
        "trades": trades,
        "actuals": actuals,
    }
    for name, columns in required.items():
        missing = sorted(columns - set(frames[name].columns))
        if missing:
            raise ValueError(f"{name} missing columns: {', '.join(missing)}")

    event_frame = events[["event_id", "target_date"]].copy()
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    event_frame["target_date"] = event_frame["target_date"].astype(str)
    if event_frame["event_id"].duplicated().any():
        raise ValueError("events contain duplicate event_id")
    event_frame["settlement_boundary"] = event_frame["target_date"].map(
        lambda value: _settlement_boundary(value, timezone)
    )

    actual_frame = actuals[["target_date", "actual_high_f"]].copy()
    actual_frame["target_date"] = actual_frame["target_date"].astype(str)
    if actual_frame["target_date"].duplicated().any():
        raise ValueError("actuals contain duplicate target_date")

    market_frame = markets.copy()
    market_frame["event_id"] = market_frame["event_id"].astype(str)
    market_frame["condition_id"] = market_frame["condition_id"].astype(str)
    market_frame["yes_token_id"] = market_frame["yes_token_id"].astype(str)
    market_frame["no_token_id"] = market_frame["no_token_id"].astype(str)
    market_frame["closed_at"] = pd.to_datetime(
        market_frame["closed_at"], utc=True, errors="coerce"
    )
    if market_frame["condition_id"].duplicated().any():
        raise ValueError("markets contain duplicate condition_id")
    if market_frame["closed_at"].isna().any():
        raise ValueError("markets require closed_at for calibration eligibility")
    if market_frame["resolved_yes"].isna().any():
        raise ValueError("schema-v2 resolved_yes is required for formal calibration")

    market_frame = market_frame.merge(
        event_frame,
        on="event_id",
        how="inner",
        validate="many_to_one",
    ).merge(
        actual_frame,
        on="target_date",
        how="left",
        validate="many_to_one",
    )
    if market_frame["actual_high_f"].isna().any():
        raise ValueError("every CLOB event requires complete actual-high truth")
    market_frame["actual_yes"] = (
        (
            market_frame["actual_high_f"].astype(float)
            >= market_frame["bucket_lo_f"].astype(float)
        )
        & (
            market_frame["actual_high_f"].astype(float)
            < market_frame["bucket_hi_f"].astype(float)
        )
    )
    winner_counts = market_frame.groupby("event_id")["actual_yes"].sum()
    if not winner_counts.eq(1).all():
        bad = ", ".join(winner_counts[winner_counts != 1].index.astype(str))
        raise ValueError(
            f"actual high must select exactly one Bucket per Event: {bad}"
        )
    resolved = market_frame["resolved_yes"].astype(bool)
    if not resolved.equals(market_frame["actual_yes"].astype(bool)):
        raise ValueError("actual-high truth disagrees with resolved_yes")

    trade_frame = trades.copy()
    trade_frame["condition_id"] = trade_frame["condition_id"].astype(str)
    trade_frame["token_id"] = trade_frame["token_id"].astype(str)
    trade_frame["timestamp"] = pd.to_datetime(
        trade_frame["timestamp"], utc=True, errors="coerce"
    )
    if trade_frame["timestamp"].isna().any():
        raise ValueError("trades contain invalid timestamps")
    panel = trade_frame.merge(
        market_frame,
        on="condition_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_market"),
    )
    is_yes = panel["token_id"].eq(panel["yes_token_id"])
    is_no = panel["token_id"].eq(panel["no_token_id"])
    if not (is_yes | is_no).all():
        raise ValueError("trade token_id does not belong to its Market")
    expected_outcome = np.where(is_yes, "YES", "NO")
    if not (
        panel["outcome"].astype(str).str.upper().to_numpy()
        == expected_outcome
    ).all():
        raise ValueError("trade outcome label disagrees with token_id")
    panel["p_token"] = pd.to_numeric(panel["price"], errors="raise").astype(float)
    if not panel["p_token"].between(0.0, 1.0).all():
        raise ValueError("token prices must be between zero and one")
    panel_resolved = panel["resolved_yes"].astype(bool)
    panel["y_token"] = np.where(
        is_yes, panel_resolved, ~panel_resolved
    ).astype(float)
    panel["lead_hours"] = (
        panel["settlement_boundary"] - panel["timestamp"]
    ).dt.total_seconds() / 3600.0
    panel["time_regime"] = np.select(
        [
            panel["timestamp"] >= panel["closed_at"],
            panel["timestamp"] >= panel["settlement_boundary"],
        ],
        ["invalid_post_close", "resolution_latency"],
        default="forecast_relevant",
    )
    panel["fold_id"] = panel["target_date"].map(
        _fold_lookup(fold_plan or historical_six_fold_plan())
    )
    if panel["fold_id"].isna().any():
        raise ValueError("target_date is outside the governed historical fold plan")
    return panel.reset_index(drop=True)


def run_market_calibration_analysis(
    *,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    trades: pd.DataFrame,
    actuals: pd.DataFrame,
    quotes: pd.DataFrame,
    timezone: str,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
    min_event_days: int = 30,
    min_condition_cells: int = 200,
    progress_callback: Callable[[str, int], None] | None = None,
) -> MarketCalibrationAnalysis:
    """Produce the approved calibration study without consuming a holdout."""
    panel = build_transaction_panel(
        events=events,
        markets=markets,
        trades=trades,
        actuals=actuals,
        timezone=timezone,
    )
    if progress_callback:
        progress_callback("transaction_panel", len(panel))
    calibration_bins = summarize_calibration_bins(
        panel,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        min_event_days=min_event_days,
        min_condition_cells=min_condition_cells,
    )
    resolution_latency = summarize_calibration_bins(
        panel,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1,
        min_event_days=min_event_days,
        min_condition_cells=min_condition_cells,
        time_regime="resolution_latency",
    )
    if progress_callback:
        progress_callback("calibration_bins", len(calibration_bins))
    price_lead_cells = summarize_price_lead_cells(
        panel,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        min_event_days=min_event_days,
        min_condition_cells=min_condition_cells,
    )
    if progress_callback:
        progress_callback("price_lead_cells", len(price_lead_cells))
    fold_stability = summarize_fold_stability(
        panel,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    candidates = select_calibration_candidates(
        price_lead_cells, fold_stability
    )
    buy_returns = build_buy_settlement_returns(panel, candidates)
    if progress_callback:
        progress_callback("fold_candidates", len(candidates))
    multibucket = build_multibucket_decomposition(
        events=events,
        markets=markets,
        quotes=quotes,
        actuals=actuals,
        timezone=timezone,
    )
    if progress_callback:
        progress_callback("multibucket_decomposition", len(multibucket))
    ledger = build_question_ledger(
        calibration_bins=calibration_bins,
        price_lead_cells=price_lead_cells,
        candidates=candidates,
        buy_returns=buy_returns,
        resolution_latency=resolution_latency,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    regime_counts = {
        str(name): int(count)
        for name, count in panel["time_regime"].value_counts().items()
    }
    report: dict[str, object] = {
        "status": "complete",
        "station_timezone": timezone,
        "split_role": "oof_development",
        "holdout_consumed": False,
        "transaction_rows": int(len(panel)),
        "event_days": int(panel["event_id"].nunique()),
        "condition_count": int(panel["condition_id"].nunique()),
        "time_regime_counts": regime_counts,
        "candidate_count": int(len(candidates)),
        "buy_return_event_days": int(buy_returns["event_id"].nunique())
        if not buy_returns.empty
        else 0,
        "multibucket_vectors": int(len(multibucket)),
        "multibucket_summary": (
            {
                "mean_midpoint_sum": float(multibucket["midpoint_sum"].mean()),
                "mean_overround": float(multibucket["midpoint_overround"].mean()),
                "mean_multiclass_brier": float(
                    multibucket["multiclass_brier"].mean()
                ),
                "mean_log_score": float(multibucket["log_score"].mean()),
                "mean_normalized_rps": float(
                    multibucket["normalized_rps"].mean()
                ),
            }
            if not multibucket.empty
            else {}
        ),
        "terminology": {
            "transaction_calibration_gap": (
                "empirical token outcome rate minus mean transaction price "
                "within a declared price/time/weighting cell"
            ),
            "executable_edge": "not established by this study",
        },
        "limitations": [
            "transaction fills are not historical L2 executable VWAP",
            "six-fold stability is development evidence, not an independent holdout",
            "resolution-latency comparisons are descriptive, not causal",
        ],
    }
    frames = {
        "transaction_panel": panel,
        "calibration_bins": calibration_bins,
        "price_lead_cells": price_lead_cells,
        "fold_stability": fold_stability,
        "resolution_latency": resolution_latency,
        "calibration_candidates": candidates,
        "buy_settlement_returns": buy_returns,
        "multibucket_decomposition": multibucket,
    }
    return MarketCalibrationAnalysis(
        frames=frames,
        report=report,
        question_ledger=ledger,
        html_artifacts=_render_calibration_plots(frames),
    )


def summarize_calibration_bins(
    panel: pd.DataFrame,
    *,
    bin_widths: Iterable[float] = DEFAULT_PRICE_BIN_WIDTHS,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
    min_event_days: int = 30,
    min_condition_cells: int = 200,
    time_regime: str = "forecast_relevant",
) -> pd.DataFrame:
    """Summarize token calibration under fill, Event-equal, and USDC weights."""
    required = {
        "event_id",
        "condition_id",
        "p_token",
        "y_token",
        "usdc",
        "time_regime",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("transaction panel missing columns: " + ", ".join(missing))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    eligible = panel[panel["time_regime"].eq(time_regime)].copy()
    eligible["p_token"] = pd.to_numeric(eligible["p_token"], errors="raise")
    eligible["y_token"] = pd.to_numeric(eligible["y_token"], errors="raise")
    eligible["usdc"] = pd.to_numeric(eligible["usdc"], errors="raise")
    rows: list[dict[str, object]] = []
    for width_index, width in enumerate(bin_widths):
        width = float(width)
        if not 0.0 < width <= 1.0:
            raise ValueError("price bin widths must be in (0, 1]")
        binned = eligible.copy()
        binned["price_bin_start"] = _price_bin_start(binned["p_token"], width)
        for price_start, cell in binned.groupby("price_bin_start", sort=True):
            contract_cells = cell[["event_id", "condition_id"]].drop_duplicates()
            counts = {
                "fills": int(len(cell)),
                "condition_cells": int(len(contract_cells)),
                "event_days": int(cell["event_id"].nunique()),
                "usdc": float(cell["usdc"].sum()),
            }
            status = (
                "eligible"
                if counts["event_days"] >= min_event_days
                and counts["condition_cells"] >= min_condition_cells
                else "inconclusive"
            )
            for weighting in ("fill", "event_contract_equal", "usdc"):
                summary, contributions = _calibration_summary(cell, weighting)
                ci_low, ci_high = _cluster_gap_interval(
                    contributions,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + width_index * 10_000 + int(round(price_start * 10_000)),
                )
                rows.append(
                    {
                        "price_bin_width": width,
                        "price_bin_start": float(round(price_start, 12)),
                        "price_bin_end": float(
                            round(min(1.0, price_start + width), 12)
                        ),
                        "weighting": weighting,
                        "time_regime": time_regime,
                        **counts,
                        **summary,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "inference_status": status,
                    }
                )
    return pd.DataFrame(rows)


def assign_lead_band(lead_hours: pd.Series) -> pd.Series:
    """Assign the frozen forecast-relevant lead-time bands."""
    numeric = pd.to_numeric(lead_hours, errors="coerce")
    return pd.cut(
        numeric,
        bins=[0, 1, 3, 6, 12, 18, 24, 30, 36, 42, 48, 72, np.inf],
        labels=LEAD_BAND_LABELS,
        right=False,
    ).astype("string")


def summarize_price_lead_cells(
    panel: pd.DataFrame,
    *,
    price_bin_width: float = 0.01,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
    min_event_days: int = 30,
    min_condition_cells: int = 200,
) -> pd.DataFrame:
    """Summarize the pre-registered price-by-lead calibration family."""
    required = {
        "event_id",
        "condition_id",
        "p_token",
        "y_token",
        "usdc",
        "lead_hours",
        "time_regime",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("transaction panel missing columns: " + ", ".join(missing))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    width = float(price_bin_width)
    if not 0.0 < width <= 1.0:
        raise ValueError("price_bin_width must be in (0, 1]")
    eligible = panel[panel["time_regime"].eq("forecast_relevant")].copy()
    eligible["lead_band"] = assign_lead_band(eligible["lead_hours"])
    if eligible["lead_band"].isna().any():
        raise ValueError("forecast-relevant trades require non-negative lead_hours")
    eligible["price_bin_start"] = _price_bin_start(eligible["p_token"], width)
    cells = list(
        eligible.groupby(["price_bin_start", "lead_band"], sort=True, observed=True)
    )
    family_size = max(1, len(cells))
    rows: list[dict[str, object]] = []
    for cell_index, ((price_start, lead_band), cell) in enumerate(cells):
        condition_cells = cell[["event_id", "condition_id"]].drop_duplicates()
        counts = {
            "fills": int(len(cell)),
            "condition_cells": int(len(condition_cells)),
            "event_days": int(cell["event_id"].nunique()),
            "usdc": float(cell["usdc"].sum()),
        }
        status = (
            "eligible"
            if counts["event_days"] >= min_event_days
            and counts["condition_cells"] >= min_condition_cells
            else "inconclusive"
        )
        cell_seed = seed + cell_index * 1_003
        for weighting in ("fill", "event_contract_equal", "usdc"):
            summary, contributions = _calibration_summary(cell, weighting)
            ci_low, ci_high = _cluster_gap_interval(
                contributions,
                bootstrap_samples=bootstrap_samples,
                seed=cell_seed,
            )
            simultaneous_low = float("nan")
            simultaneous_high = float("nan")
            if weighting == "event_contract_equal":
                simultaneous_low, simultaneous_high = _cluster_gap_interval(
                    contributions,
                    bootstrap_samples=bootstrap_samples,
                    seed=cell_seed,
                    alpha=0.05 / family_size,
                )
            rows.append(
                {
                    "price_bin_width": width,
                    "price_bin_start": float(round(price_start, 12)),
                    "price_bin_end": float(
                        round(min(1.0, price_start + width), 12)
                    ),
                    "lead_band": str(lead_band),
                    "weighting": weighting,
                    **counts,
                    **summary,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "simultaneous_ci_low": simultaneous_low,
                    "simultaneous_ci_high": simultaneous_high,
                    "multiplicity_method": (
                        "bonferroni_event_cluster_bootstrap"
                        if weighting == "event_contract_equal"
                        else "not_applicable"
                    ),
                    "inference_status": status,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_lead_order"] = result["lead_band"].map(
        {label: index for index, label in enumerate(LEAD_BAND_LABELS)}
    )
    return result.sort_values(
        ["price_bin_start", "_lead_order", "weighting"], kind="stable"
    ).drop(columns="_lead_order").reset_index(drop=True)


def summarize_fold_stability(
    panel: pd.DataFrame,
    *,
    price_bin_width: float = 0.01,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate Event-equal calibration gaps in each governed historical fold."""
    required = {
        "event_id",
        "condition_id",
        "p_token",
        "y_token",
        "usdc",
        "lead_hours",
        "time_regime",
        "fold_id",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("transaction panel missing columns: " + ", ".join(missing))
    eligible = panel[panel["time_regime"].eq("forecast_relevant")].copy()
    eligible["lead_band"] = assign_lead_band(eligible["lead_hours"])
    eligible["price_bin_start"] = _price_bin_start(
        eligible["p_token"], float(price_bin_width)
    )
    rows: list[dict[str, object]] = []
    grouped = eligible.groupby(
        ["fold_id", "price_bin_start", "lead_band"],
        sort=True,
        observed=True,
    )
    for index, ((fold_id, price_start, lead_band), cell) in enumerate(grouped):
        summary, contributions = _calibration_summary(
            cell, "event_contract_equal"
        )
        ci_low, ci_high = _cluster_gap_interval(
            contributions,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 1_003,
        )
        rows.append(
            {
                "fold_id": str(fold_id),
                "price_bin_width": float(price_bin_width),
                "price_bin_start": float(round(price_start, 12)),
                "price_bin_end": float(
                    round(min(1.0, price_start + price_bin_width), 12)
                ),
                "lead_band": str(lead_band),
                "fills": int(len(cell)),
                "condition_cells": int(
                    len(cell[["event_id", "condition_id"]].drop_duplicates())
                ),
                "event_days": int(cell["event_id"].nunique()),
                **summary,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def select_calibration_candidates(
    price_lead_cells: pd.DataFrame,
    fold_stability: pd.DataFrame,
    *,
    min_consistent_folds: int = 4,
) -> pd.DataFrame:
    """Select development-only calibration anomalies under frozen rules."""
    candidate_columns = [
        *price_lead_cells.columns.tolist(),
        "fill_calibration_gap",
        "candidate_direction",
        "observed_fold_count",
        "consistent_fold_count",
        "significant_opposite_fold_count",
        "evidence_role",
    ]
    candidate_columns = list(dict.fromkeys(candidate_columns))
    keys = [
        "price_bin_width",
        "price_bin_start",
        "price_bin_end",
        "lead_band",
    ]
    primary = price_lead_cells[
        price_lead_cells["weighting"].eq("event_contract_equal")
        & price_lead_cells["inference_status"].eq("eligible")
    ].copy()
    if primary.empty:
        return pd.DataFrame(columns=candidate_columns)
    significant = (primary["simultaneous_ci_low"] > 0.0) | (
        primary["simultaneous_ci_high"] < 0.0
    )
    primary = primary[significant].copy()
    fill = price_lead_cells[price_lead_cells["weighting"].eq("fill")][
        [*keys, "calibration_gap"]
    ].rename(columns={"calibration_gap": "fill_calibration_gap"})
    primary = primary.merge(fill, on=keys, how="inner", validate="one_to_one")
    primary = primary[
        np.sign(primary["calibration_gap"])
        == np.sign(primary["fill_calibration_gap"])
    ].copy()
    rows: list[dict[str, object]] = []
    for row in primary.itertuples(index=False):
        mask = pd.Series(True, index=fold_stability.index)
        for key in keys:
            mask &= fold_stability[key].eq(getattr(row, key))
        folds = fold_stability[mask].copy()
        direction = 1 if float(row.calibration_gap) > 0.0 else -1
        consistent = np.sign(folds["calibration_gap"]) == direction
        opposite_significant = (
            (folds["ci_high"] < 0.0)
            if direction > 0
            else (folds["ci_low"] > 0.0)
        )
        observed_fold_count = int(folds["fold_id"].nunique())
        consistent_fold_count = int(consistent.sum())
        if (
            observed_fold_count < min_consistent_folds
            or consistent_fold_count < min_consistent_folds
            or bool(opposite_significant.any())
        ):
            continue
        payload = row._asdict()
        payload.update(
            {
                "candidate_direction": (
                    "underpriced_token" if direction > 0 else "overpriced_token"
                ),
                "observed_fold_count": observed_fold_count,
                "consistent_fold_count": consistent_fold_count,
                "significant_opposite_fold_count": int(
                    opposite_significant.sum()
                ),
                "evidence_role": "oof_development_candidate",
            }
        )
        rows.append(payload)
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=candidate_columns)
    )


def build_buy_settlement_returns(
    panel: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observed BUY fills for underpriced-token candidates only."""
    output_columns = [
        "event_id",
        "target_date",
        "price_bin_width",
        "price_bin_start",
        "price_bin_end",
        "lead_band",
        "fill_count",
        "condition_count",
        "shares",
        "observed_usdc",
        "vwap_price",
        "payout_usdc",
        "gross_settlement_return_usdc",
        "fee_usdc",
        "net_settlement_return_usdc",
        "net_settlement_return_per_share",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)
    required_panel = {
        "event_id",
        "target_date",
        "condition_id",
        "side",
        "p_token",
        "y_token",
        "size",
        "usdc",
        "lead_hours",
        "time_regime",
        "fees_enabled",
        "fee_rate",
        "fee_exponent",
    }
    missing = sorted(required_panel - set(panel.columns))
    if missing:
        raise ValueError("transaction panel missing columns: " + ", ".join(missing))
    candidate_frame = candidates[
        candidates["candidate_direction"].eq("underpriced_token")
    ].copy()
    if candidate_frame.empty:
        return pd.DataFrame(columns=output_columns)
    widths = candidate_frame["price_bin_width"].astype(float).unique()
    if len(widths) != 1:
        raise ValueError("BUY settlement proxy requires one frozen price-bin width")
    width = float(widths[0])
    work = panel[
        panel["time_regime"].eq("forecast_relevant")
        & panel["side"].astype(str).str.upper().eq("BUY")
    ].copy()
    work["price_bin_width"] = width
    work["price_bin_start"] = _price_bin_start(work["p_token"], width)
    work["price_bin_end"] = (work["price_bin_start"] + width).clip(upper=1.0)
    work["lead_band"] = assign_lead_band(work["lead_hours"])
    keys = [
        "price_bin_width",
        "price_bin_start",
        "price_bin_end",
        "lead_band",
    ]
    work = work.merge(
        candidate_frame[keys].drop_duplicates(),
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if work.empty:
        return pd.DataFrame(columns=output_columns)
    work["fee_usdc"] = [
        taker_fee_usdc(
            shares=float(row.size),
            price=float(row.p_token),
            fees_enabled=bool(row.fees_enabled),
            fee_rate=(None if pd.isna(row.fee_rate) else float(row.fee_rate)),
            fee_exponent=(
                None if pd.isna(row.fee_exponent) else float(row.fee_exponent)
            ),
        )
        for row in work.itertuples(index=False)
    ]
    work["payout_usdc"] = work["y_token"].astype(float) * work["size"].astype(float)
    work["gross_settlement_return_usdc"] = (
        work["payout_usdc"] - work["usdc"].astype(float)
    )
    work["net_settlement_return_usdc"] = (
        work["gross_settlement_return_usdc"] - work["fee_usdc"]
    )
    grouped = work.groupby(
        ["event_id", "target_date", *keys], sort=True, as_index=False
    ).agg(
        fill_count=("condition_id", "size"),
        condition_count=("condition_id", "nunique"),
        shares=("size", "sum"),
        observed_usdc=("usdc", "sum"),
        payout_usdc=("payout_usdc", "sum"),
        gross_settlement_return_usdc=("gross_settlement_return_usdc", "sum"),
        fee_usdc=("fee_usdc", "sum"),
        net_settlement_return_usdc=("net_settlement_return_usdc", "sum"),
    )
    grouped["vwap_price"] = grouped["observed_usdc"] / grouped["shares"]
    grouped["net_settlement_return_per_share"] = (
        grouped["net_settlement_return_usdc"] / grouped["shares"]
    )
    return grouped[output_columns]


def build_multibucket_decomposition(
    *,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    quotes: pd.DataFrame,
    actuals: pd.DataFrame,
    timezone: str,
) -> pd.DataFrame:
    """Score complete past-trade midpoint vectors before settlement."""
    output_columns = [
        "event_id",
        "target_date",
        "timestamp",
        "lead_hours",
        "lead_band",
        "bucket_count",
        "midpoint_sum",
        "midpoint_overround",
        "multiclass_brier",
        "log_score",
        "normalized_rps",
        "pooled_binary_brier",
        "true_bucket_probability",
    ]
    event_frame = events[["event_id", "target_date"]].copy()
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    event_frame["target_date"] = event_frame["target_date"].astype(str)
    event_frame["settlement_boundary"] = event_frame["target_date"].map(
        lambda value: _settlement_boundary(value, timezone)
    )
    market_frame = markets[
        [
            "event_id",
            "condition_id",
            "bucket_lo_f",
            "bucket_hi_f",
            "closed_at",
        ]
    ].copy()
    market_frame["event_id"] = market_frame["event_id"].astype(str)
    market_frame["condition_id"] = market_frame["condition_id"].astype(str)
    market_frame["closed_at"] = pd.to_datetime(
        market_frame["closed_at"], utc=True, errors="coerce"
    )
    if market_frame["closed_at"].isna().any():
        raise ValueError("multibucket decomposition requires closed_at")
    market_frame = market_frame.merge(
        event_frame,
        on="event_id",
        how="inner",
        validate="many_to_one",
    )
    actual_frame = actuals[["target_date", "actual_high_f"]].copy()
    actual_frame["target_date"] = actual_frame["target_date"].astype(str)
    market_frame = market_frame.merge(
        actual_frame,
        on="target_date",
        how="left",
        validate="many_to_one",
    )
    if market_frame["actual_high_f"].isna().any():
        raise ValueError("multibucket decomposition requires actual-high truth")
    market_frame["yes_outcome"] = (
        (market_frame["actual_high_f"] >= market_frame["bucket_lo_f"])
        & (market_frame["actual_high_f"] < market_frame["bucket_hi_f"])
    ).astype(float)
    winners = market_frame.groupby("event_id")["yes_outcome"].sum()
    if not winners.eq(1.0).all():
        raise ValueError("each multibucket Event must have exactly one winner")

    quote_frame = quotes[["condition_id", "timestamp", "mid_price_yes"]].copy()
    quote_frame["condition_id"] = quote_frame["condition_id"].astype(str)
    quote_frame["timestamp"] = pd.to_datetime(
        quote_frame["timestamp"], utc=True, errors="coerce"
    )
    quote_frame["mid_price_yes"] = pd.to_numeric(
        quote_frame["mid_price_yes"], errors="raise"
    )
    quote_frame = quote_frame.merge(
        market_frame,
        on="condition_id",
        how="inner",
        validate="many_to_one",
    )
    quote_frame = quote_frame[
        (quote_frame["timestamp"] < quote_frame["closed_at"])
        & (quote_frame["timestamp"] < quote_frame["settlement_boundary"])
    ].copy()
    expected = market_frame.groupby("event_id")["condition_id"].nunique()
    observed = quote_frame.groupby(["event_id", "timestamp"])[
        "condition_id"
    ].nunique()
    complete = observed[
        observed
        == observed.index.get_level_values("event_id").map(expected)
    ].index
    if len(complete) == 0:
        return pd.DataFrame(columns=output_columns)
    quote_frame = quote_frame.set_index(["event_id", "timestamp"]).loc[
        complete
    ].reset_index()
    rows: list[dict[str, object]] = []
    for (event_id, timestamp), vector in quote_frame.groupby(
        ["event_id", "timestamp"], sort=True
    ):
        vector = vector.sort_values(
            ["bucket_lo_f", "bucket_hi_f", "condition_id"], kind="stable"
        )
        raw = vector["mid_price_yes"].to_numpy(dtype=float)
        raw_sum = float(raw.sum())
        if (
            raw_sum <= 0.0
            or not np.isfinite(raw).all()
            or (raw < 0.0).any()
            or (raw > 1.0).any()
        ):
            raise ValueError("synthetic midpoint vector must be finite and positive")
        probability = raw / raw_sum
        outcome = vector["yes_outcome"].to_numpy(dtype=float)
        truth_index = int(np.argmax(outcome))
        bucket_count = len(vector)
        brier = float(np.square(probability - outcome).sum())
        rps = (
            float(
                np.square(
                    np.cumsum(probability)[:-1] - np.cumsum(outcome)[:-1]
                ).sum()
                / (bucket_count - 1)
            )
            if bucket_count > 1
            else 0.0
        )
        boundary = vector["settlement_boundary"].iloc[0]
        lead_hours = float(
            (pd.Timestamp(boundary) - pd.Timestamp(timestamp)).total_seconds()
            / 3600.0
        )
        rows.append(
            {
                "event_id": str(event_id),
                "target_date": str(vector["target_date"].iloc[0]),
                "timestamp": pd.Timestamp(timestamp),
                "lead_hours": lead_hours,
                "lead_band": str(assign_lead_band(pd.Series([lead_hours])).iloc[0]),
                "bucket_count": bucket_count,
                "midpoint_sum": raw_sum,
                "midpoint_overround": raw_sum - 1.0,
                "multiclass_brier": brier,
                "log_score": float(-np.log(max(probability[truth_index], 1e-12))),
                "normalized_rps": rps,
                "pooled_binary_brier": brier / bucket_count,
                "true_bucket_probability": float(probability[truth_index]),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def write_market_calibration_artifact(
    *,
    output: Path | str,
    frames: dict[str, pd.DataFrame],
    report: dict[str, object],
    question_ledger: dict[str, object],
    source_sha256: dict[str, str],
    protocol: dict[str, object],
    html_artifacts: dict[str, str] | None = None,
) -> dict[str, object]:
    """Write one immutable, hash-bound market-calibration artifact."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for name, frame in frames.items():
        filename = f"{name}.parquet"
        path = root / filename
        frame.to_parquet(path, index=False)
        row_counts[name] = int(len(frame))
        hashes[filename] = _sha256(path)
    for filename, payload in (
        ("report.json", report),
        ("question_ledger.json", question_ledger),
    ):
        path = root / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes[filename] = _sha256(path)
    for filename, html in (html_artifacts or {}).items():
        if Path(filename).name != filename or not filename.endswith(".html"):
            raise ValueError("HTML artifact names must be plain .html filenames")
        path = root / filename
        path.write_text(html, encoding="utf-8")
        hashes[filename] = _sha256(path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_class": "historical_transaction_calibration",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
        "row_counts": row_counts,
        "source_sha256": dict(sorted(source_sha256.items())),
        "protocol": protocol,
        "sha256": dict(sorted(hashes.items())),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_question_ledger(
    *,
    calibration_bins: pd.DataFrame,
    price_lead_cells: pd.DataFrame,
    candidates: pd.DataFrame,
    buy_returns: pd.DataFrame,
    resolution_latency: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Answer every approved decision question with governed evidence states."""
    paper = calibration_bins[
        calibration_bins["price_bin_width"].eq(0.1)
        & calibration_bins["weighting"].eq("fill")
        & calibration_bins["time_regime"].eq("forecast_relevant")
        & calibration_bins["inference_status"].eq("eligible")
    ]
    paper_significant = paper[
        (paper["ci_low"] > 0.0) | (paper["ci_high"] < 0.0)
    ]
    paper_status = (
        "supported"
        if len(paper_significant)
        else ("contradicted" if len(paper) else "inconclusive")
    )
    eligible_cells = price_lead_cells[
        price_lead_cells["weighting"].eq("event_contract_equal")
        & price_lead_cells["inference_status"].eq("eligible")
    ]
    fine_status = (
        "supported"
        if len(candidates)
        else ("contradicted" if len(eligible_cells) else "inconclusive")
    )
    return_status = "inconclusive"
    return_evidence: dict[str, object] = {
        "event_days": int(buy_returns["event_id"].nunique())
        if not buy_returns.empty
        else 0
    }
    if not buy_returns.empty:
        event_values = (
            buy_returns.groupby("event_id")["net_settlement_return_per_share"]
            .mean()
            .to_numpy(dtype=float)
        )
        rng = np.random.default_rng(seed)
        draws = event_values[
            rng.integers(
                0,
                len(event_values),
                size=(bootstrap_samples, len(event_values)),
            )
        ].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        estimate = float(event_values.mean())
        return_status = (
            "supported" if low > 0.0 else ("contradicted" if high < 0.0 else "inconclusive")
        )
        return_evidence.update(
            {
                "mean_net_return_per_share": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return {
        "paper_style_weather_miscalibration": {
            "status": paper_status,
            "evidence": {
                "artifact": "calibration_bins.parquet",
                "eligible_10pct_bins": int(len(paper)),
                "significant_10pct_bins": int(len(paper_significant)),
            },
        },
        "fine_price_time_miscalibration": {
            "status": fine_status,
            "evidence": {
                "artifact": "price_lead_cells.parquet",
                "eligible_cells": int(len(eligible_cells)),
                "candidate_cells": int(len(candidates)),
            },
        },
        "weighting_and_resolution_latency_mechanism": {
            "status": "inconclusive" if len(resolution_latency) else "not_measured",
            "evidence": {
                "artifacts": [
                    "calibration_bins.parquet",
                    "resolution_latency.parquet",
                ],
                "latency_rows": int(len(resolution_latency)),
                "limitation": "descriptive contrasts do not establish a causal mechanism",
            },
        },
        "cross_fold_stability": {
            "status": (
                "supported"
                if len(candidates)
                else ("contradicted" if len(eligible_cells) else "inconclusive")
            ),
            "evidence": {
                "artifacts": [
                    "fold_stability.parquet",
                    "calibration_candidates.parquet",
                ],
                "candidate_cells": int(len(candidates)),
                "evidence_role": "oof_development_only",
            },
        },
        "observed_buy_settlement_return": {
            "status": return_status,
            "evidence": {
                "artifact": "buy_settlement_returns.parquet",
                **return_evidence,
            },
        },
        "executable_edge": {
            "status": "not_measured",
            "evidence": {
                "missing_data": "historical L2 ask/bid depth at decision time",
                "reason": (
                    "transaction fills and synthetic midpoints cannot establish "
                    "fee-adjusted executable VWAP, capacity, or queue availability"
                ),
            },
        },
    }


def _render_calibration_plots(
    frames: dict[str, pd.DataFrame],
) -> dict[str, str]:
    from plotly import graph_objects as go

    bins = frames["calibration_bins"]
    curve = bins[
        bins["price_bin_width"].eq(0.1)
        & bins["weighting"].isin(["fill", "event_contract_equal"])
        & bins["time_regime"].eq("forecast_relevant")
    ]
    curve_figure = go.Figure()
    curve_figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="perfect calibration",
            line={"dash": "dash", "color": "gray"},
        )
    )
    for weighting, rows in curve.groupby("weighting", sort=True):
        curve_figure.add_trace(
            go.Scatter(
                x=rows["mean_price"],
                y=rows["empirical_rate"],
                mode="lines+markers",
                name=str(weighting),
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (rows["ci_high"] - rows["calibration_gap"]).clip(lower=0),
                    "arrayminus": (rows["calibration_gap"] - rows["ci_low"]).clip(lower=0),
                    "visible": True,
                },
            )
        )
    curve_figure.update_layout(
        title="KLGA transaction calibration (10% bins)",
        xaxis_title="Mean token transaction price",
        yaxis_title="Empirical token outcome rate",
    )

    cells = frames["price_lead_cells"]
    primary = cells[cells["weighting"].eq("event_contract_equal")]
    heatmap_figure = go.Figure()
    if not primary.empty:
        pivot = primary.pivot(
            index="lead_band",
            columns="price_bin_start",
            values="calibration_gap",
        ).reindex([label for label in LEAD_BAND_LABELS if label in set(primary["lead_band"])])
        heatmap_figure.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=pivot.columns.astype(float),
                y=pivot.index.astype(str),
                colorscale="RdBu",
                zmid=0.0,
                colorbar={"title": "gap"},
            )
        )
    heatmap_figure.update_layout(
        title="Event-equal transaction calibration gap by price and lead",
        xaxis_title="Token price bin start",
        yaxis_title="Lead band",
    )

    stability = frames["fold_stability"]
    stability_figure = go.Figure()
    if not stability.empty:
        stability_figure.add_trace(
            go.Scatter(
                x=stability["fold_id"],
                y=stability["calibration_gap"],
                mode="markers",
                marker={
                    "color": stability["price_bin_start"],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "price"},
                },
                text=stability["lead_band"],
                name="fold cell gap",
            )
        )
    stability_figure.add_hline(y=0.0, line_dash="dash", line_color="gray")
    stability_figure.update_layout(
        title="Historical fold stability",
        xaxis_title="Governed fold",
        yaxis_title="Event-equal calibration gap",
    )
    return {
        "calibration_curve.html": curve_figure.to_html(
            full_html=True, include_plotlyjs="cdn"
        ),
        "price_lead_heatmap.html": heatmap_figure.to_html(
            full_html=True, include_plotlyjs="cdn"
        ),
        "fold_stability.html": stability_figure.to_html(
            full_html=True, include_plotlyjs="cdn"
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration_summary(
    cell: pd.DataFrame,
    weighting: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    work = cell.copy()
    eps = 1e-12
    work["binary_brier"] = (work["p_token"] - work["y_token"]) ** 2
    clipped = work["p_token"].clip(eps, 1.0 - eps)
    work["binary_log_loss"] = -(
        work["y_token"] * np.log(clipped)
        + (1.0 - work["y_token"]) * np.log(1.0 - clipped)
    )
    if weighting == "event_contract_equal":
        contracts = work.groupby(
            ["event_id", "condition_id"], sort=False, as_index=False
        ).agg(
            p=("p_token", "mean"),
            y=("y_token", "first"),
            brier=("binary_brier", "mean"),
            log_loss=("binary_log_loss", "mean"),
        )
        contributions = contracts.groupby("event_id", sort=False, as_index=False).agg(
            p_num=("p", "mean"),
            y_num=("y", "mean"),
            brier_num=("brier", "mean"),
            log_num=("log_loss", "mean"),
        )
        contributions["denom"] = 1.0
    else:
        weight = (
            pd.Series(1.0, index=work.index)
            if weighting == "fill"
            else work["usdc"].astype(float)
        )
        if weighting == "usdc" and float(weight.sum()) <= 0.0:
            raise ValueError("USDC-weighted calibration requires positive notional")
        work["denom"] = weight
        work["p_num"] = work["p_token"] * weight
        work["y_num"] = work["y_token"] * weight
        work["brier_num"] = work["binary_brier"] * weight
        work["log_num"] = work["binary_log_loss"] * weight
        contributions = work.groupby("event_id", sort=False, as_index=False).agg(
            p_num=("p_num", "sum"),
            y_num=("y_num", "sum"),
            brier_num=("brier_num", "sum"),
            log_num=("log_num", "sum"),
            denom=("denom", "sum"),
        )
    denom = float(contributions["denom"].sum())
    mean_price = float(contributions["p_num"].sum() / denom)
    empirical_rate = float(contributions["y_num"].sum() / denom)
    return (
        {
            "mean_price": mean_price,
            "empirical_rate": empirical_rate,
            "calibration_gap": empirical_rate - mean_price,
            "binary_brier": float(contributions["brier_num"].sum() / denom),
            "binary_log_loss": float(contributions["log_num"].sum() / denom),
        },
        contributions,
    )


def _cluster_gap_interval(
    contributions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = contributions[["p_num", "y_num", "denom"]].to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(values),
        np.full(len(values), 1.0 / len(values)),
        size=bootstrap_samples,
    )
    totals = counts @ values
    samples = (totals[:, 1] - totals[:, 0]) / totals[:, 2]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _price_bin_start(prices: pd.Series, width: float) -> pd.Series:
    numeric = pd.to_numeric(prices, errors="raise").astype(float)
    if not numeric.between(0.0, 1.0).all():
        raise ValueError("token prices must be between zero and one")
    bins = max(1, int(round(1.0 / width)))
    indexes = np.floor(numeric / width + 1e-12).astype(int).clip(0, bins - 1)
    return (indexes * width).round(12)


def _settlement_boundary(target_date: str, timezone: str) -> pd.Timestamp:
    local_midnight = pd.Timestamp(target_date).tz_localize(timezone)
    return (local_midnight + pd.DateOffset(days=1)).tz_convert("UTC")


def _fold_lookup(plan: HistoricalCrossfitPlan) -> dict[str, str]:
    result: dict[str, str] = {}
    for fold in plan.folds:
        for stamp in pd.date_range(fold.predict[0], fold.predict[1], freq="D"):
            result[stamp.date().isoformat()] = fold.fold_id
    return result
