"""Model-selected adjacent two-bucket union research."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.synthetic_quotes import build_synthetic_book
from core.pricing.fees import taker_fee_usdc
from core.pricing.quote import NormalizedBook, Quote, compute_quote_for_shares


_SELECTION_REQUIRED = {
    "event_id",
    "target_date",
    "timestamp",
    "snapshot_id",
    "bucket_order",
    "condition_id",
    "bucket_lo_f",
    "bucket_hi_f",
    "model_probability",
    "midpoint_probability",
    "lead_hours",
    "lead_band",
}

_FOLD_TARGET_RANGES = {
    "fold-00": ("2025-04-01", "2025-05-31"),
    "fold-01": ("2025-06-01", "2025-07-31"),
    "fold-02": ("2025-08-01", "2025-09-30"),
    "fold-03": ("2025-10-01", "2025-11-30"),
    "fold-04": ("2025-12-01", "2026-01-31"),
    "fold-05": ("2026-02-01", "2026-03-31"),
}


@dataclass(frozen=True)
class TakerUnionExecution:
    """All-or-skip equal-share execution for a mutually exclusive union."""

    complete_fill: bool
    shares: float
    total_cost_usd: float
    leg_quotes: tuple[Quote, ...]
    leg_fees_usd: tuple[float, ...]
    rejection_reason: str | None = None


@dataclass(frozen=True)
class UnionReplay:
    """Auditable decisions, individual legs, and Event settlements."""

    decisions: pd.DataFrame
    legs: pd.DataFrame
    settlements: pd.DataFrame


@dataclass(frozen=True)
class FrozenUnionCalibration:
    """Pre-evaluation union calibrator fitted from historical Event days."""

    intercept: float
    slope: float
    bootstrap_intercepts: tuple[float, ...]
    bootstrap_slopes: tuple[float, ...]
    fit_target_start: str
    fit_target_end: str
    fit_event_days: int
    ridge_strength: float
    seed: int

    def apply(self, selections: pd.DataFrame) -> pd.DataFrame:
        """Score ex-ante model probabilities without consulting outcomes."""
        if "model_union_probability" not in selections:
            raise ValueError("union diagnostics require model_union_probability")
        probability = selections["model_union_probability"].to_numpy(float)
        if (
            not np.isfinite(probability).all()
            or ((probability < 0.0) | (probability > 1.0)).any()
        ):
            raise ValueError("model union probabilities must be finite in [0, 1]")
        result = selections.copy()
        beta = np.asarray([self.intercept, self.slope], dtype=float)
        result["calibrated_union_probability"] = _logistic_predict(probability, beta)
        bootstrap_beta = np.column_stack(
            [self.bootstrap_intercepts, self.bootstrap_slopes]
        )
        bootstrap = np.vstack(
            [_logistic_predict(probability, draw) for draw in bootstrap_beta]
        )
        result["union_probability_lower_90"] = np.quantile(bootstrap, 0.1, axis=0)
        result["union_probability_upper_90"] = np.quantile(bootstrap, 0.9, axis=0)
        return result

    def coefficients(self) -> pd.DataFrame:
        """Return the complete reproducible calibration state."""
        return pd.DataFrame(
            {
                "coefficient_kind": ["point"] + ["bootstrap"] * len(self.bootstrap_intercepts),
                "draw_index": [-1] + list(range(len(self.bootstrap_intercepts))),
                "intercept": [self.intercept, *self.bootstrap_intercepts],
                "slope": [self.slope, *self.bootstrap_slopes],
                "fit_target_start": self.fit_target_start,
                "fit_target_end": self.fit_target_end,
                "fit_event_days": self.fit_event_days,
                "ridge_strength": self.ridge_strength,
                "seed": self.seed,
            }
        )


def price_equal_share_taker_union(
    books: Sequence[NormalizedBook],
    *,
    fee_schedules: Sequence[Mapping[str, object]],
    risk_usd: float = 50.0,
) -> TakerUnionExecution:
    """Find the largest equal-share union basket within an all-in risk cap."""
    if len(books) < 2 or len(books) != len(fee_schedules):
        raise ValueError("union pricing requires matching books and fee schedules")
    if risk_usd <= 0:
        raise ValueError("risk_usd must be positive")
    available = [
        sum(max(0.0, float(level["size"])) for level in book.get("asks", []))
        for book in books
    ]
    upper = min(available)
    if upper <= 0:
        return TakerUnionExecution(False, 0.0, 0.0, (), (), "insufficient_depth")

    def price(shares: float) -> tuple[tuple[Quote, ...], tuple[float, ...], float]:
        quotes = tuple(
            compute_quote_for_shares(book, "BUY", shares) for book in books
        )
        if not all(quote.complete_fill for quote in quotes):
            return quotes, (), float("inf")
        fees = tuple(
            taker_fee_usdc(
                shares=quote.filled_shares,
                price=quote.vwap_price,
                fees_enabled=bool(schedule.get("fees_enabled", False)),
                fee_rate=schedule.get("fee_rate"),
                fee_exponent=schedule.get("fee_exponent", 1.0),
            )
            for quote, schedule in zip(quotes, fee_schedules, strict=True)
        )
        total = sum(quote.max_fillable_usd for quote in quotes) + sum(fees)
        return quotes, fees, float(total)

    low, high = 0.0, upper
    upper_quotes, upper_fees, upper_cost = price(upper)
    if upper_cost <= risk_usd + 1e-9:
        return TakerUnionExecution(
            True, upper, upper_cost, upper_quotes, upper_fees
        )
    for _ in range(80):
        midpoint = (low + high) / 2.0
        _, _, total = price(midpoint)
        if total <= risk_usd:
            low = midpoint
        else:
            high = midpoint
    if low <= 1e-9:
        return TakerUnionExecution(False, 0.0, 0.0, (), (), "risk_below_minimum_fill")
    quotes, fees, total = price(low)
    if not all(quote.complete_fill for quote in quotes) or total > risk_usd + 1e-7:
        return TakerUnionExecution(False, 0.0, 0.0, (), (), "incomplete_equal_share_fill")
    return TakerUnionExecution(True, low, total, quotes, fees)


def simulate_strict_maker_fills(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Fill resting YES buys only on later, strict sell-through evidence."""
    order_required = {
        "leg_id", "condition_id", "placed_at", "expires_at",
        "limit_price", "target_shares",
    }
    trade_required = {
        "condition_id", "timestamp", "outcome", "side", "price", "size",
    }
    missing_orders = order_required - set(orders)
    missing_trades = trade_required - set(trades)
    if missing_orders:
        raise ValueError("maker orders missing columns: " + ", ".join(sorted(missing_orders)))
    if missing_trades:
        raise ValueError("maker trades missing columns: " + ", ".join(sorted(missing_trades)))
    work = trades.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["outcome"] = work["outcome"].astype(str).str.upper()
    work["side"] = work["side"].astype(str).str.upper()
    if not work["outcome"].isin(["YES", "NO"]).all():
        raise ValueError("maker trade outcome must be YES or NO")
    if not work["side"].isin(["BUY", "SELL"]).all():
        raise ValueError("maker trade side must be BUY or SELL")
    work["yes_price"] = np.where(
        work["outcome"] == "YES",
        work["price"].astype(float),
        1.0 - work["price"].astype(float),
    )
    work["yes_side"] = np.where(
        ((work["outcome"] == "YES") & (work["side"] == "BUY"))
        | ((work["outcome"] == "NO") & (work["side"] == "SELL")),
        "BUY",
        "SELL",
    )
    trade_groups = {
        str(condition_id): group.sort_values("timestamp", kind="stable")
        for condition_id, group in work.groupby("condition_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    passthrough = [column for column in orders if column not in order_required]
    for order in orders.to_dict("records"):
        placed = pd.Timestamp(order["placed_at"])
        expires = pd.Timestamp(order["expires_at"])
        placed = placed.tz_localize("UTC") if placed.tzinfo is None else placed.tz_convert("UTC")
        expires = expires.tz_localize("UTC") if expires.tzinfo is None else expires.tz_convert("UTC")
        target = float(order["target_shares"])
        limit = float(order["limit_price"])
        if target <= 0 or not 0.0 < limit < 1.0 or expires <= placed:
            raise ValueError("maker order has invalid size, price, or expiry")
        condition_trades = trade_groups.get(str(order["condition_id"]), work.iloc[0:0])
        visible = condition_trades[
            (condition_trades["timestamp"] > placed)
            & (condition_trades["timestamp"] <= expires)
            & (condition_trades["yes_side"] == "SELL")
        ].sort_values("timestamp", kind="stable")
        strict = visible[visible["yes_price"] < limit - 1e-12]
        touches = visible[np.isclose(visible["yes_price"], limit, atol=1e-12)]
        cumulative = strict["size"].astype(float).clip(lower=0.0).cumsum()
        filled = min(target, float(cumulative.iloc[-1]) if len(cumulative) else 0.0)
        first_fill_at = strict["timestamp"].iloc[0] if len(strict) else pd.NaT
        complete_fill_at = pd.NaT
        if filled >= target - 1e-9:
            complete_fill_at = strict.loc[cumulative >= target - 1e-9, "timestamp"].iloc[0]
            status = "full"
        elif filled > 0:
            status = "partial"
        else:
            status = "none"
        rows.append({
            **{column: order[column] for column in passthrough},
            "leg_id": str(order["leg_id"]),
            "condition_id": str(order["condition_id"]),
            "placed_at": placed,
            "expires_at": expires,
            "limit_price": limit,
            "target_shares": target,
            "filled_shares": filled,
            "fill_cost_usd": filled * limit,
            "strict_trade_through_shares": float(strict["size"].astype(float).clip(lower=0.0).sum()),
            "touch_unknown_shares": float(touches["size"].astype(float).clip(lower=0.0).sum()),
            "first_fill_at": first_fill_at,
            "complete_fill_at": complete_fill_at,
            "fill_status": status,
        })
    return pd.DataFrame(rows)


def fit_expanding_union_calibration(
    selections: pd.DataFrame,
    *,
    bootstrap_samples: int = 500,
    seed: int = 42,
    ridge_strength: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate each fold from strictly earlier folds with Event-equal weight."""
    required = {
        "event_id", "target_date", "fold_id",
        "model_union_probability", "union_outcome",
    }
    missing = required - set(selections)
    if missing:
        raise ValueError("union calibration missing columns: " + ", ".join(sorted(missing)))
    if bootstrap_samples <= 0 or ridge_strength < 0:
        raise ValueError("bootstrap_samples must be positive and ridge non-negative")
    work = selections.copy().reset_index(drop=True)
    probability = work["model_union_probability"].to_numpy(float)
    outcome = work["union_outcome"].to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("model union probabilities must be finite in [0, 1]")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("union outcomes must be binary")
    folds = sorted(work["fold_id"].astype(str).unique())
    calibrated_parts: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    for fold_index, fold_id in enumerate(folds[1:], start=1):
        fit_folds = folds[:fold_index]
        training = work[work["fold_id"].astype(str).isin(fit_folds)].copy()
        evaluation = work[work["fold_id"].astype(str) == fold_id].copy()
        if training["event_id"].nunique() < 2 or evaluation.empty:
            raise ValueError(f"insufficient expanding calibration support for {fold_id}")
        fit_weights = _event_equal_weights(training)
        beta = _fit_weighted_logistic(
            training["model_union_probability"].to_numpy(float),
            training["union_outcome"].to_numpy(float),
            fit_weights,
            ridge_strength=ridge_strength,
        )
        calibrated = _logistic_predict(
            evaluation["model_union_probability"].to_numpy(float), beta
        )
        bootstraps = _bootstrap_logistic_predictions(
            training,
            evaluation["model_union_probability"].to_numpy(float),
            samples=bootstrap_samples,
            seed=seed + fold_index,
            ridge_strength=ridge_strength,
        )
        evaluation["calibrated_union_probability"] = calibrated
        evaluation["union_probability_lower_90"] = np.quantile(bootstraps, 0.1, axis=0)
        evaluation["union_probability_upper_90"] = np.quantile(bootstraps, 0.9, axis=0)
        calibrated_parts.append(evaluation)
        contracts.append({
            "fold_id": fold_id,
            "fit_fold_ids": list(fit_folds),
            "fit_target_start": str(training["target_date"].astype(str).min()),
            "fit_target_end": str(training["target_date"].astype(str).max()),
            "eval_target_start": str(evaluation["target_date"].astype(str).min()),
            "eval_target_end": str(evaluation["target_date"].astype(str).max()),
            "fit_event_days": int(training["event_id"].nunique()),
            "intercept": float(beta[0]),
            "slope": float(beta[1]),
            "ridge_strength": float(ridge_strength),
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed + fold_index),
        })
    calibrated_frame = (
        pd.concat(calibrated_parts, ignore_index=True)
        if calibrated_parts else pd.DataFrame(columns=list(work) + [
            "calibrated_union_probability", "union_probability_lower_90",
            "union_probability_upper_90",
        ])
    )
    return calibrated_frame, pd.DataFrame(contracts)


def fit_frozen_union_calibration(
    training: pd.DataFrame,
    *,
    bootstrap_samples: int = 500,
    seed: int = 42,
    ridge_strength: float = 1.0,
) -> FrozenUnionCalibration:
    """Fit the final deployment calibrator from pre-evaluation Event days."""
    required = {
        "event_id", "target_date", "model_union_probability", "union_outcome",
    }
    missing = required - set(training)
    if missing:
        raise ValueError("frozen union calibration missing columns: " + ", ".join(sorted(missing)))
    if bootstrap_samples <= 0 or ridge_strength < 0:
        raise ValueError("bootstrap_samples must be positive and ridge non-negative")
    work = training.copy().reset_index(drop=True)
    if work.empty or work["event_id"].nunique() < 2:
        raise ValueError("frozen union calibration requires at least two Event days")
    probability = work["model_union_probability"].to_numpy(float)
    outcome = work["union_outcome"].to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("model union probabilities must be finite in [0, 1]")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("union outcomes must be binary")
    beta = _fit_weighted_logistic(
        probability,
        outcome,
        _event_equal_weights(work),
        ridge_strength=ridge_strength,
    )
    bootstrap_beta = _bootstrap_logistic_coefficients(
        work,
        samples=bootstrap_samples,
        seed=seed,
        ridge_strength=ridge_strength,
    )
    return FrozenUnionCalibration(
        intercept=float(beta[0]),
        slope=float(beta[1]),
        bootstrap_intercepts=tuple(float(value) for value in bootstrap_beta[:, 0]),
        bootstrap_slopes=tuple(float(value) for value in bootstrap_beta[:, 1]),
        fit_target_start=str(work["target_date"].astype(str).min()),
        fit_target_end=str(work["target_date"].astype(str).max()),
        fit_event_days=int(work["event_id"].nunique()),
        ridge_strength=float(ridge_strength),
        seed=int(seed),
    )


def replay_taker_unions(
    candidates: pd.DataFrame,
    *,
    stage: str,
    edge_threshold: float,
    risk_usd: float = 50.0,
) -> UnionReplay:
    """Replay one all-or-skip equal-share taker basket per Event."""
    if stage not in {"unconditional", "frozen"}:
        raise ValueError("stage must be unconditional or frozen")
    required = {
        "event_id", "target_date", "timestamp", "snapshot_id",
        "lower_condition_id", "upper_condition_id",
        "lower_yes_outcome", "upper_yes_outcome", "union_outcome",
        "model_union_probability", "lead_hours", "lead_band",
    }
    for prefix in ("lower", "upper"):
        required.update({
            f"{prefix}_bid_price_yes", f"{prefix}_ask_price_yes",
            f"{prefix}_capacity_usdc", f"{prefix}_impact_per_usdc",
            f"{prefix}_fees_enabled", f"{prefix}_fee_rate",
            f"{prefix}_fee_exponent", f"{prefix}_yes_token_id",
        })
    if stage == "frozen":
        required.add("union_probability_lower_90")
    missing = required - set(candidates)
    if missing:
        raise ValueError("taker replay missing columns: " + ", ".join(sorted(missing)))
    if risk_usd <= 0 or edge_threshold < 0:
        raise ValueError("risk_usd must be positive and edge_threshold non-negative")
    work = candidates.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values(["event_id", "timestamp", "snapshot_id"], kind="stable")
    decision_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    settlement_rows: list[dict[str, object]] = []
    for event_id, event_rows in work.groupby("event_id", sort=True):
        rows = event_rows.head(1) if stage == "unconditional" else event_rows
        for row in rows.itertuples(index=False):
            books = tuple(_book_from_wide_row(row, prefix) for prefix in ("lower", "upper"))
            schedules = tuple(_fee_schedule_from_wide_row(row, prefix) for prefix in ("lower", "upper"))
            execution = price_equal_share_taker_union(
                books, fee_schedules=schedules, risk_usd=risk_usd
            )
            probability_bound = float(
                row.model_union_probability
                if stage == "unconditional"
                else row.union_probability_lower_90
            )
            unit_cost = (
                execution.total_cost_usd / execution.shares
                if execution.complete_fill and execution.shares > 0 else float("nan")
            )
            edge = probability_bound - unit_cost if np.isfinite(unit_cost) else float("nan")
            qualified = bool(
                execution.complete_fill
                and (stage == "unconditional" or edge > edge_threshold)
            )
            common = _candidate_identity(row, stage=stage, edge_threshold=edge_threshold)
            decision_rows.append({
                **common,
                "execution_mode": "taker",
                "probability_bound": probability_bound,
                "effective_unit_cost": unit_cost,
                "conservative_edge": edge,
                "qualified": qualified,
                "rejection_reason": execution.rejection_reason if not execution.complete_fill else (
                    None if qualified else "edge_below_threshold"
                ),
            })
            if not qualified:
                continue
            for index, prefix in enumerate(("lower", "upper")):
                quote = execution.leg_quotes[index]
                fee = execution.leg_fees_usd[index]
                outcome = float(getattr(row, f"{prefix}_yes_outcome"))
                leg_rows.append({
                    **common,
                    "execution_mode": "taker",
                    "leg_id": prefix,
                    "condition_id": str(getattr(row, f"{prefix}_condition_id")),
                    "token_id": str(getattr(row, f"{prefix}_yes_token_id")),
                    "shares": execution.shares,
                    "filled_shares": execution.shares,
                    "vwap_price": quote.vwap_price,
                    "limit_price": float("nan"),
                    "notional_usd": quote.max_fillable_usd,
                    "fee_usd": fee,
                    "token_outcome": outcome,
                    "payout_usd": execution.shares * outcome,
                    "fill_status": "full",
                })
            payout = execution.shares * float(row.union_outcome)
            settlement_rows.append({
                **common,
                "execution_mode": "taker",
                "shares": execution.shares,
                "union_outcome": float(row.union_outcome),
                "payout_usd": payout,
                "cost_basis_usd": execution.total_cost_usd,
                "realized_pnl_usd": payout - execution.total_cost_usd,
                "roi": (payout - execution.total_cost_usd) / execution.total_cost_usd,
                "basket_fill_status": "full",
            })
            break
    return UnionReplay(
        decisions=pd.DataFrame(decision_rows),
        legs=pd.DataFrame(leg_rows),
        settlements=pd.DataFrame(settlement_rows),
    )


def replay_maker_unions(
    candidates: pd.DataFrame,
    *,
    trades: pd.DataFrame,
    stage: str,
    edge_threshold: float,
    risk_usd: float = 50.0,
    max_ttl: str | pd.Timedelta = "60min",
) -> UnionReplay:
    """Replay resting equal-share baskets with strict trade-through fills."""
    if stage not in {"unconditional", "frozen"}:
        raise ValueError("stage must be unconditional or frozen")
    required = {
        "event_id", "target_date", "timestamp", "snapshot_id",
        "lower_condition_id", "upper_condition_id",
        "lower_yes_outcome", "upper_yes_outcome", "union_outcome",
        "model_union_probability", "lead_hours", "lead_band",
        "lower_bid_price_yes", "lower_ask_price_yes",
        "upper_bid_price_yes", "upper_ask_price_yes",
        "lower_yes_token_id", "upper_yes_token_id",
    }
    if stage == "frozen":
        required.add("union_probability_lower_90")
    missing = required - set(candidates)
    if missing:
        raise ValueError("maker replay missing columns: " + ", ".join(sorted(missing)))
    ttl = pd.Timedelta(max_ttl)
    if risk_usd <= 0 or edge_threshold < 0 or ttl <= pd.Timedelta(0):
        raise ValueError("maker risk, threshold, and TTL are invalid")
    work = candidates.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values(["event_id", "timestamp", "snapshot_id"], kind="stable")
    decisions: list[dict[str, object]] = []
    orders: list[dict[str, object]] = []
    for event_id, event_rows in work.groupby("event_id", sort=True):
        event_rows = event_rows.reset_index(drop=True)
        scan = event_rows.head(1) if stage == "unconditional" else event_rows
        for row_index, row in scan.iterrows():
            lower_bid, upper_bid = float(row["lower_bid_price_yes"]), float(row["upper_bid_price_yes"])
            valid_post_only = (
                0.0 < lower_bid < float(row["lower_ask_price_yes"]) < 1.0
                and 0.0 < upper_bid < float(row["upper_ask_price_yes"]) < 1.0
            )
            unit_cost = lower_bid + upper_bid if valid_post_only else float("nan")
            bound = float(
                row["model_union_probability"]
                if stage == "unconditional" else row["union_probability_lower_90"]
            )
            edge = bound - unit_cost if np.isfinite(unit_cost) else float("nan")
            qualified = bool(valid_post_only and (stage == "unconditional" or edge > edge_threshold))
            common = _candidate_identity(
                SimpleNamespace(**row.to_dict()),
                stage=stage,
                edge_threshold=edge_threshold,
            )
            decisions.append({
                **common,
                "execution_mode": "maker",
                "probability_bound": bound,
                "effective_unit_cost": unit_cost,
                "conservative_edge": edge,
                "qualified": qualified,
                "rejection_reason": None if qualified else (
                    "edge_below_threshold" if valid_post_only else "invalid_post_only_quote"
                ),
            })
            if not qualified:
                continue
            later = event_rows.iloc[row_index + 1 :]
            later = later[later["snapshot_id"].astype(str) != str(row["snapshot_id"])]
            expiry = pd.Timestamp(row["timestamp"]) + ttl
            if not later.empty:
                expiry = min(expiry, pd.Timestamp(later.iloc[0]["timestamp"]))
            shares = risk_usd / unit_cost
            basket_id = f"{stage}:{edge_threshold:.4f}:{event_id}"
            for prefix, limit in (("lower", lower_bid), ("upper", upper_bid)):
                orders.append({
                    **common,
                    "basket_id": basket_id,
                    "leg_id": f"{basket_id}:{prefix}",
                    "leg_role": prefix,
                    "condition_id": str(row[f"{prefix}_condition_id"]),
                    "token_id": str(row[f"{prefix}_yes_token_id"]),
                    "token_outcome": float(row[f"{prefix}_yes_outcome"]),
                    "placed_at": pd.Timestamp(row["timestamp"]),
                    "expires_at": expiry,
                    "limit_price": limit,
                    "target_shares": shares,
                })
            break
    if not orders:
        return UnionReplay(pd.DataFrame(decisions), pd.DataFrame(), pd.DataFrame())
    fills = simulate_strict_maker_fills(pd.DataFrame(orders), trades)
    leg_rows: list[dict[str, object]] = []
    settlement_rows: list[dict[str, object]] = []
    for row in fills.to_dict("records"):
        leg_rows.append({
            **row,
            "execution_mode": "maker",
            "shares": float(row["target_shares"]),
            "vwap_price": float("nan"),
            "notional_usd": float(row["fill_cost_usd"]),
            "fee_usd": 0.0,
            "payout_usd": float(row["filled_shares"]) * float(row["token_outcome"]),
        })
    legs = pd.DataFrame(leg_rows)
    for basket_id, basket in legs.groupby("basket_id", sort=True):
        cost = float(basket["fill_cost_usd"].sum())
        payout = float(basket["payout_usd"].sum())
        statuses = basket["fill_status"].astype(str).tolist()
        any_filled = bool((basket["filled_shares"].astype(float) > 0).any())
        if all(status == "full" for status in statuses):
            basket_status = "full"
        elif not any_filled:
            basket_status = "none"
        elif sum(float(value) > 0 for value in basket["filled_shares"]) == 1:
            basket_status = "orphan"
        else:
            basket_status = "partial"
        first = basket.iloc[0]
        settlement_rows.append({
            "stage": first["stage"],
            "edge_threshold": float(first["edge_threshold"]),
            "event_id": str(first["event_id"]),
            "target_date": str(first["target_date"]),
            "timestamp": pd.Timestamp(first["timestamp"]),
            "snapshot_id": str(first["snapshot_id"]),
            "execution_mode": "maker",
            "shares": float(first["target_shares"]),
            "union_outcome": float(first["union_outcome"]),
            "payout_usd": payout,
            "cost_basis_usd": cost,
            "realized_pnl_usd": payout - cost,
            "roi": (payout - cost) / cost if cost > 0 else float("nan"),
            "basket_fill_status": basket_status,
            "fold_id": first.get("fold_id"),
            "coverage_type": first.get("coverage_type"),
            "lead_hours": float(first["lead_hours"]),
            "lead_band": str(first["lead_band"]),
            "model_union_probability": float(first["model_union_probability"]),
        })
    return UnionReplay(
        decisions=pd.DataFrame(decisions),
        legs=legs,
        settlements=pd.DataFrame(settlement_rows),
    )


def _book_from_wide_row(row: object, prefix: str) -> NormalizedBook:
    return build_synthetic_book(
        {
            "bid_price_yes": getattr(row, f"{prefix}_bid_price_yes"),
            "ask_price_yes": getattr(row, f"{prefix}_ask_price_yes"),
            "capacity_usdc": getattr(row, f"{prefix}_capacity_usdc"),
            "impact_per_usdc": getattr(row, f"{prefix}_impact_per_usdc"),
        },
        outcome="YES",
    )


def _fee_schedule_from_wide_row(row: object, prefix: str) -> dict[str, object]:
    return {
        "fees_enabled": bool(getattr(row, f"{prefix}_fees_enabled")),
        "fee_rate": getattr(row, f"{prefix}_fee_rate"),
        "fee_exponent": getattr(row, f"{prefix}_fee_exponent"),
    }


def _candidate_identity(
    row: object, *, stage: str, edge_threshold: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "stage": stage,
        "edge_threshold": float(edge_threshold),
        "event_id": str(row.event_id),
        "target_date": str(row.target_date),
        "timestamp": pd.Timestamp(row.timestamp),
        "snapshot_id": str(row.snapshot_id),
        "model_union_probability": float(row.model_union_probability),
        "union_outcome": float(row.union_outcome),
        "lead_hours": float(row.lead_hours),
        "lead_band": str(row.lead_band),
    }
    passthrough = (
        "fold_id", "evaluation_id", "probability_source", "coverage_type", "horizon_band",
        "model_market_gap", "model_market_gap_band",
        "modal_neighbor_direction", "midpoint_union_probability",
        "calibrated_union_probability", "union_probability_lower_90",
        "union_probability_upper_90",
    )
    probability_columns = {
        "midpoint_union_probability", "calibrated_union_probability",
        "union_probability_lower_90", "union_probability_upper_90",
    }
    for column in passthrough:
        if hasattr(row, column):
            value = getattr(row, column)
            result[column] = (
                float(value)
                if column in probability_columns
                else value
            )
    return result


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_id")["event_id"].transform("size").to_numpy(float)
    return 1.0 / counts


def _fit_weighted_logistic(
    probability: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    ridge_strength: float,
) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    x = np.column_stack([np.ones(len(clipped)), np.log(clipped / (1.0 - clipped))])
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = np.clip(np.average(y, weights=w), 1e-6, 1.0 - 1e-6)
    beta = np.asarray([np.log(mean / (1.0 - mean)), 0.0], dtype=float)
    for _ in range(100):
        fitted = _sigmoid(x @ beta)
        gradient = x.T @ (w * (fitted - y))
        gradient[1] += ridge_strength * beta[1]
        curvature = w * fitted * (1.0 - fitted)
        hessian = x.T @ (curvature[:, None] * x)
        hessian[1, 1] += ridge_strength
        hessian += np.eye(2) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _bootstrap_logistic_predictions(
    training: pd.DataFrame,
    evaluation_probability: np.ndarray,
    *,
    samples: int,
    seed: int,
    ridge_strength: float,
) -> np.ndarray:
    coefficients = _bootstrap_logistic_coefficients(
        training,
        samples=samples,
        seed=seed,
        ridge_strength=ridge_strength,
    )
    return np.vstack(
        [
            _logistic_predict(evaluation_probability, coefficient)
            for coefficient in coefficients
        ]
    )


def _bootstrap_logistic_coefficients(
    training: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    ridge_strength: float,
) -> np.ndarray:
    events = training["event_id"].astype(str).drop_duplicates().to_numpy()
    event_values = training["event_id"].astype(str).to_numpy()
    rows_per_event = pd.Series(event_values).value_counts().to_dict()
    rng = np.random.default_rng(seed)
    result = np.empty((samples, 2), dtype=float)
    for index in range(samples):
        sampled = rng.choice(events, size=len(events), replace=True)
        multiplicity = pd.Series(sampled).value_counts().to_dict()
        weights = np.asarray([
            float(multiplicity.get(event, 0)) / float(rows_per_event[event])
            for event in event_values
        ])
        result[index] = _fit_weighted_logistic(
            training["model_union_probability"].to_numpy(float),
            training["union_outcome"].to_numpy(float),
            weights,
            ridge_strength=ridge_strength,
        )
    return result


def _logistic_predict(probability: np.ndarray, beta: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return _sigmoid(beta[0] + beta[1] * logits)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def select_model_two_bucket_unions(panel: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-model-mass adjacent pair at each PIT market vector."""
    missing = _SELECTION_REQUIRED - set(panel)
    if missing:
        raise ValueError("two-bucket panel missing columns: " + ", ".join(sorted(missing)))
    rows: list[dict[str, object]] = []
    keys = ["event_id", "target_date", "timestamp", "snapshot_id"]
    for key, vector in panel.groupby(keys, sort=True, dropna=False):
        ordered = vector.sort_values("bucket_order", kind="stable").reset_index(drop=True)
        orders = ordered["bucket_order"].to_numpy(int)
        if len(ordered) < 2 or not np.array_equal(orders, np.arange(len(ordered))):
            raise ValueError("two-bucket selection requires a complete contiguous market vector")
        probabilities = ordered["model_probability"].to_numpy(float)
        midpoint = ordered["midpoint_probability"].to_numpy(float)
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("model probabilities must be finite and non-negative")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError("model probability vector must sum to one")
        if not np.isfinite(midpoint).all() or (midpoint < 0.0).any():
            raise ValueError("midpoint probabilities must be finite and non-negative")
        if not np.isclose(midpoint.sum(), 1.0, atol=1e-6):
            raise ValueError("midpoint probability vector must sum to one")

        pair_mass = probabilities[:-1] + probabilities[1:]
        pair_start = int(np.flatnonzero(np.isclose(pair_mass, pair_mass.max(), atol=1e-12))[0])
        modal_order = int(np.flatnonzero(np.isclose(probabilities, probabilities.max(), atol=1e-12))[0])
        lower = ordered.iloc[pair_start]
        upper = ordered.iloc[pair_start + 1]
        if modal_order == pair_start:
            direction = "upper"
        elif modal_order == pair_start + 1:
            direction = "lower"
        else:
            direction = "off_mode"
        row = {
            "event_id": str(key[0]),
            "target_date": str(key[1]),
            "timestamp": pd.Timestamp(key[2]),
            "snapshot_id": str(key[3]),
            "pair_start_order": pair_start,
            "pair_end_order": pair_start + 1,
            "bucket_count": int(len(ordered)),
            "modal_bucket_order": modal_order,
            "modal_neighbor_direction": direction,
            "lower_condition_id": str(lower["condition_id"]),
            "upper_condition_id": str(upper["condition_id"]),
            "lower_bucket_lo_f": float(lower["bucket_lo_f"]),
            "lower_bucket_hi_f": float(lower["bucket_hi_f"]),
            "upper_bucket_lo_f": float(upper["bucket_lo_f"]),
            "upper_bucket_hi_f": float(upper["bucket_hi_f"]),
            "model_union_probability": float(pair_mass[pair_start]),
            "midpoint_union_probability": float(midpoint[pair_start : pair_start + 2].sum()),
            "lead_hours": float(lower["lead_hours"]),
            "lead_band": str(lower["lead_band"]),
        }
        for column in ("coverage_type", "forecast_hour_count", "lineage_id"):
            if column in ordered:
                row[column] = lower[column]
        rows.append(row)
    return pd.DataFrame(rows)


def build_two_bucket_candidate_panel(
    *,
    panel: pd.DataFrame,
    snapshots: pd.DataFrame,
    quotes: pd.DataFrame,
    markets: pd.DataFrame,
    evaluation_id: str | None = None,
    evaluation_target_range: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Bind ex-ante pair selection to truth and executable evidence."""
    explicit_evaluation = evaluation_id is not None or evaluation_target_range is not None
    if explicit_evaluation and (
        not evaluation_id or evaluation_target_range is None
    ):
        raise ValueError("explicit evaluation requires an id and target range")
    snapshot_required = {
        "snapshot_id", "lineage_id", "coverage_type", "forecast_hour_count",
    }
    quote_required = {
        "condition_id", "timestamp", "bid_price_yes", "ask_price_yes",
        "capacity_usdc", "impact_per_usdc",
    }
    market_required = {
        "condition_id", "bucket_lo_f", "bucket_hi_f", "yes_token_id",
        "fees_enabled", "fee_rate", "fee_exponent",
    }
    for name, frame, required in (
        ("snapshots", snapshots, snapshot_required),
        ("quotes", quotes, quote_required),
        ("markets", markets, market_required),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(sorted(missing)))
    if "yes_outcome" not in panel:
        raise ValueError("signal panel must contain settlement outcome")
    snapshot_meta = snapshots.loc[:, sorted(snapshot_required)].drop_duplicates("snapshot_id")
    if len(snapshot_meta) != snapshots["snapshot_id"].nunique():
        raise ValueError("snapshot metadata is not unique")
    enriched = panel.merge(
        snapshot_meta,
        on="snapshot_id",
        how="inner",
        validate="many_to_one",
    )
    if len(enriched) != len(panel):
        raise ValueError("signal panel contains snapshot outside ForecastCache")
    selected = select_model_two_bucket_unions(
        enriched.drop(columns=["yes_outcome", "actual_high_f"], errors="ignore")
    )
    truth_keys = ["event_id", "target_date", "timestamp", "snapshot_id", "condition_id"]
    truth = enriched.loc[:, truth_keys + ["yes_outcome"]].copy()
    for prefix in ("lower", "upper"):
        lookup = truth.rename(columns={
            "condition_id": f"{prefix}_condition_id",
            "yes_outcome": f"{prefix}_yes_outcome",
        })
        selected = selected.merge(
            lookup,
            on=["event_id", "target_date", "timestamp", "snapshot_id", f"{prefix}_condition_id"],
            how="left",
            validate="one_to_one",
        )
    if selected[["lower_yes_outcome", "upper_yes_outcome"]].isna().any().any():
        raise ValueError("selected pair outcome cannot be resolved")
    selected["union_outcome"] = (
        selected["lower_yes_outcome"].astype(float)
        + selected["upper_yes_outcome"].astype(float)
    )
    if not selected["union_outcome"].isin([0.0, 1.0]).all():
        raise ValueError("selected union outcome must be binary")

    quote_rows = quotes.loc[:, sorted(quote_required)].copy()
    quote_rows["timestamp"] = pd.to_datetime(quote_rows["timestamp"], utc=True)
    if quote_rows.duplicated(["condition_id", "timestamp"]).any():
        raise ValueError("synthetic quote keys are not unique")
    market_rows = markets.loc[:, sorted(market_required)].copy()
    if market_rows["condition_id"].astype(str).duplicated().any():
        raise ValueError("market condition contract is not unique")
    for prefix in ("lower", "upper"):
        prefixed_quotes = quote_rows.rename(columns={
            column: f"{prefix}_{column}"
            for column in quote_required if column not in {"condition_id", "timestamp"}
        }).rename(columns={"condition_id": f"{prefix}_condition_id"})
        selected = selected.merge(
            prefixed_quotes,
            on=[f"{prefix}_condition_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        prefixed_markets = market_rows.rename(columns={
            column: f"{prefix}_{column}"
            for column in market_required if column != "condition_id"
        }).rename(columns={"condition_id": f"{prefix}_condition_id"})
        selected = selected.merge(
            prefixed_markets,
            on=f"{prefix}_condition_id",
            how="left",
            validate="many_to_one",
        )
        required_bound = selected[f"{prefix}_bucket_lo_f_x"].to_numpy(float)
        market_bound = selected[f"{prefix}_bucket_lo_f_y"].to_numpy(float)
        required_upper = selected[f"{prefix}_bucket_hi_f_x"].to_numpy(float)
        market_upper = selected[f"{prefix}_bucket_hi_f_y"].to_numpy(float)
        if not (
            np.allclose(required_bound, market_bound, equal_nan=False)
            and np.allclose(required_upper, market_upper, equal_nan=False)
        ):
            raise ValueError("selected bucket bounds do not match market contract")
        selected = selected.drop(columns=[
            f"{prefix}_bucket_lo_f_y", f"{prefix}_bucket_hi_f_y",
        ]).rename(columns={
            f"{prefix}_bucket_lo_f_x": f"{prefix}_bucket_lo_f",
            f"{prefix}_bucket_hi_f_x": f"{prefix}_bucket_hi_f",
        })
    quote_columns = [
        f"{prefix}_{column}"
        for prefix in ("lower", "upper")
        for column in ("bid_price_yes", "ask_price_yes", "capacity_usdc", "impact_per_usdc")
    ]
    if selected[quote_columns].isna().any().any():
        raise ValueError("selected pair lacks exact synthetic quote evidence")
    if explicit_evaluation:
        start, end = evaluation_target_range
        if start > end:
            raise ValueError("explicit evaluation target range is invalid")
        target_dates = selected["target_date"].astype(str)
        if ((target_dates < start) | (target_dates > end)).any():
            raise ValueError("candidate target date lies outside explicit evaluation range")
        selected["evaluation_id"] = str(evaluation_id)
    else:
        selected["fold_id"] = selected["lineage_id"].astype(str).str.extract(
            r"(fold-\d+)$", expand=False
        )
        if selected["fold_id"].isna().any():
            raise ValueError("snapshot lineage cannot be mapped to historical fold")
        invalid = selected.apply(
            lambda row: not (
                _FOLD_TARGET_RANGES[str(row["fold_id"])][0]
                <= str(row["target_date"])
                <= _FOLD_TARGET_RANGES[str(row["fold_id"])][1]
            ),
            axis=1,
        )
        if invalid.any():
            raise ValueError("candidate target date lies outside cross-fit fold")
    return selected.sort_values(
        ["event_id", "timestamp", "snapshot_id"], kind="stable"
    ).reset_index(drop=True)


def build_two_bucket_report(
    *,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    legs: pd.DataFrame,
    settlements: pd.DataFrame,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> dict[str, object]:
    """Summarize selection and absolute PnL with question-level closure."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    first = (
        candidates.sort_values(["event_id", "timestamp"], kind="stable")
        .groupby("event_id", as_index=False, sort=True)
        .head(1)
    )
    directions = (
        first["modal_neighbor_direction"].astype(str).value_counts().sort_index().to_dict()
        if "modal_neighbor_direction" in first else {}
    )
    selection = {
        "event_days": int(first["event_id"].nunique()) if len(first) else 0,
        "union_hit_rate": float(first["union_outcome"].mean()) if len(first) else None,
        "mean_model_union_probability": (
            float(first["model_union_probability"].mean()) if len(first) else None
        ),
        "mean_normalized_midpoint_union_probability": (
            float(first["midpoint_union_probability"].mean())
            if len(first) and "midpoint_union_probability" in first else None
        ),
        "direction_counts": {str(key): int(value) for key, value in directions.items()},
    }
    execution_overall: list[dict[str, object]] = []
    if not settlements.empty:
        for index, (keys, group) in enumerate(
            settlements.groupby(["stage", "execution_mode", "edge_threshold"], sort=True)
        ):
            execution_overall.append({
                "stage": str(keys[0]),
                "execution_mode": str(keys[1]),
                "edge_threshold": float(keys[2]),
                **_pnl_summary(group, bootstrap_samples=bootstrap_samples, seed=seed + index),
            })
    by_fold = _stratified_pnl(
        settlements, "fold_id", bootstrap_samples=bootstrap_samples, seed=seed + 100
    )
    by_coverage = _stratified_pnl(
        settlements, "coverage_type", bootstrap_samples=bootstrap_samples, seed=seed + 200
    )
    by_horizon = _stratified_pnl(
        settlements, "lead_band", bootstrap_samples=bootstrap_samples, seed=seed + 300
    )
    fill_status = (
        settlements[settlements["execution_mode"] == "maker"]["basket_fill_status"]
        .astype(str).value_counts().sort_index().to_dict()
        if not settlements.empty else {}
    )
    unconditional_taker = _find_execution_summary(
        execution_overall, stage="unconditional", mode="taker", threshold=0.0
    )
    frozen_taker = _find_execution_summary(
        execution_overall, stage="frozen", mode="taker", threshold=0.0
    )
    frozen_taker_folds = [
        row for row in by_fold
        if row["stage"] == "frozen"
        and row["execution_mode"] == "taker"
        and np.isclose(float(row["edge_threshold"]), 0.0)
    ]
    frozen_evidence = {
        "overall": frozen_taker,
        "positive_fold_count": sum(
            float(row["mean_event_day_pnl_usd"]) > 0 for row in frozen_taker_folds
        ),
        "evaluated_fold_count": len(frozen_taker_folds),
        "folds": frozen_taker_folds,
    }
    maker_comparison = _paired_mode_summary(
        settlements, bootstrap_samples=bootstrap_samples, seed=seed + 400
    )
    ledger = [
        {
            "question_id": "model_selected_neighbor_direction",
            "question": "Does the forecast distribution select lower and upper adjacent buckets rather than a fixed side?",
            "status": (
                "supported" if directions.get("lower", 0) > 0 and directions.get("upper", 0) > 0
                else "inconclusive"
            ),
            "evidence": selection,
            "artifact": "selection_panel.parquet",
        },
        {
            "question_id": "selected_union_contains_settlement",
            "question": "Does the model-selected adjacent pair contain the settlement bucket more often than not?",
            "status": (
                "supported" if selection["union_hit_rate"] is not None
                and float(selection["union_hit_rate"]) > 0.5
                else "contradicted" if selection["union_hit_rate"] is not None
                and float(selection["union_hit_rate"]) < 0.5
                else "inconclusive"
            ),
            "evidence": selection,
            "artifact": "selection_panel.parquet",
        },
        {
            "question_id": "unconditional_taker_absolute_pnl",
            "question": "Is the unconditional model-selected two-bucket taker strategy profitable after executable cost and fees?",
            "status": _profit_status(unconditional_taker),
            "evidence": unconditional_taker,
            "artifact": "event_pnl.parquet",
        },
        {
            "question_id": "frozen_deployment_taker_pnl",
            "question": "Does earlier-fold calibration and a frozen positive-LCB edge gate sustain taker profit?",
            "status": _frozen_profit_status(frozen_taker, frozen_taker_folds),
            "evidence": frozen_evidence,
            "artifact": "event_pnl.parquet",
        },
        {
            "question_id": "maker_after_fill_risk",
            "question": "Does strict-trade-through maker execution improve PnL after no-fill and orphan risk?",
            "status": _paired_status(maker_comparison),
            "evidence": maker_comparison,
            "artifact": "execution_legs.parquet",
        },
        {
            "question_id": "forecast_stage_and_horizon_heterogeneity",
            "question": "Are results materially different by forecast-only, partial-day, or horizon state?",
            "status": "inconclusive",
            "evidence": {
                "by_coverage_type": by_coverage,
                "by_horizon": by_horizon,
                "limitation": "descriptive strata measure risk concentration but do not identify a causal mechanism",
            },
            "artifact": "report.json",
        },
    ]
    return {
        "schema_version": 1,
        "data_class": "model_selected_two_bucket_union_study",
        "primary_estimand": "absolute_realized_pnl_after_execution_cost",
        "midpoint_role": "diagnostic_only",
        "selection": selection,
        "execution": {
            "overall": execution_overall,
            "by_fold": by_fold,
            "by_coverage_type": by_coverage,
            "by_horizon": by_horizon,
            "maker_fill_status_counts": {str(key): int(value) for key, value in fill_status.items()},
            "maker_minus_taker_paired": maker_comparison,
        },
        "counts": {
            "candidates": int(len(candidates)),
            "decisions": int(len(decisions)),
            "legs": int(len(legs)),
            "settlements": int(len(settlements)),
        },
        "decision_ledger": ledger,
    }


def _pnl_summary(
    frame: pd.DataFrame, *, bootstrap_samples: int, seed: int,
) -> dict[str, object]:
    by_day = frame.groupby("target_date", sort=True)["realized_pnl_usd"].sum()
    values = by_day.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(bootstrap_samples, len(values)))]
    means = draws.mean(axis=1)
    cost = float(frame["cost_basis_usd"].sum())
    pnl = float(frame["realized_pnl_usd"].sum())
    return {
        "event_days": int(len(by_day)),
        "attempts_or_trades": int(len(frame)),
        "union_hit_rate": float(frame["union_outcome"].mean()),
        "turnover_usd": cost,
        "total_pnl_usd": pnl,
        "roi": pnl / cost if cost > 0 else None,
        "mean_event_day_pnl_usd": float(values.mean()),
        "mean_event_day_pnl_ci90_low": float(np.quantile(means, 0.05)),
        "mean_event_day_pnl_ci90_high": float(np.quantile(means, 0.95)),
        "bootstrap_probability_mean_pnl_positive": float(np.mean(means > 0.0)),
    }


def _stratified_pnl(
    settlements: pd.DataFrame,
    stratum: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if settlements.empty or stratum not in settlements:
        return []
    rows: list[dict[str, object]] = []
    valid = settlements.dropna(subset=[stratum])
    for index, (keys, group) in enumerate(
        valid.groupby(["stage", "execution_mode", "edge_threshold", stratum], sort=True)
    ):
        rows.append({
            "stage": str(keys[0]),
            "execution_mode": str(keys[1]),
            "edge_threshold": float(keys[2]),
            stratum: str(keys[3]),
            **_pnl_summary(group, bootstrap_samples=bootstrap_samples, seed=seed + index),
        })
    return rows


def _find_execution_summary(
    summaries: list[dict[str, object]], *, stage: str, mode: str, threshold: float,
) -> dict[str, object] | None:
    return next((row for row in summaries if row["stage"] == stage
                 and row["execution_mode"] == mode
                 and np.isclose(float(row["edge_threshold"]), threshold)), None)


def _profit_status(summary: dict[str, object] | None) -> str:
    if summary is None:
        return "not measured"
    if float(summary["mean_event_day_pnl_ci90_low"]) > 0:
        return "supported"
    if float(summary["mean_event_day_pnl_ci90_high"]) < 0:
        return "contradicted"
    return "inconclusive"


def _frozen_profit_status(
    overall: dict[str, object] | None,
    folds: list[dict[str, object]],
) -> str:
    if overall is None or not folds:
        return "not measured"
    positive = sum(float(row["mean_event_day_pnl_usd"]) > 0 for row in folds)
    negative = sum(float(row["mean_event_day_pnl_usd"]) < 0 for row in folds)
    majority = len(folds) // 2 + 1
    if float(overall["mean_event_day_pnl_ci90_low"]) > 0 and positive >= majority:
        return "supported"
    if float(overall["mean_event_day_pnl_ci90_high"]) < 0 and negative >= majority:
        return "contradicted"
    return "inconclusive"


def _paired_mode_summary(
    settlements: pd.DataFrame, *, bootstrap_samples: int, seed: int,
) -> dict[str, object] | None:
    if settlements.empty:
        return None
    primary = settlements[
        (settlements["stage"] == "frozen")
        & np.isclose(settlements["edge_threshold"].astype(float), 0.0)
    ]
    pivot = primary.pivot_table(
        index=["event_id", "target_date"], columns="execution_mode",
        values="realized_pnl_usd", aggfunc="sum",
    ).dropna(subset=["maker", "taker"] if {"maker", "taker"}.issubset(primary["execution_mode"]) else None)
    if not {"maker", "taker"}.issubset(pivot.columns) or pivot.empty:
        return None
    delta = (pivot["maker"] - pivot["taker"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))].mean(axis=1)
    return {
        "common_event_days": int(len(delta)),
        "mean_maker_minus_taker_pnl_usd": float(delta.mean()),
        "ci90_low": float(np.quantile(means, 0.05)),
        "ci90_high": float(np.quantile(means, 0.95)),
        "bootstrap_probability_maker_better": float(np.mean(means > 0.0)),
    }


def _paired_status(summary: dict[str, object] | None) -> str:
    if summary is None:
        return "not measured"
    if float(summary["ci90_low"]) > 0:
        return "supported"
    if float(summary["ci90_high"]) < 0:
        return "contradicted"
    return "inconclusive"


def write_two_bucket_artifact(
    *,
    output: Path | str,
    frames: Mapping[str, pd.DataFrame],
    report: Mapping[str, object],
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Persist one immutable, hash-bound two-bucket decision artifact."""
    expected = {
        "selection_panel", "calibration_folds", "decisions",
        "execution_legs", "event_pnl",
    }
    if set(frames) != expected:
        raise ValueError("two-bucket artifact frame set is incomplete")
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"two-bucket output already exists: {root}")
    root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name in sorted(expected):
        path = root / f"{name}.parquet"
        frames[name].to_parquet(path, index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="ascii",
    )
    hashes[report_path.name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_class": "model_selected_two_bucket_union_study",
        "input_sha256": dict(sorted(input_sha256.items())),
        "sha256": hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
    )
    return manifest
