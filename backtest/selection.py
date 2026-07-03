"""Leakage-safe strategy selection using event-day bootstrap evidence."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from backtest.replay import ReplayResult


def event_day_pnl(
    result: ReplayResult,
    *,
    events: pd.DataFrame,
    markets: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct complete event-day PnL from cashflows and payouts."""
    required_events = {"event_id", "target_date"}
    required_markets = {"condition_id", "event_id"}
    if not required_events.issubset(events):
        raise ValueError("events require event_id and target_date")
    if not required_markets.issubset(markets):
        raise ValueError("markets require condition_id and event_id")
    event_frame = events[["event_id", "target_date"]].copy()
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    event_frame["target_date"] = (
        pd.to_datetime(event_frame["target_date"]).dt.date.astype(str)
    )
    if event_frame["event_id"].duplicated().any():
        raise ValueError("events contain duplicate event_id")
    pnl_by_event = pd.Series(0.0, index=event_frame["event_id"], dtype=float)

    if not result.trades.empty:
        trades = result.trades.merge(
            markets[["condition_id", "event_id"]],
            on="condition_id",
            how="left",
            validate="many_to_one",
        )
        if trades["event_id"].isna().any():
            raise ValueError("trades reference unknown condition_id")
        gross = trades["size_usd"].astype(float)
        fees = trades.get("fee_usd", pd.Series(0.0, index=trades.index))
        trades["cashflow"] = np.where(
            trades["side"].astype(str).str.upper() == "BUY",
            -gross - fees.astype(float),
            gross - fees.astype(float),
        )
        pnl_by_event = pnl_by_event.add(
            trades.groupby(trades["event_id"].astype(str))["cashflow"].sum(),
            fill_value=0.0,
        )
    if not result.settlements.empty:
        payouts = result.settlements.groupby(
            result.settlements["event_id"].astype(str)
        )["payout_usd"].sum()
        pnl_by_event = pnl_by_event.add(payouts, fill_value=0.0)

    unknown_events = set(pnl_by_event.index) - set(event_frame["event_id"])
    if unknown_events:
        raise ValueError(
            f"result references unknown event_id: {sorted(unknown_events)}"
        )
    event_frame["realized_pnl_usd"] = (
        event_frame["event_id"].map(pnl_by_event).fillna(0.0)
    )
    daily = (
        event_frame.rename(columns={"target_date": "event_day"})
        .groupby("event_day", as_index=False)["realized_pnl_usd"]
        .sum()
        .sort_values("event_day")
        .reset_index(drop=True)
    )
    expected = float(result.ending_cash_usd - result.starting_cash_usd)
    actual = float(daily["realized_pnl_usd"].sum())
    if not np.isclose(actual, expected, atol=1e-6):
        raise ValueError(
            f"event-day PnL {actual} does not reconcile to cash PnL {expected}"
        )
    return daily


def select_strategy_candidates(
    results_by_strategy: dict[str, pd.DataFrame],
    *,
    min_event_days: int = 10,
    confidence: float = 0.90,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
    max_candidates: int = 2,
) -> pd.DataFrame:
    """Rank strategies by a positive event-day mean-PnL lower bound."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if bootstrap_samples <= 0 or min_event_days <= 0:
        raise ValueError("sample and event-day counts must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    rows: list[dict] = []
    alpha = (1.0 - confidence) / 2.0
    lcb_name = f"pnl_mean_lcb_{int(round(confidence * 100))}"
    ucb_name = f"pnl_mean_ucb_{int(round(confidence * 100))}"
    for strategy, frame in results_by_strategy.items():
        missing = {"event_day", "realized_pnl_usd"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{strategy} results missing columns: {sorted(missing)}"
            )
        daily = (
            frame.assign(
                event_day=pd.to_datetime(frame["event_day"]).dt.date
            )
            .groupby("event_day", sort=True)["realized_pnl_usd"]
            .sum()
            .to_numpy(dtype=float)
        )
        strategy_seed = seed + int(
            hashlib.sha256(strategy.encode("utf-8")).hexdigest()[:8],
            16,
        )
        if len(daily):
            rng = np.random.default_rng(strategy_seed)
            indices = rng.integers(
                0,
                len(daily),
                size=(bootstrap_samples, len(daily)),
            )
            means = daily[indices].mean(axis=1)
            lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
            mean = float(daily.mean())
            total = float(daily.sum())
        else:
            lower = upper = mean = total = float("nan")
        enough_days = len(daily) >= min_event_days
        positive_lcb = bool(np.isfinite(lower) and lower > 0.0)
        rows.append(
            {
                "strategy": strategy,
                "event_days": len(daily),
                "total_pnl_usd": total,
                "mean_event_day_pnl_usd": mean,
                lcb_name: float(lower),
                ucb_name: float(upper),
                "eligible": enough_days and positive_lcb,
                "gate_reason": (
                    None
                    if enough_days and positive_lcb
                    else (
                        "insufficient_event_days"
                        if not enough_days
                        else "non_positive_lower_bound"
                    )
                ),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table.assign(selected=pd.Series(dtype=bool))
    table = table.sort_values(
        ["eligible", lcb_name, "total_pnl_usd", "strategy"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table["selected"] = False
    eligible_indices = table.index[table["eligible"]][:max_candidates]
    table.loc[eligible_indices, "selected"] = True
    return table
