"""Acceptance checks for immutable historical CLOB artifacts."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from data.clob.history import HistoricalClobDataset, HistoricalDataIncomplete


def audit_historical_clob(
    dataset: HistoricalClobDataset,
    *,
    exclusions: Mapping[str, str],
) -> dict:
    """Accept complete days plus explicitly justified provider gaps."""
    coverage = dataset.coverage()
    incomplete = set(
        coverage.loc[~coverage["complete"], "target_date"].astype(str)
    )
    excluded = {str(day): str(reason) for day, reason in exclusions.items()}
    unexplained = sorted(incomplete - set(excluded))
    if unexplained:
        raise HistoricalDataIncomplete(
            "unexplained incomplete target dates: " + ", ".join(unexplained)
        )
    unnecessary = sorted(set(excluded) - incomplete)
    if unnecessary:
        raise HistoricalDataIncomplete(
            "exclusions do not correspond to incomplete target dates: "
            + ", ".join(unnecessary)
        )
    empty_reasons = sorted(day for day, reason in excluded.items() if not reason)
    if empty_reasons:
        raise HistoricalDataIncomplete(
            "exclusions require a reason: " + ", ".join(empty_reasons)
        )

    trades = dataset.trades
    condition_ids = set(dataset.markets["condition_id"].astype(str))
    known_tokens = set(dataset.markets["yes_token_id"].astype(str)) | set(
        dataset.markets["no_token_id"].astype(str)
    )
    timestamps = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    prices = pd.to_numeric(trades["price"], errors="coerce")
    sizes = pd.to_numeric(trades["size"], errors="coerce")
    quality = {
        "duplicate_trades": int(trades.duplicated().sum()),
        "invalid_prices": int((prices.isna() | ~prices.between(0.0, 1.0)).sum()),
        "nonpositive_sizes": int((sizes.isna() | (sizes <= 0.0)).sum()),
        "null_timestamps": int(timestamps.isna().sum()),
        "unknown_conditions": int(
            (~trades["condition_id"].astype(str).isin(condition_ids)).sum()
        ),
        "unknown_tokens": int(
            (~trades["token_id"].astype(str).isin(known_tokens)).sum()
        ),
    }
    failures = {name: count for name, count in quality.items() if count}
    if failures:
        detail = ", ".join(f"{name}={count}" for name, count in failures.items())
        raise HistoricalDataIncomplete(
            f"historical CLOB quality checks failed: {detail}"
        )

    return {
        "schema_version": 1,
        "status": (
            "accepted_with_exclusions" if excluded else "accepted_complete"
        ),
        "station_id": dataset.station_id,
        "requested_range": [dataset.start_date, dataset.end_date],
        "requested_days": len(coverage),
        "eligible_days": int(coverage["complete"].sum()),
        "excluded_days": dict(sorted(excluded.items())),
        "row_counts": {
            "events": len(dataset.events),
            "markets": len(dataset.markets),
            "trades": len(dataset.trades),
        },
        "quality": quality,
    }
