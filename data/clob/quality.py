"""Acceptance checks for immutable historical CLOB artifacts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.clob.history import HistoricalClobDataset, HistoricalDataIncomplete


@dataclass(frozen=True)
class ClobAuditReport:
    """Display-ready semantic audit of historical CLOB evidence."""

    passed: bool
    failures: tuple[str, ...]
    summary: dict
    daily_coverage: pd.DataFrame
    event_checks: pd.DataFrame
    checks: pd.DataFrame
    quote_checks: pd.DataFrame


def build_clob_audit_report(
    dataset: HistoricalClobDataset,
    *,
    exclusions: Mapping[str, str] | None = None,
    analysis_exclusions: Mapping[str, str] | None = None,
    quotes: pd.DataFrame | None = None,
    actual_highs: pd.DataFrame | None = None,
    clob_manifest_sha256: str | None = None,
    quote_source_sha256: str | None = None,
    quote_max_staleness: str | pd.Timedelta | None = None,
    expected_row_counts: Mapping[str, int] | None = None,
) -> ClobAuditReport:
    """Return non-raising semantic evidence suitable for an audit UI."""
    excluded = {str(day): str(reason) for day, reason in (exclusions or {}).items()}
    analysis_excluded = {
        str(day): str(reason)
        for day, reason in (analysis_exclusions or {}).items()
    }
    coverage = dataset.coverage()
    coverage_dates = coverage["target_date"].astype(str)
    coverage["exclusion_reason"] = coverage_dates.map(
        analysis_excluded
    ).combine_first(
        coverage_dates.map(excluded)
    )
    coverage["audit_status"] = np.where(
        coverage_dates.isin(analysis_excluded),
        "EXCLUDED",
        np.where(
            coverage["complete"],
            "ELIGIBLE",
            np.where(
                coverage["exclusion_reason"].notna(),
                "EXCLUDED",
                "FAIL",
            ),
        ),
    )
    incomplete = set(
        coverage.loc[~coverage["complete"], "target_date"].astype(str)
    )
    unexplained = incomplete - set(excluded)
    unnecessary = set(excluded) - incomplete
    unknown_analysis_exclusions = set(analysis_excluded) - set(coverage_dates)
    check_rows: list[dict] = []
    failures: list[str] = []

    def add_check(
        name: str,
        failure_count: int,
        detail: str = "",
        *,
        status: str | None = None,
    ) -> None:
        resolved = status or ("FAIL" if failure_count else "PASS")
        check_rows.append(
            {
                "check": name,
                "status": resolved,
                "failure_count": int(failure_count),
                "detail": detail,
            }
        )
        if resolved == "FAIL":
            failures.append(name)

    coverage_failures = (
        len(unexplained)
        + len(unnecessary)
        + sum(not reason for reason in excluded.values())
        + len(unknown_analysis_exclusions)
        + sum(not reason for reason in analysis_excluded.values())
    )
    coverage_detail = "; ".join(
        part
        for part in [
            (
                "unexplained: " + ", ".join(sorted(unexplained))
                if unexplained
                else ""
            ),
            (
                "unnecessary exclusions: " + ", ".join(sorted(unnecessary))
                if unnecessary
                else ""
            ),
            (
                "unknown analysis exclusions: "
                + ", ".join(sorted(unknown_analysis_exclusions))
                if unknown_analysis_exclusions
                else ""
            ),
        ]
        if part
    )
    add_check("coverage", coverage_failures, coverage_detail)

    actual_row_counts = {
        "events": len(dataset.events),
        "markets": len(dataset.markets),
        "trades": len(dataset.trades),
    }
    row_count_failures = sum(
        int(actual_row_counts.get(name) != int(expected))
        for name, expected in (expected_row_counts or {}).items()
        if name in actual_row_counts
    )
    add_check("manifest_row_counts", row_count_failures)

    event_ids = dataset.events["event_id"].astype(str)
    condition_ids = dataset.markets["condition_id"].astype(str)
    yes_tokens = dataset.markets["yes_token_id"].astype(str)
    no_tokens = dataset.markets["no_token_id"].astype(str)
    market_event_ids = dataset.markets["event_id"].astype(str)
    identity_failures = (
        int(event_ids.duplicated().sum())
        + int(condition_ids.duplicated().sum())
        + int((~market_event_ids.isin(set(event_ids))).sum())
        + int(pd.concat([yes_tokens, no_tokens]).duplicated().sum())
    )
    add_check("event_market_token_identity", identity_failures)

    event_rows: list[dict] = []
    bad_partitions = 0
    for event_id, markets in dataset.markets.groupby("event_id", sort=True):
        ordered = markets.sort_values(["bucket_lo_f", "bucket_hi_f"])
        bounds = list(
            zip(
                ordered["bucket_lo_f"].astype(float),
                ordered["bucket_hi_f"].astype(float),
            )
        )
        issues: list[str] = []
        if bounds and not np.isneginf(bounds[0][0]):
            issues.append("missing -inf lower bound")
        if bounds and not np.isposinf(bounds[-1][1]):
            issues.append("missing +inf upper bound")
        for (_lo, left_hi), (right_lo, _hi) in zip(bounds, bounds[1:]):
            if left_hi < right_lo:
                issues.append(f"gap [{left_hi}, {right_lo})")
            elif left_hi > right_lo:
                issues.append(f"overlap [{right_lo}, {left_hi})")
        if issues:
            bad_partitions += 1
        target_dates = dataset.events.loc[
            dataset.events["event_id"].astype(str) == str(event_id),
            "target_date",
        ]
        target_date = str(target_dates.iloc[0]) if len(target_dates) else None
        event_rows.append(
            {
                "event_id": str(event_id),
                "target_date": target_date,
                "market_count": len(ordered),
                "bucket_partition_status": "FAIL" if issues else "PASS",
                "bucket_issue": "; ".join(issues),
                "actual_high_f": None,
                "winning_bucket_count": None,
                "resolution_status": "NOT_AVAILABLE",
                "resolved_reconciliation_status": "NOT_AVAILABLE",
            }
        )
    add_check("bucket_partitions", bad_partitions)
    event_checks = pd.DataFrame(event_rows)

    trades = dataset.trades
    trade_timestamps = pd.to_datetime(
        trades["timestamp"], utc=True, errors="coerce"
    )
    prices = pd.to_numeric(trades["price"], errors="coerce")
    sizes = pd.to_numeric(trades["size"], errors="coerce")
    usdcs = pd.to_numeric(
        trades.get("usdc", prices * sizes),
        errors="coerce",
    )
    known_tokens = set(yes_tokens) | set(no_tokens)
    token_condition = {
        str(token): str(condition)
        for condition, yes_token, no_token in zip(
            condition_ids,
            yes_tokens,
            no_tokens,
        )
        for token in (yes_token, no_token)
    }
    trade_conditions = trades["condition_id"].astype(str)
    trade_tokens = trades["token_id"].astype(str)
    wrong_market_tokens = sum(
        token_condition.get(token) != condition
        for condition, token in zip(trade_conditions, trade_tokens)
    )
    trade_failures = (
        int(trades.duplicated().sum())
        + int((prices.isna() | ~prices.between(0.0, 1.0)).sum())
        + int((sizes.isna() | (sizes <= 0.0)).sum())
        + int((usdcs.isna() | (usdcs < 0.0)).sum())
        + int(trade_timestamps.isna().sum())
        + int((~trades["condition_id"].astype(str).isin(set(condition_ids))).sum())
        + int((~trades["token_id"].astype(str).isin(known_tokens)).sum())
        + int(wrong_market_tokens)
        + int((~np.isclose(usdcs, prices * sizes, atol=1e-9)).sum())
    )
    add_check("trades", trade_failures)
    page_cap_markets = int(
        trades.groupby("condition_id", sort=False).size().eq(1_000).sum()
    )
    suspected_truncated_markets = (
        page_cap_markets if page_cap_markets >= 2 else 0
    )
    add_check(
        "trade_history_truncation",
        suspected_truncated_markets,
        (
            f"{page_cap_markets} markets contain exactly 1,000 trades, "
            "matching the Data API page cap"
            if suspected_truncated_markets
            else ""
        ),
    )

    actual_days = 0
    if actual_highs is not None:
        required_actual = {"target_date", "actual_high_f"}
        if not required_actual.issubset(actual_highs.columns):
            add_check(
                "actual_bucket_resolution",
                len(required_actual - set(actual_highs.columns)),
                "actuals require target_date and actual_high_f",
            )
        else:
            actual_map = {
                pd.Timestamp(row.target_date).date().isoformat(): float(
                    row.actual_high_f
                )
                for row in actual_highs.itertuples(index=False)
            }
            resolution_failures = 0
            resolved_reconciliation_failures = 0
            resolved_reconciliation_events = 0
            for index, row in event_checks.iterrows():
                target_date = str(row["target_date"])
                if target_date not in actual_map:
                    continue
                actual_days += 1
                actual = actual_map[target_date]
                event_markets = dataset.markets[
                    dataset.markets["event_id"].astype(str)
                    == str(row["event_id"])
                ]
                winners = int(
                    (
                        (actual >= event_markets["bucket_lo_f"].astype(float))
                        & (actual < event_markets["bucket_hi_f"].astype(float))
                    ).sum()
                )
                actual_winner = (
                    (actual >= event_markets["bucket_lo_f"].astype(float))
                    & (actual < event_markets["bucket_hi_f"].astype(float))
                )
                resolution_failures += int(winners != 1)
                event_checks.loc[index, "actual_high_f"] = actual
                event_checks.loc[index, "winning_bucket_count"] = winners
                event_checks.loc[index, "resolution_status"] = (
                    "PASS" if winners == 1 else "FAIL"
                )
                if "resolved_yes" in event_markets.columns:
                    resolved = event_markets["resolved_yes"]
                    if resolved.notna().any():
                        resolved_reconciliation_events += 1
                        complete_resolution = bool(resolved.notna().all())
                        resolved_winner = resolved.fillna(False).astype(bool)
                        agrees = bool(
                            complete_resolution
                            and int(resolved_winner.sum()) == 1
                            and actual_winner.reset_index(drop=True).equals(
                                resolved_winner.reset_index(drop=True)
                            )
                        )
                        resolved_reconciliation_failures += int(not agrees)
                        event_checks.loc[
                            index, "resolved_reconciliation_status"
                        ] = "PASS" if agrees else "FAIL"
            add_check("actual_bucket_resolution", resolution_failures)
            add_check(
                "resolved_outcome_reconciliation",
                resolved_reconciliation_failures,
                f"checked_events={resolved_reconciliation_events}",
                status=(
                    None if resolved_reconciliation_events else "NOT_MEASURED"
                ),
            )

    quote_checks = pd.DataFrame()
    quote_count = 0
    if quotes is not None:
        quote_count = len(quotes)
        quote_failure_rows: list[dict] = []
        required_quotes = {
            "condition_id",
            "timestamp",
            "bid_price_yes",
            "mid_price_yes",
            "ask_price_yes",
            "estimated_spread",
            "impact_per_usdc",
            "capacity_usdc",
            "rolling_volume_usdc",
            "last_trade_at",
            "staleness_seconds",
        }
        missing_quotes = required_quotes - set(quotes.columns)
        if missing_quotes:
            quote_failure_rows.append(
                {
                    "check": "required_columns",
                    "status": "FAIL",
                    "failure_count": len(missing_quotes),
                    "detail": ", ".join(sorted(missing_quotes)),
                }
            )
        else:
            bid = pd.to_numeric(quotes["bid_price_yes"], errors="coerce")
            mid = pd.to_numeric(quotes["mid_price_yes"], errors="coerce")
            ask = pd.to_numeric(quotes["ask_price_yes"], errors="coerce")
            spread = pd.to_numeric(
                quotes["estimated_spread"], errors="coerce"
            )
            impact = pd.to_numeric(
                quotes["impact_per_usdc"], errors="coerce"
            )
            capacity = pd.to_numeric(
                quotes["capacity_usdc"], errors="coerce"
            )
            volume = pd.to_numeric(
                quotes["rolling_volume_usdc"], errors="coerce"
            )
            timestamp = pd.to_datetime(
                quotes["timestamp"], utc=True, errors="coerce"
            )
            last_trade = pd.to_datetime(
                quotes["last_trade_at"], utc=True, errors="coerce"
            )
            stale = pd.to_numeric(
                quotes["staleness_seconds"], errors="coerce"
            )
            observed_stale = (timestamp - last_trade).dt.total_seconds()
            max_stale = (
                pd.Timedelta(quote_max_staleness).total_seconds()
                if quote_max_staleness is not None
                else None
            )
            quote_rules = {
                "known_conditions": int(
                    (
                        ~quotes["condition_id"]
                        .astype(str)
                        .isin(set(condition_ids))
                    ).sum()
                ),
                "price_order": int(
                    (
                        bid.isna()
                        | mid.isna()
                        | ask.isna()
                        | (bid < 0.0)
                        | (ask > 1.0)
                        | (bid > mid)
                        | (mid > ask)
                    ).sum()
                ),
                "spread": int(
                    (
                        spread.isna()
                        | ~np.isclose(spread, ask - bid, atol=1e-9)
                    ).sum()
                ),
                "capacity_and_impact": int(
                    (
                        impact.isna()
                        | capacity.isna()
                        | volume.isna()
                        | (impact < 0.0)
                        | (capacity < 0.0)
                        | (volume < 0.0)
                    ).sum()
                ),
                "past_only_timestamps": int(
                    (
                        timestamp.isna()
                        | last_trade.isna()
                        | (last_trade > timestamp)
                    ).sum()
                ),
                "staleness": int(
                    (
                        stale.isna()
                        | ~np.isclose(stale, observed_stale, atol=1e-9)
                        | (
                            (stale > max_stale)
                            if max_stale is not None
                            else False
                        )
                    ).sum()
                ),
                "source_manifest": int(
                    clob_manifest_sha256 is not None
                    and quote_source_sha256 != clob_manifest_sha256
                ),
            }
            quote_failure_rows.extend(
                {
                    "check": name,
                    "status": "FAIL" if count else "PASS",
                    "failure_count": count,
                    "detail": "",
                }
                for name, count in quote_rules.items()
            )
        quote_checks = pd.DataFrame(quote_failure_rows)
        quote_failures = int(
            quote_checks.loc[
                quote_checks["status"] == "FAIL", "failure_count"
            ].sum()
        )
        add_check("synthetic_quotes", quote_failures)

    quality_counts = (
        quotes["quote_quality"].value_counts().to_dict()
        if quotes is not None and "quote_quality" in quotes
        else {}
    )
    return ClobAuditReport(
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
        summary={
            **actual_row_counts,
            "quotes": quote_count,
            "actual_days": actual_days,
            "requested_days": len(coverage),
            "eligible_days": int(coverage["audit_status"].eq("ELIGIBLE").sum()),
            "excluded_days": int(coverage["audit_status"].eq("EXCLUDED").sum()),
            "trade_usdc": float(usdcs.sum()),
            "suspected_truncated_markets": suspected_truncated_markets,
            "quote_quality": quality_counts,
        },
        daily_coverage=coverage,
        event_checks=event_checks,
        checks=pd.DataFrame(check_rows),
        quote_checks=quote_checks,
    )


def audit_historical_clob(
    dataset: HistoricalClobDataset,
    *,
    exclusions: Mapping[str, str],
) -> dict:
    """Accept complete days plus explicitly justified provider gaps."""
    semantic_market_columns = {
        "event_id",
        "condition_id",
        "yes_token_id",
        "no_token_id",
        "bucket_lo_f",
        "bucket_hi_f",
    }
    semantic_trade_columns = {
        "condition_id",
        "token_id",
        "timestamp",
        "price",
        "size",
        "usdc",
    }
    if semantic_market_columns.issubset(
        dataset.markets.columns
    ) and semantic_trade_columns.issubset(dataset.trades.columns):
        semantic = build_clob_audit_report(
            dataset,
            exclusions=exclusions,
        )
        if not semantic.passed:
            raise HistoricalDataIncomplete(
                "historical CLOB semantic checks failed: "
                + ", ".join(semantic.failures)
            )
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
