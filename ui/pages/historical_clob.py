"""Read-only historical CLOB artifact audit and exploration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest.forecast_cache import ForecastCache
from backtest.synthetic_quotes import (
    SyntheticQuoteDataset,
    build_synthetic_book,
)
from config.settings import Settings
from config.stations import STATIONS
from core.pricing.quote import compute_quote
from data.clob.history import HistoricalClobDataset
from data.clob.quality import ClobAuditReport, build_clob_audit_report


@dataclass(frozen=True)
class HistoricalClobArtifact:
    """One locally persisted CLOB artifact plus its fresh audit evidence."""

    name: str
    station_id: str
    root: Path
    fetched_at: str
    manifest_sha256: str
    dataset: HistoricalClobDataset | None
    audit: ClobAuditReport
    quotes_root: Path | None = None
    quotes: SyntheticQuoteDataset | None = None
    forecast_roots: tuple[Path, ...] = ()
    forecasts: tuple[ForecastCache, ...] = ()
    forecast_binding: str = "none"
    load_error: str = ""


def discover_historical_clob_artifacts(
    data_store_dir: Path | str,
) -> list[HistoricalClobArtifact]:
    """Load and audit every local historical CLOB artifact."""
    store = Path(data_store_dir)
    clob_market_sha_by_manifest: dict[str, str] = {}
    for clob_manifest_path in (
        store / "backtest" / "clob"
    ).glob("*/*/manifest.json"):
        try:
            manifest_bytes = clob_manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            market_sha = str(
                manifest.get("sha256", {}).get("markets.parquet", "")
            )
            if market_sha:
                clob_market_sha_by_manifest[
                    hashlib.sha256(manifest_bytes).hexdigest()
                ] = market_sha
        except Exception:
            continue
    forecasts_by_source: dict[
        str, list[tuple[str, Path, ForecastCache]]
    ] = {}
    for forecast_manifest_path in (
        store / "backtest" / "forecasts"
    ).glob("*/*/manifest.json"):
        try:
            forecast_manifest = json.loads(
                forecast_manifest_path.read_text("utf-8")
            )
            source = str(
                forecast_manifest.get("source_sha256", {}).get(
                    "clob_manifest", ""
                )
            )
            if not source:
                continue
            forecasts_by_source.setdefault(source, []).append(
                (
                    str(forecast_manifest.get("created_at", "")),
                    forecast_manifest_path.parent,
                    ForecastCache.load(forecast_manifest_path.parent),
                )
            )
        except Exception:
            continue
    quote_by_source: dict[str, tuple[str, Path, dict]] = {}
    for quote_manifest_path in (
        store / "backtest" / "quotes"
    ).glob("*/*/manifest.json"):
        quote_manifest = json.loads(quote_manifest_path.read_text("utf-8"))
        source = str(quote_manifest.get("source_sha256", ""))
        created_at = str(quote_manifest.get("created_at", ""))
        candidate = (created_at, quote_manifest_path.parent, quote_manifest)
        if source and (
            source not in quote_by_source
            or created_at > quote_by_source[source][0]
        ):
            quote_by_source[source] = candidate
    actuals_by_station: dict[str, pd.DataFrame] = {}
    for station_dir in (store / "backtest" / "actuals").glob("*"):
        frames: list[pd.DataFrame] = []
        for path in sorted(station_dir.glob("*.csv")):
            frames.append(pd.read_csv(path))
        for path in sorted(station_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(path))
        if frames:
            actuals_by_station[station_dir.name] = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["target_date"], keep="last")
                .reset_index(drop=True)
            )
    artifacts: list[HistoricalClobArtifact] = []
    for manifest_path in sorted(
        (store / "backtest" / "clob").glob("*/*/manifest.json")
    ):
        root = manifest_path.parent
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            dataset = HistoricalClobDataset.load(root)
        except Exception as exc:
            artifacts.append(
                HistoricalClobArtifact(
                    name=root.name,
                    station_id=str(
                        manifest.get(
                            "station_id", manifest_path.parents[1].name
                        )
                    ),
                    root=root,
                    fetched_at=str(manifest.get("fetched_at", "")),
                    manifest_sha256=digest,
                    dataset=None,
                    audit=_integrity_failure_report(str(exc)),
                    load_error=str(exc),
                )
            )
            continue
        quality_path = root.with_suffix(".quality.json")
        exclusions: dict[str, str] = {}
        if quality_path.exists():
            quality = json.loads(quality_path.read_text("utf-8"))
            quality_source = quality.get("source_manifest_sha256")
            if quality_source in {None, digest}:
                exclusions = {
                    str(day): str(reason)
                    for day, reason in quality.get(
                        "excluded_days", {}
                    ).items()
                }
        quote_match = quote_by_source.get(digest)
        quotes_root = quote_match[1] if quote_match is not None else None
        quotes = (
            SyntheticQuoteDataset.load(quotes_root)
            if quotes_root is not None
            else None
        )
        quote_manifest = quote_match[2] if quote_match is not None else {}
        forecast_binding = "exact"
        forecast_matches = forecasts_by_source.get(digest, [])
        if not forecast_matches:
            forecast_binding = "market_compatible"
            selected_market_sha = clob_market_sha_by_manifest.get(digest)
            forecast_matches = [
                match
                for source_digest, matches in forecasts_by_source.items()
                if selected_market_sha
                and clob_market_sha_by_manifest.get(source_digest)
                == selected_market_sha
                for match in matches
            ]
        forecast_matches = sorted(
            forecast_matches,
            key=lambda match: (match[0], str(match[1])),
        )
        if not forecast_matches:
            forecast_binding = "none"
        audit = build_clob_audit_report(
            dataset,
            exclusions=exclusions,
            quotes=quotes.quotes if quotes is not None else None,
            actual_highs=actuals_by_station.get(dataset.station_id),
            clob_manifest_sha256=digest,
            quote_source_sha256=(
                str(quote_manifest.get("source_sha256"))
                if quote_match is not None
                else None
            ),
            quote_max_staleness=(
                quotes.max_staleness if quotes is not None else None
            ),
            expected_row_counts=manifest.get("row_counts", {}),
        )
        artifacts.append(
            HistoricalClobArtifact(
                name=root.name,
                station_id=str(manifest["station_id"]),
                root=root,
                fetched_at=str(manifest.get("fetched_at", "")),
                manifest_sha256=digest,
                dataset=dataset,
                audit=audit,
                quotes_root=quotes_root,
                quotes=quotes,
                forecast_roots=tuple(
                    match[1] for match in forecast_matches
                ),
                forecasts=tuple(match[2] for match in forecast_matches),
                forecast_binding=forecast_binding,
            )
        )
    return sorted(
        artifacts,
        key=lambda artifact: (artifact.fetched_at, artifact.name),
        reverse=True,
    )


def default_artifact_index(
    artifacts: list[HistoricalClobArtifact],
) -> int:
    """Select the newest model-ready passing artifact when available."""
    for index, artifact in enumerate(artifacts):
        if (
            artifact.audit.passed
            and artifact.quotes is not None
            and artifact.forecasts
        ):
            return index
    for index, artifact in enumerate(artifacts):
        if artifact.audit.passed:
            return index
    return 0


def _integrity_failure_report(error: str) -> ClobAuditReport:
    return ClobAuditReport(
        passed=False,
        failures=("artifact_integrity",),
        summary={
            "events": 0,
            "markets": 0,
            "trades": 0,
            "quotes": 0,
            "actual_days": 0,
            "requested_days": 0,
            "eligible_days": 0,
            "excluded_days": 0,
            "trade_usdc": 0.0,
            "quote_quality": {},
        },
        daily_coverage=pd.DataFrame(),
        event_checks=pd.DataFrame(),
        checks=pd.DataFrame(
            [
                {
                    "check": "artifact_integrity",
                    "status": "FAIL",
                    "failure_count": 1,
                    "detail": error,
                }
            ]
        ),
        quote_checks=pd.DataFrame(),
    )


def render(data_store_dir: Path | str | None = None) -> None:
    """Render the read-only historical CLOB audit explorer."""
    root = Path(data_store_dir or Settings().data_store_dir)
    st.header("Historical CLOB")
    st.caption(
        "Read-only point-in-time evidence audit. Synthetic quotes are "
        "estimated from public trades and are not historical L2."
    )
    artifacts = _load_catalog(str(root), _catalog_fingerprint(root))
    if not artifacts:
        st.warning(f"No historical CLOB artifacts found under {root}.")
        return
    stations = sorted({artifact.station_id for artifact in artifacts})
    station = st.selectbox("Station", stations, index=0)
    choices = [
        artifact for artifact in artifacts if artifact.station_id == station
    ]
    labels = [
        (
            f"{'PASS' if artifact.audit.passed else 'FAIL'} · "
            f"{'MODEL · ' if artifact.forecasts else ''}"
            f"{artifact.name} · "
            f"{_format_local_time(artifact.fetched_at, station)}"
        )
        for artifact in choices
    ]
    selected_label = st.selectbox(
        "Artifact",
        labels,
        index=default_artifact_index(choices),
    )
    artifact = choices[labels.index(selected_label)]
    if artifact.audit.passed:
        st.success("Fresh semantic audit PASS")
    else:
        st.error(
            "Fresh semantic audit FAIL · "
            + ", ".join(artifact.audit.failures)
        )
    suspected_truncation = int(
        artifact.audit.summary.get("suspected_truncated_markets", 0)
    )
    if suspected_truncation:
        st.warning(
            f"{suspected_truncation} markets contain exactly 1,000 trades. "
            "This artifact was truncated by the historical Data API page cap "
            "and must be downloaded again before backtesting."
        )
    if artifact.dataset is None:
        st.dataframe(artifact.audit.checks, width="stretch")
        return

    tabs = st.tabs(["Trades", "Synthetic Book"])
    with tabs[0]:
        _render_trades(artifact)
    with tabs[1]:
        _render_synthetic_book(artifact)


def _catalog_fingerprint(root: Path) -> str:
    hasher = hashlib.sha256()
    patterns = [
        "backtest/clob/*/*/manifest.json",
        "backtest/clob/*/*.quality.json",
        "backtest/quotes/*/*/manifest.json",
        "backtest/forecasts/*/*/manifest.json",
        "backtest/actuals/*/*.manifest.json",
    ]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            hasher.update(str(path.relative_to(root)).encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


@st.cache_data(show_spinner="Loading and auditing historical CLOB artifacts…")
def _load_catalog(
    data_store_dir: str,
    fingerprint: str,
) -> list[HistoricalClobArtifact]:
    del fingerprint
    return discover_historical_clob_artifacts(data_store_dir)


def _render_trades(artifact: HistoricalClobArtifact) -> None:
    condition_id, target_date = _select_market_by_date(
        artifact,
        key_prefix="historical_clob_trades",
    )
    if condition_id is None:
        return
    trades = artifact.dataset.trades[
        artifact.dataset.trades["condition_id"].astype(str) == condition_id
    ].copy()
    if trades.empty:
        st.info("No public trades for this Market.")
        return
    timezone = _station_timezone(artifact.station_id)
    trades["timestamp"] = _to_local(trades["timestamp"], timezone)
    trades["outcome"] = trades["outcome"].astype(str).str.upper()
    trades["side"] = trades["side"].astype(str).str.upper()
    actual_high = _actual_high_for_date(artifact, target_date)
    metrics = st.columns(7)
    metrics[0].metric(
        "Event volume",
        f"${event_total_volume(artifact, target_date=target_date):,.0f}",
        help=(
            "Sum of public trade size across every bucket, matching the "
            "volume convention displayed by Polymarket."
        ),
    )
    metrics[1].metric("Bucket trades", len(trades))
    metrics[2].metric("Bucket notional", f"${trades['usdc'].sum():,.2f}")
    metrics[3].metric("YES VWAP", _outcome_vwap(trades, "YES"))
    metrics[4].metric("NO VWAP", _outcome_vwap(trades, "NO"))
    metrics[5].metric(
        "Actual high",
        f"{actual_high:g}°F" if actual_high is not None else "N/A",
    )
    metrics[6].metric(
        "Selected bucket",
        _selected_bucket_result(artifact, condition_id, actual_high),
    )
    st.caption(
        "Full activity window: "
        f"{trades['timestamp'].min():%Y-%m-%d %H:%M %Z} – "
        f"{trades['timestamp'].max():%Y-%m-%d %H:%M %Z}"
    )

    for outcome in ("YES", "NO"):
        outcome_trades = trades[trades["outcome"] == outcome]
        st.subheader(f"{outcome} trades")
        if outcome_trades.empty:
            st.info(f"No {outcome} trades.")
        else:
            st.plotly_chart(
                _trade_price_figure(
                    outcome_trades,
                    outcome=outcome,
                    timezone=timezone,
                ),
                width="stretch",
            )
    volume = (
        trades.set_index("timestamp")
        .groupby("outcome")["usdc"]
        .resample("15min")
        .sum()
        .reset_index()
    )
    volume_figure = px.bar(
        volume,
        x="timestamp",
        y="usdc",
        color="outcome",
        barmode="stack",
        labels={
            "timestamp": f"Local time ({timezone})",
            "usdc": "Volume (USDC)",
            "outcome": "Outcome",
        },
    )
    volume_figure.update_layout(legend=_top_legend())
    st.plotly_chart(
        volume_figure,
        width="stretch",
    )


def _render_synthetic_book(artifact: HistoricalClobArtifact) -> None:
    if artifact.quotes is None:
        st.warning(
            "No synthetic quote artifact matches the selected CLOB artifact."
        )
        return
    st.warning(
        "Synthetic quotes are past-only estimates derived from public trades; "
        "they are not historical CLOB L2 snapshots."
    )
    condition_id, target_date = _select_market_by_date(
        artifact,
        key_prefix="historical_clob_synthetic",
    )
    if condition_id is None:
        return
    quotes = artifact.quotes.quotes[
        artifact.quotes.quotes["condition_id"].astype(str) == condition_id
    ].copy()
    if quotes.empty:
        st.info("No synthetic quotes for this Market.")
        return
    timezone = _station_timezone(artifact.station_id)
    quotes["timestamp"] = _to_local(quotes["timestamp"], timezone)
    trades = artifact.dataset.trades[
        artifact.dataset.trades["condition_id"].astype(str) == condition_id
    ].copy()
    if not trades.empty:
        trades["timestamp"] = _to_local(trades["timestamp"], timezone)
        trades["outcome"] = trades["outcome"].astype(str).str.upper()
    comparison = _bucket_comparison_frame(
        artifact,
        target_date=target_date,
        condition_id=condition_id,
        timezone=timezone,
    )
    model_probabilities = _model_probability_frame(
        artifact,
        target_date=target_date,
        condition_id=condition_id,
        timezone=timezone,
    )
    selected_model = model_probabilities[
        model_probabilities.get(
            "condition_id", pd.Series(dtype=str)
        ).astype(str)
        == str(condition_id)
    ].copy()
    theoretical_execution = pd.concat(
        [
            _theoretical_buy_price_frame(
                quotes,
                order_usdc=order_usdc,
            )
            for order_usdc in (1.0, 5.0, 10.0)
        ],
        ignore_index=True,
    )
    x_axis = _shared_synthetic_x_axis(
        quotes,
        trades,
        comparison,
        model_probabilities,
    )

    actual_high = _actual_high_for_date(artifact, target_date)
    metrics = st.columns(6)
    metrics[0].metric(
        "Actual high",
        f"{actual_high:g}°F" if actual_high is not None else "N/A",
    )
    metrics[1].metric(
        "Selected bucket",
        _selected_bucket_result(artifact, condition_id, actual_high),
    )
    metrics[2].metric(
        "Median spread", f"{quotes['estimated_spread'].median():.3f}"
    )
    metrics[3].metric(
        "Median capacity", f"${quotes['capacity_usdc'].median():,.0f}"
    )
    metrics[4].metric(
        "$50 median impact",
        f"{_price_impact_for_order(quotes, 50.0).median():.2f}¢",
    )
    metrics[5].metric(
        "Median staleness",
        f"{quotes['staleness_seconds'].median() / 60:.1f} min",
    )

    st.subheader("Bucket comparison")
    st.caption(
        "The selected Bucket is shown as one pair: synthetic CLOB midpoint "
        "and point-in-time model fair probability. Model points are placed "
        "at available_at (the first tradable time); hover shows init_time."
    )
    if artifact.forecast_binding == "market_compatible":
        st.info(
            "Forecast overlay is UI-compatible because its source CLOB has "
            "the identical markets.parquet payload (same Events, buckets, "
            "and condition IDs). Quotes and trades still come only from the "
            "selected artifact; managed Backtest Studies continue to require "
            "an exact CLOB manifest match."
        )
    if model_probabilities.empty:
        st.warning(
            "No point-in-time ForecastCache has an exact or market-compatible "
            "CLOB binding, so model probabilities cannot be overlaid."
        )
    if comparison.empty and model_probabilities.empty:
        st.info("No same-day bucket evidence is available for comparison.")
    else:
        for outcome in ("YES", "NO"):
            st.subheader(f"{outcome} bucket comparison")
            st.plotly_chart(
                _bucket_comparison_figure(
                    comparison,
                    model_probabilities=model_probabilities,
                    outcome=outcome,
                    timezone=timezone,
                    x_axis=x_axis,
                ),
                width="stretch",
            )

    for outcome in ("YES", "NO"):
        st.subheader(f"{outcome} synthetic book")
        st.caption(
            "The orange lines are hypothetical $1 / $5 / $10 BUY VWAPs: "
            "each starts at the ask (midpoint + half-spread), walks five "
            "impact levels, and deliberately ignores observed capacity. "
            "Fees are not included."
        )
        st.plotly_chart(
            _synthetic_price_figure(
                quotes,
                trades,
                outcome=outcome,
                timezone=timezone,
                model_probabilities=selected_model,
                theoretical_execution=theoretical_execution,
                x_axis=x_axis,
            ),
            width="stretch",
        )

    st.subheader("$50 executable prices")
    st.caption(
        "VWAP after walking the synthetic five-level book. BUY consumes asks; "
        "SELL consumes bids and requires an existing position—no shorting. "
        "Gaps mean the estimated capacity could not fill $50."
    )
    execution = _execution_price_frame(quotes, order_usdc=50.0)
    for outcome in ("YES", "NO"):
        st.subheader(f"{outcome} executable price")
        st.plotly_chart(
            _execution_price_figure(
                execution,
                outcome=outcome,
                timezone=timezone,
                order_usdc=50.0,
                x_axis=x_axis,
            ),
            width="stretch",
        )

    st.subheader("Executable liquidity")
    st.caption(
        "Capacity is the estimated executable notional. Impact estimates the "
        "price movement for a $50 order; lower is better."
    )
    st.plotly_chart(
        _liquidity_figure(quotes, timezone=timezone, x_axis=x_axis),
        width="stretch",
    )


def _select_market_by_date(
    artifact: HistoricalClobArtifact,
    *,
    key_prefix: str,
) -> tuple[str | None, str | None]:
    events = artifact.dataset.events.copy()
    if events.empty:
        st.info("No Events in this artifact.")
        return None, None
    target_dates = sorted(events["target_date"].astype(str).unique())
    target_date = st.selectbox(
        "Target date",
        target_dates,
        key=f"{key_prefix}_target_date",
    )
    event_ids = set(
        events.loc[
            events["target_date"].astype(str) == target_date,
            "event_id",
        ].astype(str)
    )
    markets = artifact.dataset.markets[
        artifact.dataset.markets["event_id"].astype(str).isin(event_ids)
    ].sort_values(["bucket_lo_f", "bucket_hi_f"])
    if markets.empty:
        st.info("No Markets for this target date.")
        return None, target_date
    label_to_condition: dict[str, str] = {}
    for row in markets.itertuples(index=False):
        label = _bucket_label(row.bucket_lo_f, row.bucket_hi_f)
        if label not in label_to_condition:
            label_to_condition[label] = str(row.condition_id)
    label = st.selectbox(
        "Market / Bucket",
        list(label_to_condition),
        key=f"{key_prefix}_market",
    )
    return label_to_condition[label], target_date


def _station_timezone(station_id: str) -> str:
    station = STATIONS.get(station_id)
    return station.timezone if station is not None else "UTC"


def event_total_volume(
    artifact: HistoricalClobArtifact,
    *,
    target_date: str | None,
) -> float:
    if target_date is None:
        return 0.0
    event_ids = set(
        artifact.dataset.events.loc[
            artifact.dataset.events["target_date"].astype(str)
            == str(target_date),
            "event_id",
        ].astype(str)
    )
    condition_ids = set(
        artifact.dataset.markets.loc[
            artifact.dataset.markets["event_id"].astype(str).isin(event_ids),
            "condition_id",
        ].astype(str)
    )
    trades = artifact.dataset.trades[
        artifact.dataset.trades["condition_id"].astype(str).isin(condition_ids)
    ]
    return float(pd.to_numeric(trades["size"], errors="coerce").sum())


def _actual_high_for_date(
    artifact: HistoricalClobArtifact,
    target_date: str | None,
) -> float | None:
    if target_date is None or artifact.audit.event_checks.empty:
        return None
    rows = artifact.audit.event_checks[
        artifact.audit.event_checks["target_date"].astype(str)
        == str(target_date)
    ]
    values = pd.to_numeric(
        rows.get("actual_high_f", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    return float(values.iloc[0]) if not values.empty else None


def _selected_bucket_result(
    artifact: HistoricalClobArtifact,
    condition_id: str,
    actual_high: float | None,
) -> str:
    if actual_high is None:
        return "N/A"
    markets = artifact.dataset.markets
    selected = markets[
        markets["condition_id"].astype(str) == str(condition_id)
    ]
    if selected.empty:
        return "N/A"
    row = selected.iloc[0]
    is_winner = (
        actual_high >= float(row["bucket_lo_f"])
        and actual_high < float(row["bucket_hi_f"])
    )
    return "WINNER" if is_winner else "LOSER"


def _to_local(values: object, timezone: str) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(timezone)


def _format_local_time(value: object, station_id: str) -> str:
    if not value:
        return "unknown fetch time"
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    return timestamp.tz_convert(_station_timezone(station_id)).strftime(
        "%Y-%m-%d %H:%M %Z"
    )


def _top_legend() -> dict:
    return {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0.0,
    }


def _outcome_vwap(trades: pd.DataFrame, outcome: str) -> str:
    selected = trades[trades["outcome"] == outcome]
    total_size = float(selected["size"].sum())
    if total_size <= 0:
        return "N/A"
    value = float(
        (selected["price"].astype(float) * selected["size"].astype(float)).sum()
        / total_size
    )
    return f"{value:.4f}"


def _trade_price_figure(
    trades: pd.DataFrame,
    *,
    outcome: str,
    timezone: str,
) -> go.Figure:
    figure = px.scatter(
        trades,
        x="timestamp",
        y="price",
        color="side",
        size="usdc",
        hover_data={"size": ":.2f", "usdc": ":.2f"},
        labels={
            "timestamp": f"Local time ({timezone})",
            "price": f"{outcome} price",
            "side": "Side",
        },
    )
    figure.update_layout(
        legend=_top_legend(),
        yaxis_range=[0.0, 1.0],
    )
    return figure


def _synthetic_price_figure(
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    outcome: str,
    timezone: str,
    model_probabilities: pd.DataFrame | None = None,
    theoretical_execution: pd.DataFrame | None = None,
    theoretical_order_usdc: float = 50.0,
    x_axis: dict | None = None,
) -> go.Figure:
    figure = go.Figure()
    outcome = outcome.upper()
    if outcome not in {"YES", "NO"}:
        raise ValueError("outcome must be YES or NO")
    if outcome == "YES":
        series = [
            ("bid_price_yes", "Sell YES (bid)", "#1f77b4"),
            ("mid_price_yes", "Mid (reference only)", "#111827"),
            ("ask_price_yes", "Buy YES (ask)", "#d62728"),
        ]
    else:
        transformed = quotes.copy()
        transformed["bid_price_no"] = 1.0 - transformed["ask_price_yes"]
        transformed["mid_price_no"] = 1.0 - transformed["mid_price_yes"]
        transformed["ask_price_no"] = 1.0 - transformed["bid_price_yes"]
        quotes = transformed
        series = [
            ("bid_price_no", "Sell NO (bid)", "#1f77b4"),
            ("mid_price_no", "Mid (reference only)", "#111827"),
            ("ask_price_no", "Buy NO (ask)", "#d62728"),
        ]
    for column, label, color in series:
        figure.add_trace(
            go.Scatter(
                x=quotes["timestamp"],
                y=quotes[column],
                name=label,
                mode="lines",
                line={"color": color},
            )
        )
    model = (
        model_probabilities
        if model_probabilities is not None
        else pd.DataFrame()
    )
    if not model.empty:
        probability = model["yes_probability"].astype(float)
        fair_price = probability if outcome == "YES" else 1.0 - probability
        figure.add_trace(
            go.Scatter(
                x=model["timestamp"],
                y=fair_price,
                name="Model fair (theoretical)",
                mode="lines",
                line={
                    "color": "#16a34a",
                    "dash": "dash",
                    "shape": "hv",
                    "width": 2,
                },
            )
        )
    theoretical = (
        theoretical_execution
        if theoretical_execution is not None
        else pd.DataFrame()
    )
    if not theoretical.empty:
        selected_execution = theoretical[
            theoretical["outcome"].astype(str).str.upper() == outcome
        ].copy()
        if "order_usdc" not in selected_execution:
            selected_execution["order_usdc"] = theoretical_order_usdc
        colors = ["#fbbf24", "#f59e0b", "#c2410c"]
        for index, (order_usdc, rows) in enumerate(
            selected_execution.groupby("order_usdc", sort=True)
        ):
            figure.add_trace(
                go.Scatter(
                    x=rows["timestamp"],
                    y=rows["buy_vwap"],
                    name=f"Theoretical buy ${float(order_usdc):g} VWAP",
                    mode="lines",
                    line={
                        "color": colors[index % len(colors)],
                        "dash": "dot",
                        "width": 2,
                    },
                )
            )
    if not trades.empty:
        selected = trades[trades["outcome"] == outcome]
        figure.add_trace(
            go.Scatter(
                x=selected["timestamp"],
                y=selected["price"],
                name="Public trades",
                mode="markers",
                marker={"size": 5, "opacity": 0.4, "color": "#7c3aed"},
            )
        )
    figure.update_layout(
        legend=_top_legend(),
        xaxis_title=f"Local time ({timezone})",
        yaxis_title=f"{outcome} price",
        yaxis_range=[0.0, 1.0],
    )
    _apply_shared_x_axis(figure, x_axis)
    return figure


def _execution_price_frame(
    quotes: pd.DataFrame,
    *,
    order_usdc: float,
) -> pd.DataFrame:
    sampled = (
        quotes.set_index("timestamp")
        .sort_index()
        .resample("15min")
        .last()
        .dropna(subset=["bid_price_yes", "ask_price_yes"])
        .reset_index()
    )
    rows: list[dict] = []
    for quote_row in sampled.to_dict("records"):
        for outcome in ("YES", "NO"):
            book = build_synthetic_book(quote_row, outcome=outcome)
            buy = compute_quote(book, "BUY", order_usdc)
            sell = compute_quote(book, "SELL", order_usdc)
            rows.append(
                {
                    "timestamp": quote_row["timestamp"],
                    "outcome": outcome,
                    "buy_vwap": (
                        buy.vwap_price if buy.complete_fill else np.nan
                    ),
                    "sell_vwap": (
                        sell.vwap_price if sell.complete_fill else np.nan
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["timestamp", "outcome", "buy_vwap", "sell_vwap"],
    )


def _theoretical_buy_price_frame(
    quotes: pd.DataFrame,
    *,
    order_usdc: float,
    levels: int = 5,
) -> pd.DataFrame:
    """Price a hypothetical BUY ladder without applying capacity limits."""
    if order_usdc <= 0:
        raise ValueError("order_usdc must be positive")
    if levels <= 0:
        raise ValueError("levels must be positive")
    rows: list[dict] = []
    usd_per_level = float(order_usdc) / levels
    for quote_row in quotes.sort_values("timestamp").to_dict("records"):
        impact = max(0.0, float(quote_row["impact_per_usdc"]))
        best_asks = {
            "YES": float(quote_row["ask_price_yes"]),
            "NO": 1.0 - float(quote_row["bid_price_yes"]),
        }
        for outcome, best_ask in best_asks.items():
            asks = []
            for index in range(levels):
                price = float(
                    np.clip(
                        best_ask + impact * usd_per_level * index,
                        0.001,
                        0.999,
                    )
                )
                asks.append(
                    {"price": price, "size": usd_per_level / price}
                )
            quote = compute_quote(
                {"bids": [], "asks": asks},
                "BUY",
                order_usdc,
            )
            rows.append(
                {
                    "timestamp": quote_row["timestamp"],
                    "outcome": outcome,
                    "order_usdc": float(order_usdc),
                    "buy_vwap": (
                        quote.vwap_price if quote.complete_fill else np.nan
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["timestamp", "outcome", "order_usdc", "buy_vwap"],
    )


def _execution_price_figure(
    execution: pd.DataFrame,
    *,
    outcome: str,
    timezone: str,
    order_usdc: float,
    x_axis: dict | None = None,
) -> go.Figure:
    selected = execution[execution["outcome"] == outcome]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=selected["timestamp"],
            y=selected["buy_vwap"],
            name=f"Buy ${order_usdc:g} VWAP",
            mode="lines",
            line={"color": "#dc2626"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["timestamp"],
            y=selected["sell_vwap"],
            name=f"Sell ${order_usdc:g} VWAP",
            mode="lines",
            line={"color": "#2563eb"},
        )
    )
    figure.update_layout(
        legend=_top_legend(),
        xaxis_title=f"Local time ({timezone})",
        yaxis_title=f"{outcome} executable VWAP",
        yaxis_range=[0.0, 1.0],
    )
    _apply_shared_x_axis(figure, x_axis)
    return figure


def _price_impact_for_order(
    quotes: pd.DataFrame,
    order_usdc: float,
) -> pd.Series:
    executable = np.minimum(
        quotes["capacity_usdc"].astype(float),
        float(order_usdc),
    )
    return (
        quotes["impact_per_usdc"].astype(float) * executable * 100.0
    )


def _liquidity_figure(
    quotes: pd.DataFrame,
    *,
    timezone: str,
    x_axis: dict | None = None,
) -> go.Figure:
    work = quotes.set_index("timestamp").sort_index()
    interval = (
        work[["capacity_usdc", "impact_per_usdc", "rolling_volume_usdc"]]
        .resample("15min")
        .median()
        .dropna(how="all")
        .reset_index()
    )
    interval["impact_50_cents"] = _price_impact_for_order(interval, 50.0)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=interval["timestamp"],
            y=interval["capacity_usdc"],
            name="Capacity (USDC)",
            mode="lines",
            fill="tozeroy",
            line={"color": "#2563eb"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=interval["timestamp"],
            y=interval["impact_50_cents"],
            name="$50 impact (¢)",
            mode="lines",
            yaxis="y2",
            line={"color": "#dc2626"},
        )
    )
    figure.update_layout(
        legend=_top_legend(),
        xaxis_title=f"Local time ({timezone})",
        yaxis={"title": "Capacity (USDC)", "rangemode": "tozero"},
        yaxis2={
            "title": "$50 price impact (¢)",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
        },
    )
    _apply_shared_x_axis(figure, x_axis)
    return figure


def _bucket_comparison_frame(
    artifact: HistoricalClobArtifact,
    *,
    target_date: str | None,
    condition_id: str | None = None,
    timezone: str,
) -> pd.DataFrame:
    if artifact.quotes is None or target_date is None:
        return pd.DataFrame()
    events = artifact.dataset.events
    event_ids = set(
        events.loc[
            events["target_date"].astype(str) == str(target_date),
            "event_id",
        ].astype(str)
    )
    markets = artifact.dataset.markets[
        artifact.dataset.markets["event_id"].astype(str).isin(event_ids)
    ][["condition_id", "bucket_lo_f", "bucket_hi_f"]].copy()
    if condition_id is not None:
        markets = markets[
            markets["condition_id"].astype(str) == str(condition_id)
        ]
    if markets.empty:
        return pd.DataFrame()
    markets["bucket"] = [
        _bucket_label(lo, hi)
        for lo, hi in zip(markets["bucket_lo_f"], markets["bucket_hi_f"])
    ]
    comparison = artifact.quotes.quotes.merge(
        markets[["condition_id", "bucket"]],
        on="condition_id",
        how="inner",
    )
    if comparison.empty:
        return comparison
    comparison["timestamp"] = _to_local(comparison["timestamp"], timezone)
    return comparison


def _model_probability_frame(
    artifact: HistoricalClobArtifact,
    *,
    target_date: str | None,
    condition_id: str | None = None,
    timezone: str,
) -> pd.DataFrame:
    if target_date is None or not artifact.forecasts:
        return pd.DataFrame()
    events = artifact.dataset.events
    event_ids = set(
        events.loc[
            events["target_date"].astype(str) == str(target_date),
            "event_id",
        ].astype(str)
    )
    markets = artifact.dataset.markets[
        artifact.dataset.markets["event_id"].astype(str).isin(event_ids)
    ][["condition_id", "bucket_lo_f", "bucket_hi_f"]].copy()
    if condition_id is not None:
        markets = markets[
            markets["condition_id"].astype(str) == str(condition_id)
        ]
    if markets.empty:
        return pd.DataFrame()
    markets["condition_id"] = markets["condition_id"].astype(str)
    markets["bucket"] = [
        _bucket_label(lo, hi)
        for lo, hi in zip(markets["bucket_lo_f"], markets["bucket_hi_f"])
    ]
    frames: list[pd.DataFrame] = []
    condition_ids = set(markets["condition_id"])
    for cache in artifact.forecasts:
        snapshots = cache.snapshots[
            (cache.snapshots["station_id"].astype(str) == artifact.station_id)
            & (
                cache.snapshots["target_date"].astype(str)
                == str(target_date)
            )
        ]
        if snapshots.empty:
            continue
        probabilities = cache.probabilities[
            cache.probabilities["condition_id"].astype(str).isin(
                condition_ids
            )
        ].copy()
        probabilities["condition_id"] = probabilities["condition_id"].astype(
            str
        )
        frame = snapshots[
            [
                "snapshot_id",
                "init_time",
                "available_at",
                "coverage_type",
                "observed_high_f",
                "observed_hour_count",
                "forecast_hour_count",
                "target_hour_count",
            ]
        ].merge(
            probabilities,
            on="snapshot_id",
            how="inner",
            validate="one_to_many",
        ).merge(
            markets[["condition_id", "bucket"]],
            on="condition_id",
            how="inner",
            validate="many_to_one",
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    model = (
        pd.concat(frames, ignore_index=True)
        .rename(columns={"available_at": "timestamp"})
        .drop_duplicates(
            subset=["condition_id", "timestamp"],
            keep="last",
        )
        .sort_values(["timestamp", "condition_id"])
        .reset_index(drop=True)
    )
    model["timestamp"] = _to_local(model["timestamp"], timezone)
    model["init_time"] = _to_local(model["init_time"], timezone)
    return model


def _bucket_comparison_figure(
    comparison: pd.DataFrame,
    *,
    timezone: str,
    model_probabilities: pd.DataFrame | None = None,
    outcome: str = "YES",
    x_axis: dict | None = None,
) -> go.Figure:
    outcome = outcome.upper()
    if outcome not in {"YES", "NO"}:
        raise ValueError("outcome must be YES or NO")
    figure = go.Figure()
    model = (
        model_probabilities
        if model_probabilities is not None
        else pd.DataFrame()
    )
    buckets = list(
        dict.fromkeys(
            comparison.get("bucket", pd.Series(dtype=str)).astype(str).tolist()
            + model.get("bucket", pd.Series(dtype=str)).astype(str).tolist()
        )
    )
    palette = px.colors.qualitative.Plotly
    for index, bucket in enumerate(buckets):
        color = palette[index % len(palette)]
        rows = comparison[comparison["bucket"].astype(str) == bucket]
        if not rows.empty:
            sampled = (
                rows.set_index("timestamp")["mid_price_yes"]
                .resample("15min")
                .last()
                .dropna()
            )
            values = sampled if outcome == "YES" else 1.0 - sampled
            figure.add_trace(
                go.Scatter(
                    x=sampled.index,
                    y=values,
                    name="CLOB mid",
                    mode="lines",
                    line={"color": color},
                )
            )
        model_rows = (
            model[model["bucket"].astype(str) == bucket]
            if not model.empty
            else pd.DataFrame()
        )
        if not model_rows.empty:
            probability = model_rows["yes_probability"].astype(float)
            values = probability if outcome == "YES" else 1.0 - probability
            figure.add_trace(
                go.Scatter(
                    x=model_rows["timestamp"],
                    y=values,
                    name="Model fair",
                    mode="lines+markers",
                    line={"color": color, "dash": "dash", "shape": "hv"},
                    customdata=model_rows[
                        [
                            "init_time",
                            "coverage_type",
                            "observed_high_f",
                            "observed_hour_count",
                            "forecast_hour_count",
                        ]
                    ],
                    hovertemplate=(
                        "available_at=%{x}<br>"
                        "init_time=%{customdata[0]}<br>"
                        "coverage=%{customdata[1]}<br>"
                        "observed high=%{customdata[2]:.1f}°F<br>"
                        "observed hours=%{customdata[3]}<br>"
                        "forecast hours=%{customdata[4]}<br>"
                        "model fair=%{y:.3f}<extra></extra>"
                    ),
                )
            )
    figure.update_layout(
        legend=_top_legend(),
        xaxis_title=f"Local time ({timezone})",
        yaxis_title=f"{outcome} probability / midpoint",
        yaxis_range=[0.0, 1.0],
    )
    _apply_shared_x_axis(figure, x_axis)
    return figure


def _shared_synthetic_x_axis(*frames: pd.DataFrame) -> dict:
    """Return one Plotly time-axis contract for every Synthetic Book chart."""
    timestamps: list[pd.Series] = []
    for frame in frames:
        if not frame.empty and "timestamp" in frame:
            values = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            timestamps.append(values.dropna())
    if not timestamps:
        return {}
    combined = pd.concat(timestamps, ignore_index=True)
    start = combined.min()
    end = combined.max()
    if start == end:
        start -= pd.Timedelta(minutes=30)
        end += pd.Timedelta(minutes=30)
    duration = end - start
    intervals = [
        ("15min", pd.Timedelta(minutes=15)),
        ("30min", pd.Timedelta(minutes=30)),
        ("1h", pd.Timedelta(hours=1)),
        ("2h", pd.Timedelta(hours=2)),
        ("3h", pd.Timedelta(hours=3)),
        ("6h", pd.Timedelta(hours=6)),
        ("12h", pd.Timedelta(hours=12)),
        ("1D", pd.Timedelta(days=1)),
        ("2D", pd.Timedelta(days=2)),
        ("7D", pd.Timedelta(days=7)),
    ]
    frequency, interval = intervals[-1]
    for candidate_frequency, candidate in intervals:
        if duration / candidate <= 8:
            frequency, interval = candidate_frequency, candidate
            break
    return {
        "range": [start, end],
        "tick0": start.floor(frequency),
        "dtick": int(interval.total_seconds() * 1000),
        "tickformat": "%m-%d\n%H:%M",
        "margin": {"l": 80, "r": 80, "t": 40, "b": 70},
    }


def _apply_shared_x_axis(figure: go.Figure, contract: dict | None) -> None:
    if not contract:
        return
    figure.update_xaxes(
        range=contract["range"],
        tick0=contract["tick0"],
        dtick=contract["dtick"],
        tickformat=contract["tickformat"],
    )
    figure.update_layout(margin=contract["margin"])


def _bucket_label(lo: object, hi: object) -> str:
    lower = float(lo)
    upper = float(hi)
    if np.isneginf(lower):
        return f"{upper - 1:g}°F or below"
    if np.isposinf(upper):
        return f"{lower:g}°F or above"
    return f"{lower:g}–{upper - 1:g}°F"
