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
    load_error: str = ""


def discover_historical_clob_artifacts(
    data_store_dir: Path | str,
) -> list[HistoricalClobArtifact]:
    """Load and audit every local historical CLOB artifact."""
    store = Path(data_store_dir)
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
    """Select the newest freshly audited artifact, or newest artifact."""
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

    for outcome in ("YES", "NO"):
        st.subheader(f"{outcome} synthetic book")
        st.plotly_chart(
            _synthetic_price_figure(
                quotes,
                trades,
                outcome=outcome,
                timezone=timezone,
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
            ),
            width="stretch",
        )

    st.subheader("Executable liquidity")
    st.caption(
        "Capacity is the estimated executable notional. Impact estimates the "
        "price movement for a $50 order; lower is better."
    )
    st.plotly_chart(
        _liquidity_figure(quotes, timezone=timezone),
        width="stretch",
    )

    st.subheader("Bucket comparison")
    st.caption(
        "Synthetic YES midpoint by bucket. Click a legend item to hide or "
        "restore its line."
    )
    comparison = _bucket_comparison_frame(
        artifact,
        target_date=target_date,
        timezone=timezone,
    )
    if comparison.empty:
        st.info("No same-day synthetic quotes available for comparison.")
    else:
        st.plotly_chart(
            _bucket_comparison_figure(comparison, timezone=timezone),
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
) -> go.Figure:
    figure = go.Figure()
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


def _execution_price_figure(
    execution: pd.DataFrame,
    *,
    outcome: str,
    timezone: str,
    order_usdc: float,
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
    return figure


def _bucket_comparison_frame(
    artifact: HistoricalClobArtifact,
    *,
    target_date: str | None,
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


def _bucket_comparison_figure(
    comparison: pd.DataFrame,
    *,
    timezone: str,
) -> go.Figure:
    figure = go.Figure()
    for bucket, rows in comparison.groupby("bucket", sort=False):
        sampled = (
            rows.set_index("timestamp")["mid_price_yes"]
            .resample("15min")
            .last()
            .dropna()
        )
        figure.add_trace(
            go.Scatter(
                x=sampled.index,
                y=sampled,
                name=str(bucket),
                mode="lines",
            )
        )
    figure.update_layout(
        legend=_top_legend(),
        xaxis_title=f"Local time ({timezone})",
        yaxis_title="Synthetic YES midpoint",
        yaxis_range=[0.0, 1.0],
    )
    return figure


def _bucket_label(lo: object, hi: object) -> str:
    lower = float(lo)
    upper = float(hi)
    if np.isneginf(lower):
        return f"{upper - 1:g}°F or below"
    if np.isposinf(upper):
        return f"{lower:g}°F or above"
    return f"{lower:g}–{upper - 1:g}°F"
