from __future__ import annotations

import json
import hashlib
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset
from ui.pages.historical_clob import (
    _bucket_comparison_figure,
    _execution_price_figure,
    _liquidity_figure,
    _shared_synthetic_x_axis,
    _synthetic_price_figure,
    default_artifact_index,
    discover_historical_clob_artifacts,
    event_total_volume,
    render,
)


def _artifact(root, name: str, *, gap: bool, fetched_at: str) -> None:
    high_start = 61.0 if gap else 60.0
    HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": name, "target_date": "2026-04-01"}]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": name,
                    "condition_id": f"{name}-low",
                    "yes_token_id": f"{name}-low-yes",
                    "no_token_id": f"{name}-low-no",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": 60.0,
                },
                {
                    "event_id": name,
                    "condition_id": f"{name}-high",
                    "yes_token_id": f"{name}-high-yes",
                    "no_token_id": f"{name}-high-no",
                    "bucket_lo_f": high_start,
                    "bucket_hi_f": float("inf"),
                },
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": f"{name}-low",
                    "token_id": f"{name}-low-yes",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "outcome": "YES",
                    "side": "BUY",
                    "price": 0.5,
                    "size": 10.0,
                    "usdc": 5.0,
                    "source": "fixture",
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    ).save(
        root / "backtest" / "clob" / "KLGA" / name,
        fetched_at=fetched_at,
    )


def _attach_synthetic_quotes(root, artifact_name: str) -> None:
    manifest_path = (
        root
        / "backtest"
        / "clob"
        / "KLGA"
        / artifact_name
        / "manifest.json"
    )
    source_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    SyntheticQuoteDataset(
        quotes=pd.DataFrame(
            [
                {
                    "condition_id": f"{artifact_name}-low",
                    "timestamp": "2026-04-01T12:05:00Z",
                    "bid_price_yes": 0.49,
                    "mid_price_yes": 0.50,
                    "ask_price_yes": 0.51,
                    "estimated_spread": 0.02,
                    "impact_per_usdc": 0.001,
                    "capacity_usdc": 100.0,
                    "rolling_volume_usdc": 200.0,
                    "trade_count": 1,
                    "buy_yes_trade_count": 1,
                    "sell_yes_trade_count": 0,
                    "last_trade_at": "2026-04-01T12:00:00Z",
                    "staleness_seconds": 300.0,
                    "quote_quality": "one_sided_trade_proxy",
                    "source": "past_public_trades",
                }
            ]
        ),
        start_at="2026-04-01T12:05:00Z",
        end_at="2026-04-01T12:05:00Z",
        decision_tick="5min",
        lookback="60min",
        max_staleness="30min",
    ).save(
        root / "backtest" / "quotes" / "KLGA" / "quotes",
        source_sha256=source_sha256,
    )


def test_catalog_defaults_to_latest_passing_artifact(tmp_path) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    _artifact(
        tmp_path,
        "newer-invalid",
        gap=True,
        fetched_at="2026-07-02T00:00:00Z",
    )

    artifacts = discover_historical_clob_artifacts(tmp_path)

    assert [artifact.name for artifact in artifacts] == [
        "newer-invalid",
        "valid",
    ]
    assert not artifacts[0].audit.passed
    assert artifacts[1].audit.passed
    assert default_artifact_index(artifacts) == 1


def test_catalog_pairs_synthetic_quotes_by_source_manifest_hash(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    manifest_path = (
        tmp_path
        / "backtest"
        / "clob"
        / "KLGA"
        / "valid"
        / "manifest.json"
    )
    import hashlib

    source_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    SyntheticQuoteDataset(
        quotes=pd.DataFrame(
            [
                {
                    "condition_id": "valid-low",
                    "timestamp": "2026-04-01T12:05:00Z",
                    "bid_price_yes": 0.49,
                    "mid_price_yes": 0.50,
                    "ask_price_yes": 0.51,
                    "estimated_spread": 0.02,
                    "impact_per_usdc": 0.001,
                    "capacity_usdc": 10.0,
                    "rolling_volume_usdc": 20.0,
                    "trade_count": 1,
                    "buy_yes_trade_count": 1,
                    "sell_yes_trade_count": 0,
                    "last_trade_at": "2026-04-01T12:00:00Z",
                    "staleness_seconds": 300.0,
                    "quote_quality": "one_sided_trade_proxy",
                    "source": "past_public_trades",
                }
            ]
        ),
        start_at="2026-04-01T12:05:00Z",
        end_at="2026-04-01T12:05:00Z",
        decision_tick="5min",
        lookback="60min",
        max_staleness="30min",
    ).save(
        tmp_path / "backtest" / "quotes" / "KLGA" / "quotes",
        source_sha256=source_sha256,
    )

    artifact = discover_historical_clob_artifacts(tmp_path)[0]

    assert artifact.quotes is not None
    assert artifact.quotes_root is not None
    assert artifact.audit.passed
    assert artifact.audit.summary["quotes"] == 1


def test_catalog_applies_explicit_exclusions_from_quality_artifact(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    root = tmp_path / "backtest" / "clob" / "KLGA" / "valid"
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    manifest["requested_range"] = ["2026-04-01", "2026-04-02"]
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (root.with_suffix(".quality.json")).write_text(
        json.dumps(
            {
                "excluded_days": {
                    "2026-04-02": "provider_event_unavailable"
                }
            }
        ),
        encoding="utf-8",
    )

    artifact = discover_historical_clob_artifacts(tmp_path)[0]

    assert artifact.audit.passed
    assert artifact.audit.summary["excluded_days"] == 1


def test_catalog_uses_available_actual_highs_without_requiring_full_coverage(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    actual_root = tmp_path / "backtest" / "actuals" / "KLGA"
    actual_root.mkdir(parents=True)
    pd.DataFrame(
        [{"target_date": "2026-04-01", "actual_high_f": 61.0}]
    ).to_csv(actual_root / "actuals.csv", index=False)

    artifact = discover_historical_clob_artifacts(tmp_path)[0]

    assert artifact.audit.passed
    assert artifact.audit.summary["actual_days"] == 1
    assert artifact.audit.event_checks.iloc[0]["resolution_status"] == "PASS"


def test_historical_clob_page_renders_only_trades_and_synthetic_tabs(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    artifact = discover_historical_clob_artifacts(tmp_path)[0]

    def select_first(_label, options, **kwargs):
        return options[kwargs.get("index", 0)]

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.caption"))
        stack.enter_context(patch("streamlit.success"))
        stack.enter_context(patch("streamlit.error"))
        stack.enter_context(
            patch("streamlit.selectbox", side_effect=select_first)
        )
        tabs = stack.enter_context(
            patch(
                "streamlit.tabs",
                return_value=[nullcontext() for _ in range(2)],
            )
        )
        stack.enter_context(
            patch(
                "ui.pages.historical_clob._load_catalog",
                return_value=[artifact],
            )
        )
        trades = stack.enter_context(
            patch("ui.pages.historical_clob._render_trades")
        )
        synthetic = stack.enter_context(
            patch("ui.pages.historical_clob._render_synthetic_book")
        )

        render(data_store_dir=tmp_path)

    tabs.assert_called_once_with(["Trades", "Synthetic Book"])
    for renderer in (trades, synthetic):
        renderer.assert_called_once_with(artifact)


def test_catalog_recomputes_manifest_row_counts(tmp_path) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    manifest_path = (
        tmp_path
        / "backtest"
        / "clob"
        / "KLGA"
        / "valid"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["row_counts"]["trades"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact = discover_historical_clob_artifacts(tmp_path)[0]

    assert not artifact.audit.passed
    assert "manifest_row_counts" in artifact.audit.failures


def test_catalog_retains_artifact_with_checksum_failure(tmp_path) -> None:
    _artifact(
        tmp_path,
        "corrupt",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    trades_path = (
        tmp_path
        / "backtest"
        / "clob"
        / "KLGA"
        / "corrupt"
        / "trades.parquet"
    )
    with trades_path.open("ab") as handle:
        handle.write(b"tampered")

    artifact = discover_historical_clob_artifacts(tmp_path)[0]

    assert artifact.dataset is None
    assert not artifact.audit.passed
    assert artifact.audit.failures == ("artifact_integrity",)
    assert "checksum mismatch" in artifact.load_error


def test_page_selects_bucket_without_low_level_filters_and_degrades_without_quotes(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    app = AppTest.from_string(
        "from ui.pages.historical_clob import render\n"
        f"render(data_store_dir={str(tmp_path)!r})\n"
    ).run(timeout=20)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Trades",
        "Synthetic Book",
    ]
    labels = [selectbox.label for selectbox in app.selectbox]
    assert "Market / Bucket" in labels
    assert "Event" not in labels
    assert "Outcome" not in labels
    assert "Side" not in labels
    assert "Raw row limit" not in labels
    assert all("valid-" not in option for option in app.selectbox[
        labels.index("Market / Bucket")
    ].options)
    assert not app.date_input
    assert any(
        "No synthetic quote artifact matches" in warning.value
        for warning in app.warning
    )


def test_trades_page_shows_actual_high_and_selected_bucket_result(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    actual_root = tmp_path / "backtest" / "actuals" / "KLGA"
    actual_root.mkdir(parents=True)
    pd.DataFrame(
        [{"target_date": "2026-04-01", "actual_high_f": 61.0}]
    ).to_csv(actual_root / "actuals.csv", index=False)

    app = AppTest.from_string(
        "from ui.pages.historical_clob import render\n"
        f"render(data_store_dir={str(tmp_path)!r})\n"
    ).run(timeout=20)

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Actual high"] == "61°F"
    assert metrics["Selected bucket"] == "LOSER"


def test_synthetic_page_charts_actionable_50_dollar_execution_prices(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )
    _attach_synthetic_quotes(tmp_path, "valid")

    app = AppTest.from_string(
        "from ui.pages.historical_clob import render\n"
        f"render(data_store_dir={str(tmp_path)!r})\n"
    ).run(timeout=20)

    subheaders = [subheader.value for subheader in app.subheader]
    assert "$50 executable prices" in subheaders
    assert "YES executable price" in subheaders
    assert "NO executable price" in subheaders


def test_synthetic_book_figures_share_the_full_page_time_axis() -> None:
    selected_quotes = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01T12:00:00Z"),
                "bid_price_yes": 0.40,
                "mid_price_yes": 0.41,
                "ask_price_yes": 0.42,
                "capacity_usdc": 100.0,
                "impact_per_usdc": 0.001,
                "rolling_volume_usdc": 200.0,
            },
            {
                "timestamp": pd.Timestamp("2026-04-01T18:00:00Z"),
                "bid_price_yes": 0.45,
                "mid_price_yes": 0.46,
                "ask_price_yes": 0.47,
                "capacity_usdc": 120.0,
                "impact_per_usdc": 0.001,
                "rolling_volume_usdc": 250.0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01T13:00:00Z"),
                "outcome": "YES",
                "price": 0.41,
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01T06:00:00Z"),
                "bucket": "59°F or below",
                "mid_price_yes": 0.10,
            },
            {
                "timestamp": pd.Timestamp("2026-04-02T00:00:00Z"),
                "bucket": "60°F or above",
                "mid_price_yes": 0.90,
            },
        ]
    )
    execution = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "outcome": outcome,
                "buy_vwap": 0.48,
                "sell_vwap": 0.44,
            }
            for timestamp in selected_quotes["timestamp"]
            for outcome in ("YES", "NO")
        ]
    )
    axis = _shared_synthetic_x_axis(
        selected_quotes,
        trades,
        comparison,
    )
    figures = [
        _synthetic_price_figure(
            selected_quotes,
            trades,
            outcome=outcome,
            timezone="America/New_York",
            x_axis=axis,
        )
        for outcome in ("YES", "NO")
    ]
    figures.extend(
        _execution_price_figure(
            execution,
            outcome=outcome,
            timezone="America/New_York",
            order_usdc=50.0,
            x_axis=axis,
        )
        for outcome in ("YES", "NO")
    )
    figures.extend(
        [
            _liquidity_figure(
                selected_quotes,
                timezone="America/New_York",
                x_axis=axis,
            ),
            _bucket_comparison_figure(
                comparison,
                timezone="America/New_York",
                x_axis=axis,
            ),
        ]
    )

    layouts = [
        (
            tuple(figure.layout.xaxis.range),
            figure.layout.xaxis.tick0,
            figure.layout.xaxis.dtick,
            figure.layout.xaxis.tickformat,
            figure.layout.margin.to_plotly_json(),
        )
        for figure in figures
    ]
    assert all(layout == layouts[0] for layout in layouts)
    assert pd.Timestamp(layouts[0][0][0]) == pd.Timestamp(
        "2026-04-01T06:00:00Z"
    )
    assert pd.Timestamp(layouts[0][0][1]) == pd.Timestamp(
        "2026-04-02T00:00:00Z"
    )


def test_trades_page_shows_official_compatible_event_volume_without_raw_rows(
    tmp_path,
) -> None:
    _artifact(
        tmp_path,
        "valid",
        gap=False,
        fetched_at="2026-07-01T00:00:00Z",
    )

    app = AppTest.from_string(
        "from ui.pages.historical_clob import render\n"
        f"render(data_store_dir={str(tmp_path)!r})\n"
    ).run(timeout=20)

    artifact = discover_historical_clob_artifacts(tmp_path)[0]
    assert event_total_volume(
        artifact,
        target_date="2026-04-01",
    ) == 10.0
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Event volume"] == "$10"
    subheaders = [subheader.value for subheader in app.subheader]
    assert "Daily total volume · all buckets" not in subheaders
    assert "Latest 500 public trades" not in subheaders
    assert not app.dataframe
