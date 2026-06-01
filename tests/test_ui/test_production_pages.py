from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from db.schema import DbClient
from ui.pages import (
    data_health,
    forecast_model,
    overview,
    pnl_risk,
    polymarket,
    signals_orders,
)


@pytest.fixture
def db() -> DbClient:
    return DbClient(":memory:")


def _patch_streamlit():
    return (
        patch("streamlit.header"),
        patch("streamlit.subheader"),
        patch("streamlit.info"),
        patch("streamlit.warning"),
        patch("streamlit.dataframe"),
        patch("streamlit.plotly_chart"),
        patch("streamlit.selectbox", side_effect=lambda _label, options, **_kw: options[0]),
        patch("streamlit.columns", side_effect=lambda n, **_kw: [MagicMock() for _ in range(n)]),
    )


def test_production_pages_render_empty_states(db: DbClient) -> None:
    """New production-monitoring pages render empty DB states without crashing."""
    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        overview.render(db, is_paper=True)
        data_health.render(db)
        polymarket.render(db)
        signals_orders.render(db, is_paper=True)
        forecast_model.render(db)
        pnl_risk.render(db, is_paper=True)


def test_forecast_model_renders_snapshot_chart(db: DbClient) -> None:
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-ui", "Q", "slug", "2026-07-04", "KLGA", "active")
    db.upsert_market("tok-ui", "cid-ui", "slug-yes", 60.0, 70.0, "YES")
    sid = db.record_forecast_snapshot(
        "cid-ui", "KLGA", "2026-07-04T12:00:00+00:00",
        "2026-07-04", 0, "2026-07-04T12:01:00+00:00", "success", 65.0
    )
    db.record_forecast_bucket_prob(sid, "tok-ui", 60.0, 70.0, 0.7)
    db.record_forecast_quantile(sid, 1, 61.0, 65.0, 69.0)

    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        plotly_chart = stack.enter_context(patch("streamlit.plotly_chart"))
        forecast_model.render(db)

    assert plotly_chart.call_count == 2


def test_signals_orders_renders_signal_and_order_tables(db: DbClient) -> None:
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-so", "Q", "slug", "2026-07-04", "KLGA", "active")
    db.upsert_market("tok-so", "cid-so", "slug-yes", 60.0, 70.0, "YES")
    db.record_signal(
        "sig-so", "tok-so", "", "bullish", 0.8, 0.5, 0.3,
        "2026-07-04T12:00:00+00:00"
    )
    db.update_signal_decision("sig-so", "tok-so", "BUY", 0.51, 1, None, 20.0)

    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        signals_orders.render(db, is_paper=True)

    assert dataframe.called
