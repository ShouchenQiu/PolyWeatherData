"""Render-layer tests for ui/pages/dashboard.py.

Uses unittest.mock.patch to intercept Streamlit calls so that
dashboard.render() can be exercised without a running Streamlit server.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from db.schema import DbClient
from ui.alerts import StalenessState
from ui.pages import dashboard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db() -> DbClient:
    return DbClient(":memory:")


def _run_render(db: DbClient, staleness_state: StalenessState | None = None) -> MagicMock:
    """Run dashboard.render() with all Streamlit calls mocked.

    Returns the mock for ``st.warning`` so callers can assert on it.
    """
    col_mock = MagicMock()
    with (
        patch("streamlit.header"),
        patch("streamlit.warning") as mock_warning,
        patch("streamlit.divider"),
        patch("streamlit.subheader"),
        patch("streamlit.info"),
        patch("streamlit.dataframe"),
        patch("streamlit.columns", return_value=[col_mock, col_mock, col_mock]),
    ):
        dashboard.render(db, staleness_state=staleness_state)
    return mock_warning


# ---------------------------------------------------------------------------
# staleness banner
# ---------------------------------------------------------------------------


def test_stale_banner_shown_when_is_stale(db: DbClient) -> None:
    """dashboard.render shows st.warning when staleness_state.is_stale is True."""
    state = StalenessState()
    state.mark_stale(_NOW, "KLGA")

    mock_warning = _run_render(db, staleness_state=state)

    assert mock_warning.called
    call_args = mock_warning.call_args[0][0]
    assert "KLGA" in call_args
    assert "12:00 UTC" in call_args


def test_stale_banner_absent_when_not_stale(db: DbClient) -> None:
    """dashboard.render does NOT call st.warning when staleness_state.is_stale is False."""
    state = StalenessState()  # not stale

    mock_warning = _run_render(db, staleness_state=state)

    mock_warning.assert_not_called()


def test_stale_banner_absent_when_no_state(db: DbClient) -> None:
    """dashboard.render does NOT crash when staleness_state=None."""
    mock_warning = _run_render(db, staleness_state=None)

    mock_warning.assert_not_called()


# ---------------------------------------------------------------------------
# _load_orders / _compute_pnl
# ---------------------------------------------------------------------------


def test_load_orders_empty_db(db: DbClient) -> None:
    """_load_orders returns an empty DataFrame when there are no orders."""
    df = dashboard._load_orders(db)
    assert df.empty


def test_load_orders_no_signal_id_join(db: DbClient) -> None:
    """_load_orders does not require the nonexistent signal_id column."""
    db.upsert_station("KLGA", 40.78, -73.88, "America/New_York")
    db.upsert_event("cid1", "Q", "slug", "2026-06-01", "KLGA")
    db.upsert_market("tok1", "cid1", "slug-60-70", 60.0, 70.0, "YES")
    db.execute(
        "INSERT INTO orders (order_id, token_id, condition_id, side, size_usd, price, status, created_at)"
        " VALUES ('ord1', 'tok1', 'cid1', 'BUY', 50.0, 0.5, 'filled', '2026-05-01')"
    )
    df = dashboard._load_orders(db)
    assert len(df) == 1
    assert df.iloc[0]["order_id"] == "ord1"


def test_compute_pnl_uses_settled_payout(db: DbClient) -> None:
    """_compute_pnl returns (settled_payout - size_usd) sum for settled orders."""
    import pandas as pd
    df = pd.DataFrame([
        {"status": "settled", "size_usd": 100.0, "settled_payout": 100.0},  # win
        {"status": "settled", "size_usd": 50.0,  "settled_payout": 0.0},   # loss
        {"status": "filled",  "size_usd": 200.0, "settled_payout": None},   # open
    ])
    pnl = dashboard._compute_pnl(df)
    assert pnl == pytest.approx(100.0 - 100.0 + 0.0 - 50.0)  # 0 + (-50) = -50
