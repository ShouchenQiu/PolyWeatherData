"""Read-only DataFrame queries for Streamlit pages.

Keep UI SQL here so page modules stay focused on layout and charts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from core.pnl import PnLCalculator

if TYPE_CHECKING:
    from db.schema import DbClient


RECENT_SIGNAL_DECISION_COLUMNS = (
    "signal_id",
    "tick_utc",
    "direction",
    "mc_prob",
    "reference_price",
    "edge",
    "final_token_id",
    "final_side",
    "executable_price",
    "approved",
    "reject_reason",
    "computed_size",
    "order_id",
    "condition_id",
    "station_id",
    "event_end_date",
    "order_status",
    "size_usd",
    "fill_price",
)

FORECAST_BUCKET_HISTORY_COLUMNS = (
    "snapshot_id",
    "condition_id",
    "station_id",
    "init_time",
    "event_end_date",
    "day_offset",
    "tick_utc",
    "status",
    "p50_high_f",
    "token_id",
    "bucket_lo_f",
    "bucket_hi_f",
    "mc_prob",
)

MARKET_LADDER_COLUMNS = (
    "token_id",
    "condition_id",
    "station_id",
    "event_end_date",
    "market_id",
    "market_slug",
    "bucket_lo_f",
    "bucket_hi_f",
    "outcome",
    "discovery_status",
    "price_tick_utc",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread",
    "depth_bid",
    "depth_ask",
)


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _ensure_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[list(columns)]


def overview_metrics(db: "DbClient", is_paper: bool | None = None) -> dict:
    """Return top-level production monitoring counters."""
    active_events = db.query(
        "SELECT COUNT(*) FROM events WHERE status='active'"
    )[0][0]
    tradeable_markets = db.query(
        "SELECT COUNT(*) FROM markets"
        " WHERE discovery_status='parsed'"
        " AND bucket_lo_f IS NOT NULL AND bucket_hi_f IS NOT NULL"
    )[0][0]
    open_orders = db.query(
        "SELECT COUNT(*) FROM orders WHERE status IN ('pending', 'filled')"
    )[0][0]
    approved_signals = db.query(
        "SELECT COUNT(*) FROM signals WHERE approved=1"
    )[0][0]
    rejected_signals = db.query(
        "SELECT COUNT(*) FROM signals WHERE approved=0 AND reject_reason IS NOT NULL"
    )[0][0]

    latest_pnl = pnl_history(db, is_paper=is_paper, limit=1)
    if latest_pnl.empty:
        pnl = PnLCalculator(db)
        total_pnl = pnl.total_pnl({}, is_paper=is_paper)
        gross_exposure = db.query(
            "SELECT COALESCE(SUM(size_usd), 0.0) FROM orders"
            " WHERE status IN ('pending', 'filled')"
        )[0][0]
    else:
        total_pnl = float(latest_pnl.iloc[0]["total_pnl"])
        gross_exposure = float(latest_pnl.iloc[0]["gross_exposure"])

    return {
        "active_events": int(active_events),
        "tradeable_markets": int(tradeable_markets),
        "open_orders": int(open_orders),
        "approved_signals": int(approved_signals),
        "rejected_signals": int(rejected_signals),
        "total_pnl": float(total_pnl),
        "gross_exposure": float(gross_exposure),
    }


def loop_ticks(db: "DbClient", loop_name: str | None = None, limit: int = 100) -> pd.DataFrame:
    sql = "SELECT * FROM loop_ticks"
    params: list = []
    if loop_name is not None:
        sql += " WHERE loop_name=?"
        params.append(loop_name)
    sql += " ORDER BY tick_utc DESC, tick_id DESC LIMIT ?"
    params.append(limit)
    return _df(db.query(sql, tuple(params)))


def events_overview(
    db: "DbClient",
    station_id: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    sql = (
        "SELECT e.condition_id, e.question, e.event_slug, e.event_end_date,"
        "       e.station_id, e.status,"
        "       COUNT(m.token_id) AS market_count"
        " FROM events e"
        " LEFT JOIN markets m ON m.condition_id=e.condition_id"
    )
    clauses: list[str] = []
    params: list = []
    if station_id is not None:
        clauses.append("e.station_id=?")
        params.append(station_id)
    if status is not None:
        clauses.append("e.status=?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += (
        " GROUP BY e.condition_id, e.question, e.event_slug, e.event_end_date,"
        "          e.station_id, e.status"
        " ORDER BY e.event_end_date DESC, e.condition_id"
    )
    return _df(db.query(sql, tuple(params)))


def market_ladder(db: "DbClient", condition_id: str | None = None) -> pd.DataFrame:
    sql = (
        "SELECT m.token_id, m.condition_id, e.station_id, e.event_end_date,"
        "       m.market_id, m.market_slug, m.bucket_lo_f, m.bucket_hi_f,"
        "       m.outcome, m.discovery_status,"
        "       p.tick_utc AS price_tick_utc, p.best_bid, p.best_ask,"
        "       p.mid_price, p.spread, p.depth_bid, p.depth_ask"
        " FROM markets m"
        " JOIN events e ON e.condition_id=m.condition_id"
        " LEFT JOIN market_price_snapshots p"
        "   ON p.price_snapshot_id = ("
        "       SELECT MAX(p2.price_snapshot_id)"
        "       FROM market_price_snapshots p2"
        "       WHERE p2.token_id=m.token_id"
        "   )"
    )
    params: list = []
    if condition_id is not None:
        sql += " WHERE m.condition_id=?"
        params.append(condition_id)
    sql += " ORDER BY e.event_end_date DESC, m.bucket_lo_f, m.outcome"
    return _ensure_columns(_df(db.query(sql, tuple(params))), MARKET_LADDER_COLUMNS)


def forecast_bucket_history(
    db: "DbClient",
    condition_id: str | None = None,
    station_id: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    sql = (
        "SELECT fs.snapshot_id, fs.condition_id, fs.station_id, fs.init_time,"
        "       fs.event_end_date, fs.day_offset, fs.tick_utc, fs.status,"
        "       fs.p50_high_f, bp.token_id, bp.bucket_lo_f, bp.bucket_hi_f,"
        "       bp.mc_prob"
        " FROM forecast_snapshots fs"
        " JOIN forecast_bucket_probs bp ON bp.snapshot_id=fs.snapshot_id"
    )
    clauses: list[str] = []
    params: list = []
    if condition_id is not None:
        clauses.append("fs.condition_id=?")
        params.append(condition_id)
    if station_id is not None:
        clauses.append("fs.station_id=?")
        params.append(station_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY fs.tick_utc DESC, bp.bucket_lo_f LIMIT ?"
    params.append(limit)
    return _ensure_columns(
        _df(db.query(sql, tuple(params))),
        FORECAST_BUCKET_HISTORY_COLUMNS,
    )


def forecast_quantiles(db: "DbClient", snapshot_id: int) -> pd.DataFrame:
    return _df(
        db.query(
            "SELECT * FROM forecast_quantiles"
            " WHERE snapshot_id=? ORDER BY fxx",
            (snapshot_id,),
        )
    )


def recent_signal_decisions(db: "DbClient", limit: int = 100) -> pd.DataFrame:
    return _ensure_columns(_df(
        db.query(
            """
            SELECT s.signal_id, s.tick_utc, s.direction, s.mc_prob,
                   s.reference_price, s.edge, s.final_token_id,
                   s.final_side, s.executable_price, s.approved,
                   s.reject_reason, s.computed_size, s.order_id,
                   COALESCE(m_final.condition_id, m_yes.condition_id) AS condition_id,
                   e.station_id, e.event_end_date,
                   o.status AS order_status, o.size_usd, o.fill_price
            FROM signals s
            LEFT JOIN markets m_final ON m_final.token_id=s.final_token_id
            LEFT JOIN markets m_yes ON m_yes.token_id=s.yes_token_id
            LEFT JOIN events e ON e.condition_id=COALESCE(m_final.condition_id, m_yes.condition_id)
            LEFT JOIN orders o ON o.order_id=s.order_id
            ORDER BY s.tick_utc DESC
            LIMIT ?
            """,
            (limit,),
        )
    ), RECENT_SIGNAL_DECISION_COLUMNS)


def orders_history(db: "DbClient", is_paper: bool | None = None, limit: int = 500) -> pd.DataFrame:
    sql = (
        "SELECT o.order_id, o.token_id, o.condition_id, e.station_id,"
        "       e.event_end_date, o.side, o.size_usd, o.price,"
        "       o.executable_price, o.fill_price, o.status, o.created_at,"
        "       o.filled_at, o.closed_at, o.cancelled_at, o.settled_payout,"
        "       o.is_paper, o.filled_shares, o.signed_shares, o.cash_flow_usd,"
        "       s.signal_id, s.edge, s.mc_prob, s.reference_price"
        " FROM orders o"
        " LEFT JOIN events e ON e.condition_id=o.condition_id"
        " LEFT JOIN signals s ON s.order_id=o.order_id"
    )
    params: list = []
    if is_paper is not None:
        sql += " WHERE o.is_paper=?"
        params.append(int(is_paper))
    sql += " ORDER BY o.created_at DESC LIMIT ?"
    params.append(limit)
    return _df(db.query(sql, tuple(params)))


def pnl_history(db: "DbClient", is_paper: bool | None = None, limit: int = 500) -> pd.DataFrame:
    sql = "SELECT * FROM pnl_snapshots"
    params: list = []
    if is_paper is not None:
        sql += " WHERE is_paper=?"
        params.append(int(is_paper))
    sql += " ORDER BY tick_utc DESC, pnl_snapshot_id DESC LIMIT ?"
    params.append(limit)
    return _df(db.query(sql, tuple(params)))
