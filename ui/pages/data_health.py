"""Data Health page: ingestion and forecast freshness observability."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import Settings
from config.stations import STATIONS
from data.storage.parquet_store import ParquetStore
from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient", data_store_dir: Path | str | None = None) -> None:
    st.header("Data Health")

    store_summary = _parquet_store_summary(data_store_dir)
    st.subheader("Bootstrap Stores")
    st.dataframe(store_summary, use_container_width=True)

    st.subheader("Daily Coverage")
    st.dataframe(_daily_coverage_summary(data_store_dir), use_container_width=True)

    st.subheader("Cleaning Checks")
    st.dataframe(_cleaning_check_summary(data_store_dir), use_container_width=True)

    timing = _seed_timing_summary(data_store_dir)
    st.subheader("Seed Timing")
    if timing.empty:
        st.info("No seed timing summary.")
    else:
        st.dataframe(timing, use_container_width=True)

    ticks = queries.loop_ticks(db, loop_name="ForecastLoop")
    if ticks.empty:
        st.info("No ForecastLoop ticks recorded yet.")
    else:
        st.plotly_chart(
            px.bar(ticks.head(50), x="tick_utc", y="events_processed", color="status"),
            use_container_width=True,
        )
        st.dataframe(ticks, use_container_width=True)

    snapshots = queries.forecast_bucket_history(db)
    if snapshots.empty:
        st.info("No forecast snapshots recorded yet.")
    else:
        latest = snapshots.sort_values("tick_utc").groupby("station_id").tail(1)
        st.subheader("Latest Station Forecast Snapshots")
        st.dataframe(
            latest[["station_id", "condition_id", "tick_utc", "p50_high_f", "mc_prob"]],
            use_container_width=True,
        )


def _parquet_store_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    rows = []
    for label, subdir in (("hrrr", "hrrr"), ("obs", "obs")):
        store = ParquetStore(root / subdir)
        total = 0
        latest = None
        for station_id in STATIONS:
            df = store.read(
                station_id,
                start=pd.Timestamp("2000-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            total += len(df)
            if not df.empty:
                time_col = "valid_time" if "valid_time" in df.columns else "init_time"
                station_latest = pd.to_datetime(df[time_col], utc=True).max()
                latest = station_latest if latest is None else max(latest, station_latest)
        rows.append(
            {
                "store": label,
                "path": str(root / subdir),
                "rows": int(total),
                "latest_time": str(latest) if latest is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _resolve_data_store_dir(data_store_dir: Path | str | None = None) -> Path:
    if data_store_dir is None:
        env_dir = os.environ.get("POLYWEATHER_DATA_STORE")
        data_store_dir = Path(env_dir) if env_dir else Settings().data_store_dir
    return Path(data_store_dir)


def _latest_day(df: pd.DataFrame, time_col: str) -> pd.Timestamp | None:
    if df.empty or time_col not in df.columns:
        return None
    ts = pd.to_datetime(df[time_col], utc=True)
    return ts.max().floor("D")


def _read_store(root: Path, subdir: str, station_id: str) -> pd.DataFrame:
    return ParquetStore(root / subdir).read(
        station_id,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        end=pd.Timestamp("2100-01-01", tz="UTC"),
    )


def _daily_coverage_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    seed_day = _seed_summary_day(root)
    rows = []
    for station_id in STATIONS:
        hrrr = _read_store(root, "hrrr", station_id)
        obs = _read_store(root, "obs", station_id)
        hrrr_day = seed_day or _latest_day(hrrr, "init_time")
        obs_day = seed_day or _latest_day(obs, "valid_time")
        rows.append(_coverage_row("hrrr", station_id, hrrr, "init_time", hrrr_day, 24))
        rows.append(_coverage_row("obs", station_id, obs, "valid_time", obs_day, 24))
    return pd.DataFrame(rows)


def _coverage_row(
    store: str,
    station_id: str,
    df: pd.DataFrame,
    time_col: str,
    day: pd.Timestamp | None,
    expected_rows: int,
) -> dict:
    if day is None or df.empty or time_col not in df.columns:
        return {
            "store": store,
            "station_id": station_id,
            "day": None,
            "expected_rows": expected_rows,
            "actual_rows": 0,
            "missing_rows": expected_rows,
            "start_time": None,
            "end_time": None,
        }
    ts = pd.to_datetime(df[time_col], utc=True)
    mask = (ts >= day) & (ts < day + pd.Timedelta(days=1))
    day_df = df[mask]
    day_ts = ts[mask]
    actual = int(len(day_df))
    return {
        "store": store,
        "station_id": station_id,
        "day": str(day.date()),
        "expected_rows": expected_rows,
        "actual_rows": actual,
        "missing_rows": max(expected_rows - actual, 0),
        "start_time": str(day_ts.min()) if actual else None,
        "end_time": str(day_ts.max()) if actual else None,
    }


def _cleaning_check_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    rows = []
    for station_id in STATIONS:
        hrrr = _read_store(root, "hrrr", station_id)
        obs = _read_store(root, "obs", station_id)
        rows.extend([
            _check_row("hrrr_required_columns", station_id, {"init_time", "fxx", "valid_time", "TMP_2m"}.issubset(hrrr.columns), hrrr),
            _check_row("hrrr_fxx_all_one", station_id, bool(not hrrr.empty and "fxx" in hrrr.columns and (hrrr["fxx"] == 1).all()), hrrr),
            _check_row("obs_required_columns", station_id, {"valid_time", "raw_tmpc_tenths"}.issubset(obs.columns), obs),
        ])
    return pd.DataFrame(rows)


def _check_row(check: str, station_id: str, passed: bool, df: pd.DataFrame) -> dict:
    return {
        "check": check,
        "station_id": station_id,
        "passed": bool(passed),
        "rows": int(len(df)),
        "nan_fraction": float(df.isna().mean().mean()) if not df.empty else 0.0,
    }


def _seed_timing_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    summary = _load_seed_summary(root)
    if not summary:
        return pd.DataFrame()
    rows = []
    for stage, values in (summary.get("timings") or {}).items():
        rows.append(
            {
                "stage": stage,
                "elapsed_seconds": float(values.get("elapsed_seconds", 0.0)),
                "rows_written": int(values.get("rows_written", 0)),
                "errors": int(values.get("errors", 0)),
                "rows_per_second": float(values.get("rows_per_second", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _seed_summary_day(root: Path) -> pd.Timestamp | None:
    summary = _load_seed_summary(root)
    seed_date = summary.get("seed_date") if summary else None
    if not seed_date:
        return None
    return pd.Timestamp(seed_date, tz="UTC")


def _load_seed_summary(root: Path) -> dict:
    summary_path = root.parent / "seed_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="ascii"))
    except (json.JSONDecodeError, OSError):
        return {}
