"""Data Health page: ingestion and forecast freshness observability."""
from __future__ import annotations

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
    if data_store_dir is None:
        env_dir = os.environ.get("POLYWEATHER_DATA_STORE")
        data_store_dir = Path(env_dir) if env_dir else Settings().data_store_dir
    root = Path(data_store_dir)
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
