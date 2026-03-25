"""Model page: current model version list and per-station RMSE."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from db.schema import DbClient


def _load_model_versions(db: "DbClient") -> pd.DataFrame:
    cur = db._conn.execute(
        """
        SELECT version, checkpoint_path, train_mae, test_mae, created_at
        FROM model_versions
        ORDER BY created_at DESC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_station_rmse(db: "DbClient") -> pd.DataFrame:
    """Approximate station-level RMSE from signal residuals.

    Uses |mc_prob - clob_price| as a proxy for forecast error.
    """
    cur = db._conn.execute(
        """
        SELECT e.station_id,
               ROUND(AVG(ABS(s.mc_prob - s.clob_price)), 4) AS mean_abs_edge,
               COUNT(*) AS n_signals
        FROM signals s
        JOIN markets m ON s.token_id = m.token_id
        JOIN events  e ON m.condition_id = e.condition_id
        GROUP BY e.station_id
        ORDER BY mean_abs_edge DESC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render(db: "DbClient") -> None:
    """Render the Model page."""
    st.header("Model")

    # -- Version table ---------------------------------------------------
    st.subheader("Registered Model Versions")
    versions_df = _load_model_versions(db)
    if versions_df.empty:
        st.info("No model versions registered yet.")
    else:
        # Highlight the latest row
        st.dataframe(versions_df, use_container_width=True)

        latest = versions_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest version",  latest.get("version", "--"))
        c2.metric("Train MAE",
                  f"{latest['train_mae']:.4f}" if latest.get("train_mae") else "--")
        c3.metric("Test MAE",
                  f"{latest['test_mae']:.4f}"  if latest.get("test_mae")  else "--")

    st.divider()

    # -- Per-station signal residuals ------------------------------------
    st.subheader("Per-Station Mean |Edge| (proxy for forecast calibration)")
    rmse_df = _load_station_rmse(db)
    if rmse_df.empty:
        st.info("No signals recorded yet -- run the bot first.")
    else:
        st.bar_chart(rmse_df.set_index("station_id")["mean_abs_edge"])
        with st.expander("Raw table"):
            st.dataframe(rmse_df, use_container_width=True)
