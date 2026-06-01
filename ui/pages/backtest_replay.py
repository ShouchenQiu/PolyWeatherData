"""Backtest & Replay page wrapper."""
from __future__ import annotations

import pandas as pd

from ui.pages import backtest


def render(results_df: pd.DataFrame | None = None) -> None:
    backtest.render(results_df=results_df)
