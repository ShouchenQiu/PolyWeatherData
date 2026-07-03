from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.replay import ReplayResult
from backtest.reporting import ReplayArtifact
from ui.pages import backtest_replay


def test_backtest_page_shows_selection_and_blocks_paper_without_candidate(
    tmp_path,
) -> None:
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    pd.DataFrame(
        [
            {
                "strategy": "baseline",
                "event_days": 30,
                "total_pnl_usd": -100.0,
                "mean_event_day_pnl_usd": -3.33,
                "pnl_mean_lcb_90": -5.0,
                "pnl_mean_ucb_90": -1.0,
                "eligible": False,
                "gate_reason": "non_positive_lower_bound",
                "selected": False,
            }
        ]
    ).to_parquet(selection_root / "selection.parquet", index=False)
    (selection_root / "manifest.json").write_text(
        json.dumps(
            {
                "target_range": ["2026-04-01", "2026-04-30"],
                "selected": [],
            }
        ),
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    ReplayArtifact(
        result=ReplayResult(
            trades=pd.DataFrame(),
            settlements=pd.DataFrame(),
            starting_cash_usd=500.0,
            ending_cash_usd=400.0,
        ),
        strategy="baseline",
    ).save(results_root / "baseline")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.caption"))
        warning = stack.enter_context(patch("streamlit.warning"))
        stack.enter_context(patch("streamlit.success"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.line_chart"))
        stack.enter_context(patch("streamlit.bar_chart"))
        stack.enter_context(
            patch(
                "streamlit.selectbox",
                side_effect=lambda _label, options, **_kwargs: options[0],
            )
        )
        stack.enter_context(
            patch(
                "streamlit.columns",
                side_effect=lambda n, **_kwargs: [MagicMock() for _ in range(n)],
            )
        )
        stack.enter_context(
            patch(
                "streamlit.expander",
                return_value=MagicMock(),
            )
        )

        backtest_replay.render(
            selection_root=selection_root,
            results_root=results_root,
        )

    assert any(
        "Paper promotion blocked" in str(call.args[0])
        for call in warning.call_args_list
    )
    tables = [call.args[0] for call in dataframe.call_args_list]
    assert any(
        isinstance(table, pd.DataFrame)
        and "strategy" in table.columns
        for table in tables
    )
