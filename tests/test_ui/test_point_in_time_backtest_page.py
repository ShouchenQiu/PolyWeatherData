from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.replay import ReplayResult
from backtest.reporting import ReplayArtifact
from backtest.signal_slice_study import save_signal_slice_artifact
from backtest.signal_study import (
    save_signal_study_artifact,
    score_signal_panel,
)
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


def test_signal_quality_renders_without_strategy_selection(tmp_path) -> None:
    panel = pd.DataFrame(
        [
            {
                "event_id": "event",
                "target_date": "2026-04-01",
                "timestamp": pd.Timestamp("2026-04-01T12:00:00Z"),
                "condition_id": condition,
                "bucket_order": order,
                "model_probability": model,
                "midpoint_probability": midpoint,
                "midpoint_sum": 1.05,
                "midpoint_overround": 0.05,
                "yes_outcome": float(order == 0),
                "lead_band": "12-24h",
            }
            for order, (condition, model, midpoint) in enumerate(
                [("low", 0.7, 0.4), ("high", 0.3, 0.6)]
            )
        ]
    )
    report = score_signal_panel(panel, bootstrap_samples=10)
    signal_root = tmp_path / "signal"
    save_signal_study_artifact(
        signal_root,
        study_id="signal-test",
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        split_role="oof_development",
        input_sha256={"forecast_manifest": "a" * 64},
        panel=panel,
        report=report,
    )
    slice_root = tmp_path / "slices"
    slice_summary = {
        "sample": {
            "event_count": 1,
            "fixed_slice_count": 1,
            "crossfit_rule_count": 1,
        },
        "shortlist": {
            "tier_counts": {
                "stable_candidate": 1,
                "promising_candidate": 0,
                "exploratory_only": 0,
            }
        },
        "parameters": {
            "folds": 5,
            "bootstrap_samples": 10,
            "seed": 42,
            "min_training_events": 1,
            "minimum_shortlist_events": 1,
        },
    }
    save_signal_slice_artifact(
        slice_root,
        parent_manifest_sha256="a" * 64,
        forecast_manifest_sha256="b" * 64,
        parent_study_id="signal-test",
        station_id="KLGA",
        target_range=["2026-04-01", "2026-04-01"],
        summary=slice_summary,
        vector_features=pd.DataFrame([{"event_id": "event"}]),
        token_candidates=pd.DataFrame([{"event_id": "event"}]),
        slice_atlas=pd.DataFrame(
            [
                {
                    "slice_id": "lead_band=12-24h&tv_band=0.15-0.30",
                    "dimensions": "lead_band|tv_band",
                    "values": "12-24h|0.15-0.30",
                    "event_count": 1,
                    "brier_delta": -0.1,
                    "rps_delta": -0.01,
                    "theoretical_midpoint_excess": 0.1,
                }
            ]
        ),
        crossfit_results=pd.DataFrame(
            [{"fold": 0, "slice_id": "rule", "brier_delta": -0.1}]
        ),
        fold_rules=pd.DataFrame(
            [{"fold": 0, "slice_id": "rule", "selected_top3": True}]
        ),
        shortlist=pd.DataFrame(
            [
                {
                    "slice_id": "rule",
                    "evidence_tier": "stable_candidate",
                    "held_out_event_count": 1,
                }
            ]
        ),
    )

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        subheader = stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.caption"))
        warning = stack.enter_context(patch("streamlit.warning"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.line_chart"))
        stack.enter_context(
            patch(
                "streamlit.columns",
                side_effect=lambda n, **_kwargs: [MagicMock() for _ in range(n)],
            )
        )

        backtest_replay.render(
            signal_root=signal_root,
            slice_root=slice_root,
            selection_root=tmp_path / "missing-selection",
            results_root=tmp_path / "missing-results",
        )

    assert any(
        "Signal Quality" in str(call.args[0])
        for call in subheader.call_args_list
    )
    assert any(
        "Advantage Slices" in str(call.args[0])
        for call in subheader.call_args_list
    )
    assert any(
        "non-executable" in str(call.args[0]).lower()
        for call in warning.call_args_list
    )
    assert any(
        "No strategy-selection artifact" in str(call.args[0])
        for call in warning.call_args_list
    )
    tables = [call.args[0] for call in dataframe.call_args_list]
    assert any(
        isinstance(table, pd.DataFrame)
        and "model_minus_midpoint" in table.columns
        for table in tables
    )
