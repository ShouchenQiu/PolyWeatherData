from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from config.settings import Settings
from db.schema import DbClient
from ui.pages import console_backtest, console_event_trace, console_home, console_training


def _patch_streamlit():
    return (
        patch("streamlit.header"),
        patch("streamlit.subheader"),
        patch("streamlit.info"),
        patch("streamlit.dataframe"),
        patch("streamlit.metric"),
        patch("streamlit.divider"),
        patch("streamlit.columns", side_effect=lambda n, **_kw: [MagicMock() for _ in range(n)]),
    )


def test_console_pages_render_empty_states(tmp_path):
    db = DbClient(":memory:")
    settings = Settings(
        data_store_dir=tmp_path / "data_store",
        model_store_dir=tmp_path / "models",
    )

    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        stack.enter_context(patch("ui.pages.backtest.render"))

        console_home.render(db)
        console_training.render(db, settings=settings)
        console_backtest.render(db)

    db.close()


def test_console_training_renders_model_artifact_summary(tmp_path):
    db = DbClient(":memory:")
    settings = Settings(
        data_store_dir=tmp_path / "data_store",
        model_store_dir=tmp_path / "models",
    )
    version_dir = settings.model_store_dir / "v1"
    version_dir.mkdir(parents=True)
    (version_dir / "model.ckpt").write_text("ckpt", encoding="ascii")
    (version_dir / "training_dataset.pkl").write_text("pkl", encoding="ascii")
    (version_dir / "meta.json").write_text("{}", encoding="ascii")
    (settings.model_store_dir / "active_version.txt").write_text("v1", encoding="ascii")
    db.register_model_version(
        "v1",
        str(version_dir / "model.ckpt"),
        train_mae=0.5,
        test_mae=0.7,
        created_at="2026-06-05T00:00:00+00:00",
    )

    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        dataframe = stack.enter_context(patch("streamlit.dataframe"))

        console_training.render(db, settings=settings)

    rendered_tables = [call.args[0] for call in dataframe.call_args_list]
    assert any("version" in table.columns for table in rendered_tables)
    assert any(
        "training_dataset_pkl" in table.columns
        and bool(table.loc[table["version"] == "v1", "training_dataset_pkl"].iloc[0])
        for table in rendered_tables
    )
    db.close()


def test_console_event_trace_hides_token_ids_and_shows_clob_snapshot():
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-ui", "Highest temp?", "highest-temperature-in-nyc", "2026-06-05", "KLGA", "active")
    db.upsert_market("tok-yes", "cid-ui", "bucket-80-81f", 80.0, 81.0, "Yes", market_id="mkt-1")
    db.record_market_price_snapshot(
        "tok-yes",
        "cid-ui",
        "2026-06-05T00:00:00+00:00",
        best_bid=0.42,
        best_ask=0.44,
        mid_price=0.43,
        spread=0.02,
        depth_bid=100.0,
        depth_ask=120.0,
    )

    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        stack.enter_context(patch("streamlit.selectbox", side_effect=lambda _label, options, **_kw: options[0]))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))

        console_event_trace.render(db)

    rendered_tables = [call.args[0] for call in dataframe.call_args_list]
    ladder = next(table for table in rendered_tables if "market_slug" in table.columns)
    assert "token_id" not in ladder.columns
    assert ladder.loc[0, "best_bid"] == 0.42
    assert ladder.loc[0, "depth_ask"] == 120.0
    db.close()


def test_console_event_trace_renders_markets_without_clob_snapshot():
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-ui", "Highest temp?", "highest-temperature-in-nyc", "2026-06-05", "KLGA", "active")
    db.upsert_market("tok-yes", "cid-ui", "bucket-80-81f", 80.0, 81.0, "Yes", market_id="mkt-1")

    with ExitStack() as stack:
        for patcher in _patch_streamlit():
            stack.enter_context(patcher)
        stack.enter_context(patch("streamlit.selectbox", side_effect=lambda _label, options, **_kw: options[0]))
        info = stack.enter_context(patch("streamlit.info"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))

        console_event_trace.render(db)

    rendered_tables = [call.args[0] for call in dataframe.call_args_list]
    ladder = next(table for table in rendered_tables if "market_slug" in table.columns)
    assert "token_id" not in ladder.columns
    assert ladder.loc[0, "market_id"] == "mkt-1"
    assert any("No CLOB snapshots" in call.args[0] for call in info.call_args_list)
    db.close()
