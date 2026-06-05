from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from config.settings import Settings
from db.schema import DbClient
from ui.pages import console_backtest, console_home, console_training


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
