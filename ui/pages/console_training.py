"""Training verification page."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from config.settings import Settings
from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(db: "DbClient", settings: Settings | None = None) -> None:
    st.header("Training")
    settings = settings or Settings()

    _render_split_settings(settings)
    _render_model_versions(db, settings)
    _render_training_artifacts(settings)


def _render_split_settings(settings: Settings) -> None:
    st.subheader("Configured Splits")
    rows = [
        {"split": "train", "start": settings.train_start, "end": settings.train_end},
        {"split": "val", "start": settings.val_start, "end": settings.val_end},
        {"split": "test", "start": settings.test_start, "end": settings.test_end},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_model_versions(db: "DbClient", settings: Settings) -> None:
    st.subheader("Model Registry")
    versions = queries.model_versions(db)
    if versions.empty:
        st.info("No model versions registered yet.")
    else:
        st.dataframe(versions, use_container_width=True)

    active_path = settings.model_store_dir / "active_version.txt"
    active = active_path.read_text(encoding="ascii").strip() if active_path.exists() else None
    cols = st.columns(3)
    cols[0].metric("Registered Versions", len(versions))
    cols[1].metric("Active Version", active or "--")
    cols[2].metric("Store Exists", "yes" if settings.model_store_dir.exists() else "no")


def _render_training_artifacts(settings: Settings) -> None:
    st.subheader("Training Artifacts")
    rows = []
    for path in sorted(_version_dirs(settings.model_store_dir)):
        rows.append({
            "version": path.name,
            "model_ckpt": (path / "model.ckpt").exists(),
            "training_dataset_pkl": (path / "training_dataset.pkl").exists(),
            "meta_json": (path / "meta.json").exists(),
            "covariance_npy": (path / "covariance.npy").exists(),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No version artifact directories found.")

    cache_dir = settings.training_decoder_cache_dir
    cache_files = list(cache_dir.rglob("*.parquet")) if cache_dir.exists() else []
    st.metric("Decoder Cache Files", len(cache_files))


def _version_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]
