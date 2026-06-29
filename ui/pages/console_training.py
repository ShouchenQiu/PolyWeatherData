"""Training verification page."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from config.settings import Settings
from training.catalog import TrainingCatalog
from training.config import ResolvedTrialSnapshot
from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient


def render(
    db: "DbClient",
    settings: Settings | None = None,
    *,
    catalog_path: Path | str | None = None,
) -> None:
    st.header("Training")
    settings = settings or Settings()
    managed_catalog = (
        Path(catalog_path)
        if catalog_path is not None
        else settings.project_root
        / "runs"
        / "_management"
        / "training_catalog.db"
    )

    _render_experiment_management(managed_catalog)
    _render_split_settings(settings)
    _render_model_versions(db, settings)
    _render_training_artifacts(settings)


def _render_experiment_management(catalog_path: Path) -> None:
    st.subheader("Managed Experiments")
    if not catalog_path.exists():
        st.info(f"Training catalog not found: {catalog_path}")
        return
    with TrainingCatalog(catalog_path) as catalog:
        workspaces = catalog.list_workspaces()
        if workspaces:
            st.dataframe(
                pd.DataFrame(
                    [asdict(workspace) for workspace in workspaces]
                ),
                use_container_width=True,
            )
        studies = catalog.list_studies()
        if not studies:
            st.info("No managed studies.")
            return
        st.dataframe(
            pd.DataFrame([asdict(study) for study in studies]),
            use_container_width=True,
        )
        study_id = st.selectbox(
            "Study",
            [study.study_id for study in studies],
            key="training-study",
        )
        study = catalog.get_study(study_id)
        executions = catalog.list_executions(study_id)
        approvals = catalog.list_study_approvals(study_id)
        if approvals:
            st.dataframe(
                pd.DataFrame([asdict(approval) for approval in approvals]),
                use_container_width=True,
            )
        selected_execution = None
        if executions:
            st.dataframe(
                pd.DataFrame(
                    [asdict(execution) for execution in executions]
                ),
                use_container_width=True,
            )
            execution_id = st.selectbox(
                "Execution",
                [execution.execution_id for execution in executions],
                key=f"training-execution-{study_id}",
            )
            selected_execution = catalog.get_execution(execution_id)
            _render_execution_log(selected_execution.log_path)
        else:
            st.info("No executions for selected study.")
        try:
            if study.status == "draft" and st.button(
                "Queue smoke",
                key=f"queue-smoke-{study_id}",
            ):
                catalog.enqueue_smoke(study_id)
                st.success("Smoke queued.")
            elif study.status == "smoke_passed" and st.button(
                "Queue calibration",
                key=f"queue-calibration-{study_id}",
            ):
                catalog.enqueue_calibration(study_id)
                st.success("Calibration queued.")
            elif study.status == "gated":
                approved_by = st.text_input(
                    "Search approver",
                    key=f"search-approver-{study_id}",
                )
                if st.button(
                    "Approve search",
                    key=f"approve-search-{study_id}",
                ):
                    catalog.approve_search(
                        study_id,
                        approved_by=approved_by,
                    )
                    st.success("Search approved and queued.")
            elif study.status == "awaiting_selection":
                artifacts = catalog.list_artifacts(study_id)
                evidence_artifact = next(
                    (
                        item
                        for item in artifacts
                        if item.artifact_type
                        == "top_k_validation_evidence"
                    ),
                    None,
                )
                if evidence_artifact is None:
                    st.warning("Build frozen Top-K evidence before selection.")
                else:
                    evidence = json.loads(
                        Path(evidence_artifact.path).read_text(
                            encoding="utf-8"
                        )
                    )
                    trials = evidence.get("trials", [])
                    st.dataframe(
                        pd.DataFrame(trials),
                        use_container_width=True,
                    )
                    trial_number = st.selectbox(
                        "Final trial",
                        [int(item["trial_number"]) for item in trials],
                        key=f"final-trial-{study_id}",
                    )
                    approved_by = st.text_input(
                        "Final approver",
                        key=f"final-approver-{study_id}",
                    )
                    if st.button(
                        "Approve one final test",
                        key=f"approve-final-{study_id}",
                    ):
                        execution_id = next(
                            item["execution_id"]
                            for item in trials
                            if int(item["trial_number"]) == trial_number
                        )
                        snapshot_artifact = next(
                            item
                            for item in artifacts
                            if item.artifact_type == "trial_snapshot"
                            and item.execution_id == execution_id
                        )
                        snapshot = ResolvedTrialSnapshot.from_dict(
                            json.loads(
                                Path(snapshot_artifact.path).read_text(
                                    encoding="utf-8"
                                )
                            )
                        )
                        catalog.approve_final(
                            study_id,
                            snapshot,
                            approved_by=approved_by,
                        )
                        st.success("Final test approved exactly once.")
            elif (
                selected_execution is not None
                and selected_execution.status in {"queued", "running"}
                and st.button(
                    "Cancel execution",
                    key=f"cancel-{selected_execution.execution_id}",
                )
            ):
                catalog.request_cancel(selected_execution.execution_id)
                st.success("Cancellation recorded.")
            elif (
                selected_execution is not None
                and selected_execution.execution_type != "trial"
                and selected_execution.status
                in {"failed", "interrupted", "cancelled"}
                and st.button(
                    "Retry execution",
                    key=f"retry-{selected_execution.execution_id}",
                )
            ):
                catalog.retry_execution(selected_execution.execution_id)
                st.success("Retry queued.")
        except (PermissionError, RuntimeError, ValueError) as exc:
            st.warning(str(exc))


def _render_execution_log(log_path: str | None) -> None:
    if not log_path:
        return
    path = Path(log_path)
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    st.code("\n".join(lines[-200:]), language="text")


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
