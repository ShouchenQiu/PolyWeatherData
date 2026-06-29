from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from training.catalog import TrainingCatalog
from training.optuna_engine import OptunaEngine, TrialEvaluation
from training.selection import TopKEvidenceBuilder
from training.workspace import TrainingWorkspace


def _approved_study(tmp_path: Path) -> tuple[TrainingCatalog, str]:
    workspace = TrainingWorkspace(tmp_path / "runs" / "ready")
    workspace.ensure()
    workspace.write_json(
        workspace.audit_summary_path,
        {
            "ready": True,
            "station_id": "KLGA",
            "data_order_hash": "data-order-hash",
        },
    )
    for split, (start, end) in {
        "train": ("2024-10-07", "2026-03-31"),
        "val": ("2026-04-01", "2026-04-30"),
        "test": ("2026-05-01", "2026-05-31"),
    }.items():
        workspace.write_json(
            workspace.audit_frames_dir / f"{split}.manifest.json",
            {
                "schema_version": 1,
                "split": split,
                "station_id": "KLGA",
                "start_date": start,
                "end_date": end,
                "source_fingerprint": f"{split}-fingerprint",
                "row_count": 120,
            },
        )
    catalog = TrainingCatalog(tmp_path / "training_catalog.db")
    study = catalog.create_study(
        workspace.run_dir,
        "experiments/klga/tft_optuna_v1.json",
    )
    smoke = catalog.enqueue_smoke(study.study_id)
    catalog.start_execution(
        smoke.execution_id,
        log_path=workspace.run_dir / "studies" / study.study_id / "smoke.log",
    )
    catalog.finish_execution(
        smoke.execution_id,
        metrics={"best_val_loss": 0.7, "gpu_seconds": 10.0},
    )
    calibration = catalog.enqueue_calibration(study.study_id)
    catalog.start_execution(
        calibration.execution_id,
        log_path=(
            workspace.run_dir
            / "studies"
            / study.study_id
            / "calibration.log"
        ),
    )
    catalog.finish_execution(
        calibration.execution_id,
        metrics={"best_val_loss": 0.6, "gpu_seconds": 600.0},
    )
    catalog.approve_search(study.study_id, approved_by="pytest")
    return catalog, study.study_id


def test_prepare_creates_resumable_optuna_study_and_frozen_snapshot(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    engine = OptunaEngine(catalog, storage_url=storage_url)

    first = engine.prepare(study_id)
    resumed = engine.prepare(study_id)

    assert first.study_name == study_id
    assert resumed.study_name == first.study_name
    assert isinstance(first.sampler, optuna.samplers.TPESampler)
    assert isinstance(first.pruner, optuna.pruners.MedianPruner)
    assert optuna.get_all_study_names(storage=storage_url) == [study_id]
    snapshot = (
        Path(catalog.get_study(study_id).workspace_path)
        / "studies"
        / study_id
        / "optuna_study_snapshot.json"
    )
    assert snapshot.is_file()
    assert '"calibration_gpu_seconds": 600.0' in snapshot.read_text()


def test_trial_parameters_are_sampled_only_from_template_search_space(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    )
    optuna_study = engine.prepare(study_id)

    parameters = engine.suggest_parameters(
        study_id,
        optuna_study.ask(),
    )

    assert set(parameters) == {
        "attention_head_size",
        "batch_size",
        "dropout",
        "gradient_clip_val",
        "hidden_continuous_size",
        "hidden_size",
        "learning_rate",
        "weight_decay",
    }
    assert parameters["batch_size"] in {4, 8}
    assert 0.1 <= parameters["dropout"] <= 0.4


def test_small_search_records_completed_trial_evidence_in_both_stores(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    )

    def evaluate(context):
        context.report_epoch(0, 0.8)
        context.report_epoch(1, 0.5)
        return TrialEvaluation(
            best_val_loss=0.5,
            final_val_loss=0.5,
            epochs_run=2,
            gpu_seconds=3.0,
            gpu_peak_memory_allocated_bytes=1024,
            checkpoint_path=None,
        )

    summary = engine.run(
        study_id,
        evaluator=evaluate,
        max_trials=2,
    )

    assert summary.completed_trials == 2
    assert summary.gpu_seconds == 6.0
    assert catalog.get_study(study_id).status == "completed"
    executions = [
        item
        for item in catalog.list_executions(study_id)
        if item.execution_type == "trial"
    ]
    assert [item.status for item in executions] == ["succeeded", "succeeded"]
    assert all(item.metrics["best_val_loss"] == 0.5 for item in executions)
    loaded = engine.prepare(study_id)
    assert [trial.value for trial in loaded.trials] == [0.5, 0.5]


def test_pruned_trial_keeps_epoch_metric_and_catalog_evidence(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    )

    def prune(context):
        context.report_epoch(0, 0.9)
        raise optuna.TrialPruned("integration prune")

    summary = engine.run(study_id, evaluator=prune, max_trials=1)

    assert summary.pruned_trials == 1
    trial = engine.prepare(study_id).trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.intermediate_values == {0: 0.9}
    execution = next(
        item
        for item in catalog.list_executions(study_id)
        if item.execution_type == "trial"
    )
    assert execution.status == "pruned"
    assert execution.metrics["gpu_seconds"] >= 0


def test_oom_and_nan_trials_fail_with_explicit_classification(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    )

    def fail(context):
        if context.trial.number == 0:
            raise RuntimeError("CUDA out of memory")
        return TrialEvaluation(
            best_val_loss=float("nan"),
            final_val_loss=float("nan"),
            epochs_run=1,
            gpu_seconds=1.0,
            gpu_peak_memory_allocated_bytes=0,
            checkpoint_path=None,
        )

    summary = engine.run(study_id, evaluator=fail, max_trials=2)

    assert summary.failed_trials == 2
    executions = [
        item
        for item in catalog.list_executions(study_id)
        if item.execution_type == "trial"
    ]
    assert [item.metrics["failure_class"] for item in executions] == [
        "oom",
        "nan",
    ]


def test_broken_output_pipe_aborts_search_instead_of_spawning_failures(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    )

    def broken_output(_context):
        raise BrokenPipeError("operator output pipe closed")

    with pytest.raises(BrokenPipeError, match="output pipe"):
        engine.run(study_id, evaluator=broken_output, max_trials=10)

    executions = [
        item
        for item in catalog.list_executions(study_id)
        if item.execution_type == "trial"
    ]
    assert len(executions) == 1


def test_gpu_budget_counts_completed_trials_and_stops_before_new_trial(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    )

    def evaluate(_context):
        return TrialEvaluation(
            best_val_loss=0.5,
            final_val_loss=0.5,
            epochs_run=1,
            gpu_seconds=3.0,
            gpu_peak_memory_allocated_bytes=100,
            checkpoint_path=None,
        )

    summary = engine.run(
        study_id,
        evaluator=evaluate,
        budget_gpu_seconds=5.0,
    )

    assert summary.completed_trials == 2
    assert summary.gpu_seconds == 6.0
    assert summary.status == "budget_exhausted"
    assert catalog.get_study(study_id).status == "budget_exhausted"


def test_running_search_resumes_from_same_storage_without_repeating_trial(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"

    def evaluate(context):
        return TrialEvaluation(
            best_val_loss=0.5 + context.trial.number,
            final_val_loss=0.5 + context.trial.number,
            epochs_run=1,
            gpu_seconds=1.0,
            gpu_peak_memory_allocated_bytes=100,
            checkpoint_path=None,
        )

    first = OptunaEngine(catalog, storage_url=storage_url).run(
        study_id,
        evaluator=evaluate,
        max_trials=1,
        finalize_at_trial_cap=False,
    )
    resumed = OptunaEngine(catalog, storage_url=storage_url).run(
        study_id,
        evaluator=evaluate,
        max_trials=1,
    )

    assert first.status == "running"
    assert resumed.status == "completed"
    assert resumed.completed_trials == 2
    trials = OptunaEngine(
        catalog,
        storage_url=storage_url,
    ).prepare(study_id).trials
    assert [trial.number for trial in trials] == [0, 1]


def test_search_pause_finishes_current_trial_and_resume_starts_next(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    engine = OptunaEngine(catalog, storage_url=storage_url)

    def evaluate(context):
        return TrialEvaluation(
            best_val_loss=0.5 + context.trial.number,
            final_val_loss=0.5 + context.trial.number,
            epochs_run=1,
            gpu_seconds=1.0,
            gpu_peak_memory_allocated_bytes=100,
            checkpoint_path=None,
        )

    paused = engine.run(
        study_id,
        evaluator=evaluate,
        max_trials=3,
        pause_requested=lambda: True,
    )
    assert paused.status == "paused"
    assert catalog.get_study(study_id).status == "paused"
    assert paused.completed_trials == 1

    catalog.resume_search(study_id)
    resumed = engine.run(study_id, evaluator=evaluate, max_trials=1)

    assert resumed.status == "completed"
    assert [trial.number for trial in engine.prepare(study_id).trials] == [0, 1]


def test_default_tft_adapter_reports_epoch_and_indexes_runtime_artifacts(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    received_parameters = None

    def training_backend(
        *,
        output_dir,
        progress_reporter,
        trial_parameters,
        **_kwargs,
    ):
        nonlocal received_parameters
        received_parameters = dict(trial_parameters)
        progress_reporter(
            {
                "phase": "validation_epoch_end",
                "epoch": 0,
                "validation_quantile_loss": 0.4,
            }
        )
        checkpoint = Path(output_dir) / "best.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="ascii")
        return {
            "best_val_loss": 0.4,
            "final_val_loss": 0.4,
            "epochs_run": 1,
            "gpu_seconds": 2.0,
            "gpu_peak_memory_allocated_bytes": 2048,
            "checkpoint_path": str(checkpoint),
        }

    engine = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
        training_backend=training_backend,
    )

    summary = engine.run(study_id, max_trials=1)

    assert summary.completed_trials == 1
    assert received_parameters is not None
    trial = engine.prepare(study_id).trials[0]
    assert trial.intermediate_values == {0: 0.4}
    artifact_types = {
        item.artifact_type for item in catalog.list_artifacts(study_id)
    }
    assert {
        "trial_snapshot",
        "execution_log",
        "validation_result",
        "checkpoint",
    } <= artifact_types


def test_resume_marks_abandoned_optuna_trial_failed_and_starts_next_number(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    engine = OptunaEngine(catalog, storage_url=storage_url)
    optuna_study = engine.prepare(study_id)
    catalog.start_search(study_id)
    abandoned = optuna_study.ask()
    catalog.start_trial_execution(
        study_id,
        trial_number=abandoned.number,
        limits={},
        log_path=tmp_path / "abandoned.log",
    )
    catalog.recover_interrupted_executions(worker_id="replacement")

    def evaluate(_context):
        return TrialEvaluation(
            best_val_loss=0.4,
            final_val_loss=0.4,
            epochs_run=1,
            gpu_seconds=1.0,
            gpu_peak_memory_allocated_bytes=0,
            checkpoint_path=None,
        )

    resumed = engine.run(study_id, evaluator=evaluate, max_trials=1)

    assert resumed.completed_trials == 1
    assert resumed.failed_trials == 1
    assert [
        trial.number for trial in engine.prepare(study_id).trials
    ] == [0, 1]


def test_running_optuna_trial_can_be_cancelled_through_catalog(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    catalog_path = catalog.path

    def training_backend(*, version, progress_reporter, **_kwargs):
        with TrainingCatalog(catalog_path) as operator:
            operator.request_cancel(version)
        progress_reporter(
            {
                "phase": "validation_epoch_end",
                "epoch": 0,
                "validation_quantile_loss": 0.8,
            }
        )
        raise AssertionError("cancelled trial must stop immediately")

    summary = OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
        training_backend=training_backend,
    ).run(study_id, max_trials=3)

    assert summary.status == "cancelled"
    assert catalog.get_study(study_id).status == "cancelled"
    trial_execution = next(
        item
        for item in catalog.list_executions(study_id)
        if item.execution_type == "trial"
    )
    assert trial_execution.status == "cancelled"


def test_top_k_manifest_ranks_validation_only_trials_and_locks_selection(
    tmp_path: Path,
) -> None:
    catalog, study_id = _approved_study(tmp_path)
    losses = [0.6, 0.3, 0.4]

    def evaluate(context):
        loss = losses[context.trial.number]
        return TrialEvaluation(
            best_val_loss=loss,
            final_val_loss=loss,
            epochs_run=1,
            gpu_seconds=1.0,
            gpu_peak_memory_allocated_bytes=100,
            checkpoint_path=None,
        )

    OptunaEngine(
        catalog,
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
    ).run(study_id, evaluator=evaluate, max_trials=3)

    seen_splits = []

    def downstream(context):
        seen_splits.append(context.split)
        return {
            "p50_mae": context.trial_number / 10,
            "pinball": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "coverage_p10_p90": 0.8,
            "interval_width": 2.0,
        }

    manifest = TopKEvidenceBuilder(catalog).build(
        study_id,
        evidence_provider=downstream,
    )

    assert [item["trial_number"] for item in manifest["trials"]] == [1, 2, 0]
    assert seen_splits == ["val", "val", "val"]
    assert manifest["test_accessed"] is False
    assert catalog.get_study(study_id).status == "awaiting_selection"
