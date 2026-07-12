from __future__ import annotations

from pathlib import Path
import json

from training.config import ExperimentTemplate

from training.historical_crossfit import (
    DefaultHistoricalFoldExecutor,
    FoldExecutionArtifacts,
    HistoricalCrossfitRunner,
    crossfit_template_payload,
    historical_six_fold_plan,
    load_historical_crossfit_predictions,
)


def test_historical_plan_has_strict_chronological_boundaries() -> None:
    folds = historical_six_fold_plan().folds

    assert len(folds) == 6
    assert folds[0].train == ("2024-10-07", "2025-01-31")
    assert folds[0].tune == ("2025-02-01", "2025-03-31")
    assert folds[0].predict == ("2025-04-01", "2025-05-31")
    assert folds[-1].predict == ("2026-02-01", "2026-03-31")
    assert len({date for fold in folds for date in fold.predict_months}) == 12
    for fold in folds:
        assert fold.train[1] < fold.tune[0] < fold.predict[0]


def test_crossfit_runner_resumes_only_hash_valid_completed_fold(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(fold, source_run, output):
        calls.append(fold.fold_id)
        checkpoint = output / "model.ckpt"
        dataset = output / "training_dataset.pkl"
        predictions = output / "predictions.parquet"
        for path, content in (
            (checkpoint, "checkpoint"),
            (dataset, "dataset"),
            (predictions, "predictions"),
        ):
            path.write_text(content, encoding="ascii")
        return FoldExecutionArtifacts(checkpoint, dataset, predictions)

    runner = HistoricalCrossfitRunner(execute_fold=execute)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    first = runner.run(source_run=source, output=output, resume=True)
    second = runner.run(source_run=source, output=output, resume=True)

    assert len(first.completed) == 6
    assert second.skipped == tuple(fold.fold_id for fold in historical_six_fold_plan().folds)
    assert calls == [fold.fold_id for fold in historical_six_fold_plan().folds]

    (output / "fold-00" / "predictions.parquet").write_text("changed", encoding="ascii")
    third = runner.run(source_run=source, output=output, resume=True)

    assert third.completed == ("fold-00",)
    assert len(calls) == 7


def test_crossfit_runner_accepts_absolute_artifacts_for_a_relative_output_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("source")
    source.mkdir()

    def execute(fold, source_run, output):
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = (output / "checkpoints" / "model.ckpt").resolve()
        checkpoint.parent.mkdir()
        dataset = (output / "training_dataset.pkl").resolve()
        predictions = (output / "predictions.parquet").resolve()
        checkpoint.write_text("checkpoint", encoding="ascii")
        dataset.write_text("dataset", encoding="ascii")
        predictions.write_text("predictions", encoding="ascii")
        return FoldExecutionArtifacts(checkpoint, dataset, predictions)

    result = HistoricalCrossfitRunner(execute_fold=execute).run(
        source_run=source, output=Path("output"), resume=True
    )

    assert len(result.completed) == 6
    manifest = json.loads(Path("output/fold-00/crossfit_manifest.json").read_text())
    assert manifest["artifacts"]["checkpoint"] == "checkpoints/model.ckpt"


def test_crossfit_runner_preserves_training_workspace_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    def execute(fold, source_run, output):
        (output / "manifest.json").write_text('{"status": "validated"}', encoding="ascii")
        paths = [output / "model.ckpt", output / "dataset.pkl", output / "predictions.parquet"]
        for path in paths:
            path.write_text(path.name, encoding="ascii")
        return FoldExecutionArtifacts(*paths)

    HistoricalCrossfitRunner(execute_fold=execute).run(
        source_run=source, output=tmp_path / "out", resume=True
    )

    fold = tmp_path / "out" / "fold-00"
    assert json.loads((fold / "manifest.json").read_text()) == {"status": "validated"}
    assert (fold / "crossfit_manifest.json").is_file()


def test_default_executor_recovers_direct_validation_result_without_retraining(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "data_store").mkdir(parents=True)
    fold = tmp_path / "fold"
    fold.mkdir()
    checkpoint = fold / "checkpoints" / "fold-00" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="ascii")
    (checkpoint.parent / "training_dataset.pkl").write_text("dataset", encoding="ascii")
    (fold / "predictions.parquet").write_text("predictions", encoding="ascii")
    (fold / "validation_result.json").write_text(
        json.dumps({"checkpoint_path": str(checkpoint)}), encoding="ascii"
    )
    (fold / "audit_summary.json").write_text(
        json.dumps({
            "ready": True,
            "splits": {
                name: {"source_fingerprint": name}
                for name in ("train", "val", "test")
            },
        }),
        encoding="ascii",
    )
    reference = Path(
        "runs/klga_train_default_20241001_20260604/artifacts/ab_experiments/hist-error-ab-a-seed42.json"
    )

    artifacts = DefaultHistoricalFoldExecutor(reference_template=reference)(
        historical_six_fold_plan().folds[0], source, fold
    )

    assert artifacts.checkpoint == checkpoint


def test_crossfit_template_uses_tune_for_early_stopping_and_predict_for_test() -> None:
    reference = {
        "schema_version": 1,
        "station": "KLGA",
        "template_id": "reference",
        "splits": {
            "train": ["2024-10-07", "2026-03-31"],
            "val": ["2026-04-01", "2026-04-30"],
            "test": ["2026-05-01", "2026-05-31"],
        },
    }

    payload = crossfit_template_payload(reference, historical_six_fold_plan().folds[0])

    assert payload["template_id"] == "historical-crossfit-fold-00"
    assert payload["splits"] == {
        "train": ["2024-10-07", "2025-01-31"],
        "val": ["2025-02-01", "2025-03-31"],
        "test": ["2025-04-01", "2025-05-31"],
    }


def test_crossfit_template_normalizes_the_verified_schema_two_reference() -> None:
    reference = json.loads(
        Path(
            "runs/klga_train_default_20241001_20260604/artifacts/ab_experiments/hist-error-ab-a-seed42.json"
        ).read_text(encoding="ascii")
    )

    payload = crossfit_template_payload(reference, historical_six_fold_plan().folds[0])
    template = ExperimentTemplate.from_dict(payload)

    assert template.template_id == "historical-crossfit-fold-00"
    assert template.model.hidden_size == 32


def test_crossfit_loader_rejects_prediction_dates_outside_fold(tmp_path: Path) -> None:
    import pandas as pd

    root = tmp_path / "crossfit"
    source = tmp_path / "source"
    source.mkdir()

    def execute(fold, source_run, output):
        checkpoint = output / "model.ckpt"
        dataset = output / "dataset.pkl"
        predictions = output / "predictions.parquet"
        checkpoint.write_text("checkpoint", encoding="ascii")
        dataset.write_text("dataset", encoding="ascii")
        pd.DataFrame({
            "init_time": [f"{fold.predict[0]}T00:00:00Z"],
            "valid_time": [f"{fold.predict[0]}T01:00:00Z"],
        }).to_parquet(predictions, index=False)
        return FoldExecutionArtifacts(checkpoint, dataset, predictions)

    HistoricalCrossfitRunner(execute_fold=execute).run(
        source_run=source, output=root, resume=True
    )
    loaded = load_historical_crossfit_predictions(root)

    assert loaded["fold_id"].nunique() == 6
    assert len(loaded) == 6
