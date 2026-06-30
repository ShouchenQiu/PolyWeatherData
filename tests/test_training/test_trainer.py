from __future__ import annotations

import sys
import types
from pathlib import Path
import json

import pandas as pd
import pytest

from config.settings import Settings
from config.stations import STATIONS
from training.audit_frames import AuditedFrameCache
from training.config import ExperimentTemplate, ResolvedTrialSnapshot
from training.experiment import FinalEvaluationApproval, ValidationExecutionLimits
from training.trainer import TftTrainer


class FakeBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp, str]] = []
        self.skipped = []

    def build_frame(self, station, init_times, split):
        self.calls.append((station.station_id, init_times[0], init_times[-1], split))
        rows = []
        base = init_times[0]
        for i in range(120):
            rows.append(
                {
                    "valid_time": base + pd.Timedelta(hours=i),
                    "init_time_str": str(base),
                    "Node_ID": station.station_id,
                    "local_idx": i,
                    "time_idx": i,
                    "fxx": max(1, i - 71),
                    "TMP_2m": 5.0,
                    "temp_error": 1.0,
                    "hist_temp_error": 1.0 if i < 72 else 0.0,
                    "decoder_weight": 1.0,
                    "split": split,
                }
            )
        return pd.DataFrame(rows)


class FakeRegistry:
    def __init__(self) -> None:
        self.publish_kwargs = None

    def publish(self, **kwargs):
        self.publish_kwargs = kwargs


class FakeTimeSeriesDataSet:
    created_train_frames: list[pd.DataFrame] = []
    created_train_kwargs: list[dict] = []

    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
        self.target_normalizer = kwargs.get("target_normalizer")
        FakeTimeSeriesDataSet.created_train_frames.append(df.copy())
        FakeTimeSeriesDataSet.created_train_kwargs.append(dict(kwargs))

    @classmethod
    def from_dataset(cls, dataset, df, **kwargs):
        obj = cls.__new__(cls)
        obj.df = df
        obj.kwargs = kwargs
        return obj

    def to_dataloader(self, **kwargs):
        return object()


class FakeTFT:
    kwargs = None
    loaded_checkpoint = None

    @classmethod
    def from_dataset(cls, *args, **kwargs):
        cls.kwargs = kwargs
        return cls()

    @classmethod
    def load_from_checkpoint(cls, path):
        cls.loaded_checkpoint = path
        return cls()


class FakeTrainer:
    kwargs = None

    def __init__(self, *args, **kwargs):
        FakeTrainer.kwargs = kwargs
        self.current_epoch = 0
        self.num_training_batches = kwargs.get("limit_train_batches", 1)
        self.num_val_batches = [kwargs.get("limit_val_batches", 1)]
        self.callback_metrics = {"val_loss": 0.25}
        self.sanity_checking = False

    def fit(self, *args, **kwargs):
        for callback in FakeTrainer.kwargs.get("callbacks", []):
            hook = getattr(callback, "on_train_batch_end", None)
            if hook is not None:
                for batch_idx in range(int(self.num_training_batches)):
                    hook(self, args[0], None, None, batch_idx)
            hook = getattr(callback, "on_validation_batch_end", None)
            if hook is not None:
                for batch_idx in range(int(self.num_val_batches[0])):
                    hook(self, args[0], None, None, batch_idx, 0)
            hook = getattr(callback, "on_validation_epoch_end", None)
            if hook is not None:
                hook(self, args[0])
        return None


class FakeCheckpoint:
    def __init__(self, dirpath, *args, **kwargs):
        self.best_model_path = str(Path(dirpath) / "best.ckpt")
        self.best_model_score = 0.25
        Path(self.best_model_path).write_text("fake", encoding="ascii")


def _install_training_fakes(monkeypatch):
    FakeTimeSeriesDataSet.created_train_frames = []
    FakeTimeSeriesDataSet.created_train_kwargs = []
    FakeTrainer.kwargs = None
    torch_mod = types.ModuleType("torch")
    torch_mod.deterministic_calls = []
    torch_mod.use_deterministic_algorithms = (
        lambda enabled, **kwargs: torch_mod.deterministic_calls.append(
            (enabled, kwargs)
        )
    )
    torch_mod.__version__ = "2.fake"
    torch_mod.version = types.SimpleNamespace(cuda="12.fake")
    torch_mod.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(version=lambda: 8900)
    )
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda *_args, **_kwargs: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
        memory_allocated=lambda: 0,
    )
    lightning_pkg = types.ModuleType("lightning")
    pl_mod = types.ModuleType("lightning.pytorch")
    callbacks_mod = types.ModuleType("lightning.pytorch.callbacks")
    pf_mod = types.ModuleType("pytorch_forecasting")
    pf_mod.__version__ = "1.fake"
    metrics_mod = types.ModuleType("pytorch_forecasting.metrics")
    data_mod = types.ModuleType("pytorch_forecasting.data")
    encoders_mod = types.ModuleType("pytorch_forecasting.data.encoders")

    pl_mod.seed_everything = lambda *args, **kwargs: None
    pl_mod.__version__ = "2.lightning"
    pl_mod.Callback = object
    pl_mod.Trainer = FakeTrainer
    callbacks_mod.EarlyStopping = lambda *args, **kwargs: object()
    callbacks_mod.ModelCheckpoint = FakeCheckpoint
    pf_mod.TemporalFusionTransformer = FakeTFT
    pf_mod.TimeSeriesDataSet = FakeTimeSeriesDataSet
    metrics_mod.QuantileLoss = lambda *args, **kwargs: object()
    encoders_mod.NaNLabelEncoder = lambda *args, **kwargs: object()
    encoders_mod.TorchNormalizer = lambda *args, **kwargs: object()

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "lightning", lightning_pkg)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pl_mod)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", pf_mod)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting.data", data_mod)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting.data.encoders", encoders_mod)

    def prediction_frame(_self, _checkpoint, _training, frame, _batch_size):
        decoder = frame.sort_values("local_idx").tail(48)
        return pd.DataFrame(
            {
                "init_time": pd.to_datetime(
                    decoder["init_time_str"], utc=True
                ),
                "valid_time": decoder["valid_time"].to_numpy(),
                "fxx": decoder["fxx"].to_numpy(),
                "target": decoder["temp_error"].to_numpy(),
                "p10": decoder["temp_error"].to_numpy() - 1.0,
                "p50": decoder["temp_error"].to_numpy(),
                "p90": decoder["temp_error"].to_numpy() + 1.0,
                "base_temp_c": decoder["TMP_2m"].to_numpy(),
                "weight": decoder["decoder_weight"].to_numpy(),
            }
        )

    monkeypatch.setattr(
        TftTrainer,
        "_validation_prediction_frame",
        prediction_frame,
        raising=False,
    )


def _final_approval(
    snapshot: ResolvedTrialSnapshot | None = None,
) -> FinalEvaluationApproval:
    snapshot = snapshot or _trial_snapshot()
    return FinalEvaluationApproval.from_catalog_record(
        {
            "approval_id": "approval-test",
            "study_id": snapshot.study_id,
            "selected_trial_number": snapshot.trial_number,
            "trial_snapshot_hash": snapshot.snapshot_hash(),
            "approved_by": "pytest",
            "approved_at": "2026-06-29T00:00:00+00:00",
        }
    )


def _trial_snapshot(
    template: ExperimentTemplate | None = None,
) -> ResolvedTrialSnapshot:
    template = template or _experiment_template()
    parameters = {
        "hidden_size": template.model.hidden_size,
        "attention_head_size": template.model.attention_head_size,
        "hidden_continuous_size": template.model.hidden_continuous_size,
        "dropout": template.model.dropout,
        "learning_rate": template.model.learning_rate,
        "weight_decay": template.model.weight_decay,
        "batch_size": template.trainer.batch_size,
        "gradient_clip_val": template.trainer.gradient_clip_val,
    }
    return ResolvedTrialSnapshot.from_dict(
        {
            "schema_version": 1,
            "study_id": "study-test",
            "trial_number": 1,
            "execution_type": "trial",
            "template_hash": template.config_hash(),
            "data_order_hash": "data-hash",
            "parameters": parameters,
        }
    )


def _experiment_template(
    use_history_target: bool = False,
    **model_overrides,
) -> ExperimentTemplate:
    payload = json.loads(
        Path("experiments/klga/tft_optuna_v1.json").read_text(encoding="ascii")
    )
    payload["splits"] = {
        "train": ["2026-01-01", "2026-01-01"],
        "val": ["2026-01-02", "2026-01-02"],
        "test": ["2026-01-03", "2026-01-03"],
    }
    if use_history_target:
        payload["schema_version"] = 2
        payload["model"]["use_history_target"] = True
    payload["model"].update(model_overrides)
    return ExperimentTemplate.from_dict(payload)


def test_training_interfaces_live_in_training_factory():
    assert TftTrainer.__module__ == "training.trainer"
    assert ExperimentTemplate.__module__ == "training.config"


def test_eval_mae_excludes_zero_weight_tail_targets():
    torch = pytest.importorskip("torch")
    prediction = types.SimpleNamespace(
        output=torch.tensor([[1.0, 10.0]]),
        y=(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ),
    )
    model = types.SimpleNamespace(predict=lambda *_args, **_kwargs: prediction)
    trainer = TftTrainer.__new__(TftTrainer)

    mae = trainer._eval_mae(None, model, object())

    assert mae == pytest.approx(0.0)


def test_eval_mae_moves_prediction_targets_to_prediction_device():
    torch = pytest.importorskip("torch")

    class DeviceBoundTarget:
        def __init__(self, value):
            self.value = value
            self.moved = False

        def to(self, reference):
            self.moved = True
            return self.value.to(reference)

    target = DeviceBoundTarget(torch.tensor([[1.0, 2.0]]))
    prediction = types.SimpleNamespace(
        output=torch.tensor([[1.0, 2.0]]),
        y=(target, torch.tensor([[1.0, 1.0]])),
    )
    model = types.SimpleNamespace(predict=lambda *_args, **_kwargs: prediction)
    trainer = TftTrainer.__new__(TftTrainer)

    mae = trainer._eval_mae(None, model, object())

    assert target.moved is True
    assert mae == pytest.approx(0.0)


def test_production_tft_does_not_import_training_module():
    source = Path("model/tft.py").read_text(encoding="utf-8")

    assert "model.training" not in source


def test_tft_trainer_uses_settings_splits_and_publishes_dataset_pkl(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    settings = Settings(
        data_store_dir=tmp_path / "data",
        train_start="2026-01-01",
        train_end="2026-01-01",
        val_start="2026-01-02",
        val_end="2026-01-02",
        test_start="2026-01-03",
        test_end="2026-01-03",
    )
    builder = FakeBuilder()
    registry = FakeRegistry()
    monkeypatch.setattr(TftTrainer, "_eval_mae", lambda *args, **kwargs: 1.23)

    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        experiment_config=_experiment_template(),
    )

    result = trainer.run_final(
        station_id="KLGA",
        version="vtest",
        approval=_final_approval(),
        trial_snapshot=_trial_snapshot(),
    )

    assert [call[3] for call in builder.calls] == ["train", "val", "test"]
    assert builder.calls[0][1] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    assert builder.calls[0][2] == pd.Timestamp("2026-01-01 23:00:00", tz="UTC")
    assert FakeTimeSeriesDataSet.created_train_frames[0]["split"].eq("train").all()
    assert FakeTimeSeriesDataSet.created_train_kwargs[0]["weight"] == "decoder_weight"
    assert registry.publish_kwargs["training_dataset_pkl"].exists()
    assert registry.publish_kwargs["metrics"]["train_mae"] != registry.publish_kwargs["metrics"]["best_val_loss"]
    assert result["test_mae"] == 1.23


def test_run_final_rejects_non_finite_test_mae(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=FakeBuilder(),
        station_config=STATIONS["KLGA"],
        experiment_config=_experiment_template(),
    )
    monkeypatch.setattr(
        TftTrainer,
        "_eval_mae",
        lambda *_args, **_kwargs: float("nan"),
    )

    with pytest.raises(RuntimeError, match="finite test MAE"):
        trainer.run_final(
            station_id="KLGA",
            version="invalid-test-metrics",
            approval=_final_approval(),
            trial_snapshot=_trial_snapshot(),
        )


def test_tft_trainer_reuses_source_validated_audited_frames(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    settings = Settings(
        data_store_dir=tmp_path / "data",
        train_start="2026-01-01",
        train_end="2026-01-01",
        val_start="2026-01-02",
        val_end="2026-01-02",
        test_start="2026-01-03",
        test_end="2026-01-03",
    )
    cache = AuditedFrameCache(tmp_path / "audit_frames", settings.data_store_dir)
    source_builder = FakeBuilder()
    for split, start, end in (
        ("train", settings.train_start, settings.train_end),
        ("val", settings.val_start, settings.val_end),
        ("test", settings.test_start, settings.test_end),
    ):
        init_times = list(
            pd.date_range(
                f"{start} 00:00:00",
                f"{end} 23:00:00",
                freq="h",
                tz="UTC",
            )
        )
        cache.save(
            split,
            "KLGA",
            start,
            end,
            source_builder.build_frame(STATIONS["KLGA"], init_times, split),
        )
    unused_builder = FakeBuilder()
    registry = FakeRegistry()
    monkeypatch.setattr(TftTrainer, "_eval_mae", lambda *args, **kwargs: 1.23)
    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=unused_builder,
        station_config=STATIONS["KLGA"],
        audited_frame_cache=cache,
    )

    result = trainer.run_validation(
        station_id="KLGA",
        execution_id="cached",
    )

    assert unused_builder.calls == []
    assert "test_mae" not in result


def test_validation_execution_never_loads_test_or_publishes_candidate(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    settings = Settings(
        data_store_dir=tmp_path / "data",
        train_start="2026-01-01",
        train_end="2026-01-01",
        val_start="2026-01-02",
        val_end="2026-01-02",
        test_start="2026-01-03",
        test_end="2026-01-03",
    )
    builder = FakeBuilder()
    registry = FakeRegistry()
    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        experiment_config=_experiment_template(),
    )

    result = trainer.run_validation(
        station_id="KLGA",
        execution_id="trial-0001",
    )

    assert [call[3] for call in builder.calls] == ["train", "val"]
    assert "test_mae" not in result
    assert result["best_val_loss"] == 0.25
    assert registry.publish_kwargs is None


def test_target_winsor_does_not_mutate_history_target_feature(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)

    class ExtremeBuilder(FakeBuilder):
        def build_frame(self, station, init_times, split):
            frame = super().build_frame(station, init_times, split)
            frame["temp_error"] = 12.0
            frame["hist_temp_error"] = 12.0
            return frame

    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(
            data_store_dir=tmp_path / "data",
            tft_target_winsor=8.0,
        ),
        dataset_builder=ExtremeBuilder(),
        station_config=STATIONS["KLGA"],
    )

    trainer.run_validation(station_id="KLGA", execution_id="winsor")

    training_frame = FakeTimeSeriesDataSet.created_train_frames[0]
    assert training_frame["temp_error"].eq(8.0).all()
    assert training_frame["hist_temp_error"].eq(12.0).all()


def test_history_target_experiment_flag_controls_tft_inputs(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    baseline = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "baseline",
        settings=Settings(data_store_dir=tmp_path / "data-a"),
        dataset_builder=FakeBuilder(),
        station_config=STATIONS["KLGA"],
        experiment_config=_experiment_template(use_history_target=False),
    )
    baseline.run_validation(station_id="KLGA", execution_id="a")
    baseline_unknown = FakeTimeSeriesDataSet.created_train_kwargs[-1][
        "time_varying_unknown_reals"
    ]

    residual = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "residual",
        settings=Settings(data_store_dir=tmp_path / "data-b"),
        dataset_builder=FakeBuilder(),
        station_config=STATIONS["KLGA"],
        experiment_config=_experiment_template(use_history_target=True),
    )
    residual.run_validation(station_id="KLGA", execution_id="b")
    residual_unknown = FakeTimeSeriesDataSet.created_train_kwargs[-1][
        "time_varying_unknown_reals"
    ]

    assert "hist_temp_error" not in baseline_unknown
    assert "hist_temp_error" in residual_unknown


def test_smoke_execution_applies_hard_training_limits(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    experiment = _experiment_template()
    settings = Settings(data_store_dir=tmp_path / "data")
    builder = FakeBuilder()
    registry = FakeRegistry()
    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        experiment_config=experiment,
    )

    result = trainer.run_validation(
        station_id="KLGA",
        execution_id="smoke-0001",
        limits=ValidationExecutionLimits.for_smoke(experiment.execution.smoke),
    )

    assert FakeTrainer.kwargs["max_epochs"] == 1
    assert FakeTrainer.kwargs["min_epochs"] == 1
    assert FakeTrainer.kwargs["limit_train_batches"] == 20
    assert FakeTrainer.kwargs["limit_val_batches"] == 5
    assert FakeTrainer.kwargs["max_time"].total_seconds() == 30 * 60
    assert FakeTrainer.kwargs["logger"] is False
    assert FakeTFT.kwargs["mask_bias"] == -float("inf")
    assert sys.modules["torch"].deterministic_calls == [
        (True, {"warn_only": True})
    ]
    assert [call[3] for call in builder.calls] == ["train", "val"]
    assert registry.publish_kwargs is None
    assert result["execution_type"] == "smoke"
    assert result["environment"]["deterministic_mode"] == "warn_only"


def test_smoke_reports_live_batch_progress_and_verifies_checkpoint_reload(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    experiment = _experiment_template()
    progress = []
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=FakeBuilder(),
        station_config=STATIONS["KLGA"],
        experiment_config=experiment,
    )

    result = trainer.run_validation(
        station_id="KLGA",
        execution_id="smoke-progress",
        limits=ValidationExecutionLimits.for_smoke(experiment.execution.smoke),
        progress_reporter=progress.append,
    )

    assert any(item["phase"] == "fit" for item in progress)
    assert any(item.get("train_batches_completed") == 1 for item in progress)
    assert any(item.get("val_batches_completed") == 1 for item in progress)
    assert [
        item["train_batches_completed"]
        for item in progress
        if "train_batches_completed" in item
    ] == [1, 10, 20]
    assert [
        item["val_batches_completed"]
        for item in progress
        if "val_batches_completed" in item
    ] == [1, 5]
    assert any(
        item.get("validation_quantile_loss") == 0.25
        for item in progress
    )
    assert FakeTFT.loaded_checkpoint == result["checkpoint_path"]
    assert result["checkpoint_reload_verified"] is True
    assert result["environment"]["cuda_available"] is False


def test_trial_execution_applies_resolved_search_parameters(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    experiment = _experiment_template()
    parameters = {
        "hidden_size": 64,
        "attention_head_size": 2,
        "hidden_continuous_size": 24,
        "dropout": 0.35,
        "learning_rate": 0.001,
        "weight_decay": 0.002,
        "batch_size": 8,
        "gradient_clip_val": 0.5,
    }
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=FakeBuilder(),
        station_config=STATIONS["KLGA"],
        experiment_config=experiment,
    )

    trainer.run_validation(
        station_id="KLGA",
        execution_id="trial-00001",
        limits=ValidationExecutionLimits.for_trial(
            experiment,
            parameters,
        ),
        trial_parameters=parameters,
    )

    assert FakeTFT.kwargs["hidden_size"] == 64
    assert FakeTFT.kwargs["attention_head_size"] == 2
    assert FakeTFT.kwargs["hidden_continuous_size"] == 24
    assert FakeTFT.kwargs["dropout"] == 0.35
    assert FakeTFT.kwargs["learning_rate"] == 0.001
    assert FakeTFT.kwargs["weight_decay"] == 0.002
    assert FakeTrainer.kwargs["gradient_clip_val"] == 0.5


def test_final_execution_rejects_missing_approval_before_loading_data(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    builder = FakeBuilder()
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
    )

    with pytest.raises(PermissionError, match="final approval"):
        trainer.run_final(
            station_id="KLGA",
            version="candidate-v1",
            approval=None,
        )

    assert builder.calls == []


def test_final_execution_rejects_unverified_approval_object(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    builder = FakeBuilder()
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
    )

    with pytest.raises(PermissionError, match="verified final approval"):
        trainer.run_final(
            station_id="KLGA",
            version="candidate-v1",
            approval=object(),
        )

    assert builder.calls == []


def test_final_execution_rejects_approval_for_another_trial_snapshot(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    builder = FakeBuilder()
    approval = _final_approval()
    mismatched_snapshot = ResolvedTrialSnapshot.from_dict(
        {
            **_trial_snapshot().to_dict(),
            "trial_number": 2,
        }
    )
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
    )

    with pytest.raises(PermissionError, match="does not match selected trial"):
        trainer.run_final(
            station_id="KLGA",
            version="candidate-v1",
            approval=approval,
            trial_snapshot=mismatched_snapshot,
        )

    assert builder.calls == []


def test_approved_final_evaluates_test_and_publishes_inactive_candidate(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    settings = Settings(
        data_store_dir=tmp_path / "data",
        train_start="2026-01-01",
        train_end="2026-01-01",
        val_start="2026-01-02",
        val_end="2026-01-02",
        test_start="2026-01-03",
        test_end="2026-01-03",
    )
    builder = FakeBuilder()
    registry = FakeRegistry()
    experiment = _experiment_template(use_history_target=True)
    snapshot = _trial_snapshot(experiment)
    monkeypatch.setattr(TftTrainer, "_eval_mae", lambda *args, **kwargs: 1.23)
    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        experiment_config=experiment,
    )
    approval = _final_approval(snapshot)

    result = trainer.run_final(
        station_id="KLGA",
        version="candidate-v1",
        approval=approval,
        trial_snapshot=snapshot,
    )

    assert [call[3] for call in builder.calls] == ["train", "val", "test"]
    assert result["test_mae"] == 1.23
    assert registry.publish_kwargs["activate"] is False
    artifact_names = {
        path.name for path in registry.publish_kwargs["extra_artifacts"]
    }
    assert {
        "mc_calibration.npz",
        "mc_calibration.json",
        "mc_calibration_evidence.json",
        "validation_predictions.parquet",
    } <= artifact_names


def test_final_replays_the_selected_trial_parameters(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    experiment = _experiment_template()
    snapshot_payload = _trial_snapshot(experiment).to_dict()
    snapshot_payload["parameters"].update(
        {
            "hidden_size": 64,
            "attention_head_size": 2,
            "hidden_continuous_size": 24,
            "dropout": 0.35,
            "learning_rate": 0.001,
            "weight_decay": 0.002,
            "batch_size": 8,
            "gradient_clip_val": 0.5,
        }
    )
    snapshot = ResolvedTrialSnapshot.from_dict(snapshot_payload)
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=FakeBuilder(),
        station_config=STATIONS["KLGA"],
        experiment_config=experiment,
    )
    monkeypatch.setattr(TftTrainer, "_eval_mae", lambda *_args: 1.23)

    trainer.run_final(
        station_id="KLGA",
        version="selected-trial",
        approval=_final_approval(snapshot),
        trial_snapshot=snapshot,
    )

    assert FakeTFT.kwargs["hidden_size"] == 64
    assert FakeTFT.kwargs["dropout"] == 0.35
    assert FakeTrainer.kwargs["gradient_clip_val"] == 0.5


def test_legacy_run_entrypoint_cannot_bypass_test_lock(
    tmp_path,
    monkeypatch,
):
    _install_training_fakes(monkeypatch)
    builder = FakeBuilder()
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=Settings(data_store_dir=tmp_path / "data"),
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
    )

    with pytest.raises(PermissionError, match="run_validation.*run_final"):
        trainer.run(station_id="KLGA", version="bypass")

    assert builder.calls == []


def test_tft_trainer_rejects_missing_or_stale_audited_frame(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    settings = Settings(
        data_store_dir=tmp_path / "data",
        train_start="2026-01-01",
        train_end="2026-01-01",
        val_start="2026-01-02",
        val_end="2026-01-02",
        test_start="2026-01-03",
        test_end="2026-01-03",
    )
    builder = FakeBuilder()
    trainer = TftTrainer(
        store=None,
        registry=FakeRegistry(),
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        audited_frame_cache=AuditedFrameCache(
            tmp_path / "audit_frames",
            settings.data_store_dir,
        ),
    )

    with pytest.raises(RuntimeError, match="rerun scripts.audit_training_data"):
        trainer.run_validation(station_id="KLGA", execution_id="stale")

    assert builder.calls == []


def test_tft_trainer_writes_training_metadata_and_result_json(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    settings = Settings(
        data_store_dir=tmp_path / "data",
        train_start="2026-01-01",
        train_end="2026-01-01",
        val_start="2026-01-02",
        val_end="2026-01-02",
        test_start="2026-01-03",
        test_end="2026-01-03",
    )
    builder = FakeBuilder()
    registry = FakeRegistry()
    monkeypatch.setattr(TftTrainer, "_eval_mae", lambda *args, **kwargs: 1.23)

    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        random_seed=7,
        torch_deterministic=True,
        precision="32",
        num_workers=2,
        experiment_config=_experiment_template(),
    )

    result = trainer.run_final(
        station_id="KLGA",
        version="vmeta",
        approval=_final_approval(),
        trial_snapshot=_trial_snapshot(),
    )

    version_dir = tmp_path / "checkpoints" / "vmeta"
    metadata = json.loads((version_dir / "training_metadata.json").read_text())
    result_json = json.loads((version_dir / "train_result.json").read_text())
    experiment = json.loads((version_dir / "experiment_config.json").read_text())
    assert registry.publish_kwargs["activate"] is False
    assert metadata["random_seed"] == 42
    assert metadata["torch_deterministic"] is True
    assert metadata["precision"] == "16-mixed"
    assert metadata["num_workers"] == 0
    assert metadata["train_sample_count"] == 1
    assert metadata["val_sample_count"] == 1
    assert metadata["test_sample_count"] == 1
    assert metadata["train_row_count"] == 120
    assert metadata["feature_list_and_order"]["known_reals"] == metadata["known_reals"]
    assert metadata["target"] == "temp_error"
    assert metadata["data_order_hash"]
    assert result_json["version"] == result["version"]
    assert experiment["runtime_profile"] == "john_linux_rtx2060"


def test_tft_trainer_uses_experiment_config_and_publishes_contract(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    experiment = _experiment_template(
        hidden_size=64,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=8,
        learning_rate=0.001,
        weight_decay=0.002,
        reduce_lr_patience=4,
        target_winsor=6.0,
    )
    settings = Settings(data_store_dir=tmp_path / "data")
    experiment.apply_to_settings(settings)
    builder = FakeBuilder()
    registry = FakeRegistry()
    monkeypatch.setattr(TftTrainer, "_eval_mae", lambda *args, **kwargs: 1.23)

    trainer = TftTrainer(
        store=None,
        registry=registry,
        output_dir=tmp_path / "checkpoints",
        settings=settings,
        dataset_builder=builder,
        station_config=STATIONS["KLGA"],
        experiment_config=experiment,
    )

    snapshot = _trial_snapshot(experiment)
    trainer.run_final(
        station_id="KLGA",
        version="vexp",
        approval=_final_approval(snapshot),
        trial_snapshot=snapshot,
    )

    version_dir = tmp_path / "checkpoints" / "vexp"
    written_config = json.loads((version_dir / "experiment_config.json").read_text())
    contract = json.loads((version_dir / "model_contract.json").read_text())
    assert [call[3] for call in builder.calls] == ["train", "val", "test"]
    assert FakeTFT.kwargs["hidden_size"] == 64
    assert FakeTFT.kwargs["learning_rate"] == 0.001
    assert written_config["runtime_profile"] == "john_linux_rtx2060"
    assert contract["station_id"] == "KLGA"
    assert contract["max_encoder_length"] == 72
    assert contract["max_prediction_length"] == 48
    assert contract["quantiles"] == [0.1, 0.5, 0.9]
    artifact_names = {path.name for path in registry.publish_kwargs["extra_artifacts"]}
    assert {"experiment_config.json", "model_contract.json"} <= artifact_names
