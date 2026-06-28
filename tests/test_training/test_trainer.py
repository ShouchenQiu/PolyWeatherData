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
from training.config import TrainingExperimentConfig
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

    @classmethod
    def from_dataset(cls, *args, **kwargs):
        cls.kwargs = kwargs
        return cls()


class FakeTrainer:
    def __init__(self, *args, **kwargs):
        self.current_epoch = 3

    def fit(self, *args, **kwargs):
        return None


class FakeCheckpoint:
    def __init__(self, dirpath, *args, **kwargs):
        self.best_model_path = str(Path(dirpath) / "best.ckpt")
        self.best_model_score = 0.25
        Path(self.best_model_path).write_text("fake", encoding="ascii")


def _install_training_fakes(monkeypatch):
    FakeTimeSeriesDataSet.created_train_frames = []
    FakeTimeSeriesDataSet.created_train_kwargs = []
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.fake"
    torch_mod.version = types.SimpleNamespace(cuda="12.fake")
    torch_mod.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(version=lambda: 8900)
    )
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda *_args, **_kwargs: None,
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


def test_training_interfaces_live_in_training_factory():
    assert TftTrainer.__module__ == "training.trainer"
    assert TrainingExperimentConfig.__module__ == "training.config"


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
    )

    result = trainer.run(station_id="KLGA", version="vtest")

    assert [call[3] for call in builder.calls] == ["train", "val", "test"]
    assert builder.calls[0][1] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    assert builder.calls[0][2] == pd.Timestamp("2026-01-01 23:00:00", tz="UTC")
    assert FakeTimeSeriesDataSet.created_train_frames[0]["split"].eq("train").all()
    assert FakeTimeSeriesDataSet.created_train_kwargs[0]["weight"] == "decoder_weight"
    assert registry.publish_kwargs["training_dataset_pkl"].exists()
    assert registry.publish_kwargs["metrics"]["train_mae"] != registry.publish_kwargs["metrics"]["best_val_loss"]
    assert result["test_mae"] == 1.23


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

    result = trainer.run(station_id="KLGA", version="cached")

    assert unused_builder.calls == []
    assert result["test_mae"] == 1.23


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
        trainer.run(station_id="KLGA", version="stale")

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
    )

    result = trainer.run(station_id="KLGA", version="vmeta")

    version_dir = tmp_path / "checkpoints" / "vmeta"
    metadata = json.loads((version_dir / "training_metadata.json").read_text())
    result_json = json.loads((version_dir / "train_result.json").read_text())
    experiment = json.loads((version_dir / "experiment_config.json").read_text())
    assert registry.publish_kwargs["activate"] is False
    assert metadata["random_seed"] == 7
    assert metadata["torch_deterministic"] is True
    assert metadata["precision"] == "32"
    assert metadata["num_workers"] == 2
    assert metadata["train_sample_count"] == 1
    assert metadata["val_sample_count"] == 1
    assert metadata["test_sample_count"] == 1
    assert metadata["train_row_count"] == 120
    assert metadata["feature_list_and_order"]["known_reals"] == metadata["known_reals"]
    assert metadata["target"] == "temp_error"
    assert metadata["data_order_hash"]
    assert result_json["version"] == result["version"]
    assert experiment["runtime_profile"] == "inline_settings"


def test_tft_trainer_uses_experiment_config_and_publishes_contract(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "station": "KLGA",
                "splits": {
                    "train": ["2026-01-01", "2026-01-01"],
                    "val": ["2026-01-02", "2026-01-02"],
                    "test": ["2026-01-03", "2026-01-03"],
                },
                "model": {
                    "hidden_size": 64,
                    "attention_head_size": 2,
                    "dropout": 0.15,
                    "hidden_continuous_size": 8,
                    "learning_rate": 0.001,
                    "weight_decay": 0.002,
                    "reduce_lr_patience": 4,
                    "quantiles": [0.05, 0.5, 0.95],
                    "target_winsor": 6.0,
                    "max_encoder_length": 72,
                    "max_prediction_length": 48,
                },
                "trainer": {
                    "min_epochs": 2,
                    "max_epochs": 4,
                    "batch_size": 24,
                    "seed": 11,
                    "deterministic": True,
                    "precision": "32",
                    "num_workers": 1,
                    "early_stop_patience": 3,
                    "gradient_clip_val": 0.2,
                },
                "runtime_profile": "john_linux_2060",
            }
        ),
        encoding="ascii",
    )
    experiment = TrainingExperimentConfig.from_json(config_path)
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

    trainer.run(station_id="KLGA", version="vexp")

    version_dir = tmp_path / "checkpoints" / "vexp"
    written_config = json.loads((version_dir / "experiment_config.json").read_text())
    contract = json.loads((version_dir / "model_contract.json").read_text())
    assert [call[3] for call in builder.calls] == ["train", "val", "test"]
    assert FakeTFT.kwargs["hidden_size"] == 64
    assert FakeTFT.kwargs["learning_rate"] == 0.001
    assert written_config["runtime_profile"] == "john_linux_2060"
    assert contract["station_id"] == "KLGA"
    assert contract["max_encoder_length"] == 72
    assert contract["max_prediction_length"] == 48
    assert contract["quantiles"] == [0.05, 0.5, 0.95]
    artifact_names = {path.name for path in registry.publish_kwargs["extra_artifacts"]}
    assert {"experiment_config.json", "model_contract.json"} <= artifact_names
