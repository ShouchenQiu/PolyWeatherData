from __future__ import annotations

import sys
import types
from pathlib import Path
import json

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from scripts.train_tft import TftTrainer, _settings_from_cli_args


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

    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
        self.target_normalizer = kwargs.get("target_normalizer")
        FakeTimeSeriesDataSet.created_train_frames.append(df.copy())

    @classmethod
    def from_dataset(cls, dataset, df, **kwargs):
        obj = cls.__new__(cls)
        obj.df = df
        obj.kwargs = kwargs
        return obj

    def to_dataloader(self, **kwargs):
        return object()


class FakeTFT:
    @classmethod
    def from_dataset(cls, *args, **kwargs):
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
    assert registry.publish_kwargs["training_dataset_pkl"].exists()
    assert registry.publish_kwargs["metrics"]["train_mae"] != registry.publish_kwargs["metrics"]["best_val_loss"]
    assert result["test_mae"] == 1.23


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
        activate_model=False,
    )

    result = trainer.run(station_id="KLGA", version="vmeta")

    version_dir = tmp_path / "checkpoints" / "vmeta"
    metadata = json.loads((version_dir / "training_metadata.json").read_text())
    result_json = json.loads((version_dir / "train_result.json").read_text())
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


def test_settings_from_cli_args_applies_smoke_summary_and_paths(tmp_path):
    summary_path = tmp_path / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "data_store_dir": str(tmp_path / "summary_data"),
                "smoke_split": {
                    "train": ["2026-05-26", "2026-05-30"],
                    "val": ["2026-05-31", "2026-06-01"],
                    "test": ["2026-06-02", "2026-06-04"],
                },
            }
        ),
        encoding="ascii",
    )
    args = types.SimpleNamespace(
        data_store=tmp_path / "data",
        model_store=tmp_path / "models",
        smoke_summary=summary_path,
        min_epochs=1,
        max_epochs=2,
        batch_size=16,
        seed=7,
        deterministic=True,
        precision="bf16-mixed",
        num_workers=2,
    )

    settings = _settings_from_cli_args(args)

    assert settings.data_store_dir == tmp_path / "data"
    assert settings.forecast_store_dir == tmp_path / "data" / "hrrr"
    assert settings.obs_store_dir == tmp_path / "data" / "obs"
    assert settings.training_decoder_cache_dir == tmp_path / "data" / "_decoder_cache"
    assert settings.model_store_dir == tmp_path / "models"
    assert (settings.train_start, settings.train_end) == ("2026-05-26", "2026-05-30")
    assert (settings.val_start, settings.val_end) == ("2026-05-31", "2026-06-01")
    assert (settings.test_start, settings.test_end) == ("2026-06-02", "2026-06-04")
    assert settings.tft_min_epochs == 1
    assert settings.tft_max_epochs == 2
    assert settings.tft_batch_size == 16
