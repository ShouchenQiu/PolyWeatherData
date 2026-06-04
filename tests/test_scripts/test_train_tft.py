from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from scripts.train_tft import TftTrainer


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
    lightning_pkg = types.ModuleType("lightning")
    pl_mod = types.ModuleType("lightning.pytorch")
    callbacks_mod = types.ModuleType("lightning.pytorch.callbacks")
    pf_mod = types.ModuleType("pytorch_forecasting")
    metrics_mod = types.ModuleType("pytorch_forecasting.metrics")
    data_mod = types.ModuleType("pytorch_forecasting.data")
    encoders_mod = types.ModuleType("pytorch_forecasting.data.encoders")

    pl_mod.seed_everything = lambda *args, **kwargs: None
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
