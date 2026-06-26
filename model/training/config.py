"""Training experiment configuration for reproducible TFT runs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.20
    hidden_continuous_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    reduce_lr_patience: int = 3
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    target_winsor: float = 8.0
    max_encoder_length: int = 72
    max_prediction_length: int = 48


@dataclass
class TrainerConfig:
    min_epochs: int = 5
    max_epochs: int = 20
    batch_size: int = 32
    seed: int = 42
    deterministic: bool = True
    precision: str = "32"
    num_workers: int = 0
    early_stop_patience: int = 2
    gradient_clip_val: float = 0.1


@dataclass
class TrainingExperimentConfig:
    """Serializable experiment contract shared by lab training machines."""

    schema_version: int = 1
    station: str = "KLGA"
    splits: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "train": ("2024-10-01", "2026-03-31"),
            "val": ("2026-04-01", "2026-04-30"),
            "test": ("2026-05-01", "2026-06-04"),
        }
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runtime_profile: str = "default"

    @classmethod
    def from_json(cls, path: Path | str) -> "TrainingExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="ascii"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingExperimentConfig":
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(f"unsupported training experiment schema_version={schema_version}")
        raw_splits = payload.get("splits") or {}
        splits = {
            name: _split_tuple(raw_splits.get(name), name)
            for name in ("train", "val", "test")
        }
        cfg = cls(
            schema_version=schema_version,
            station=str(payload.get("station", "KLGA")),
            splits=splits,
            model=ModelConfig(**(payload.get("model") or {})),
            trainer=TrainerConfig(**(payload.get("trainer") or {})),
            runtime_profile=str(payload.get("runtime_profile", "default")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if set(self.splits) != {"train", "val", "test"}:
            raise ValueError("splits must contain train, val, and test")
        quantiles = self.model.quantiles
        if not quantiles or quantiles != sorted(quantiles):
            raise ValueError("model.quantiles must be sorted and non-empty")
        if any(q <= 0 or q >= 1 for q in quantiles):
            raise ValueError("model.quantiles must be between 0 and 1")
        if self.model.max_encoder_length <= 0 or self.model.max_prediction_length <= 0:
            raise ValueError("encoder and prediction lengths must be positive")
        if self.trainer.min_epochs < 0 or self.trainer.max_epochs < self.trainer.min_epochs:
            raise ValueError("trainer epoch bounds are invalid")
        if self.trainer.batch_size <= 0:
            raise ValueError("trainer.batch_size must be positive")

    def apply_to_settings(self, settings: Any) -> None:
        settings.train_start, settings.train_end = self.splits["train"]
        settings.val_start, settings.val_end = self.splits["val"]
        settings.test_start, settings.test_end = self.splits["test"]
        settings.tft_hidden_size = self.model.hidden_size
        settings.tft_attention_head_size = self.model.attention_head_size
        settings.tft_dropout = self.model.dropout
        settings.tft_hidden_continuous_size = self.model.hidden_continuous_size
        settings.tft_learning_rate = self.model.learning_rate
        settings.tft_weight_decay = self.model.weight_decay
        settings.tft_reduce_lr_patience = self.model.reduce_lr_patience
        settings.tft_quantiles = list(self.model.quantiles)
        settings.tft_target_winsor = self.model.target_winsor
        settings.tft_max_encoder_length = self.model.max_encoder_length
        settings.tft_max_prediction_length = self.model.max_prediction_length
        settings.tft_min_epochs = self.trainer.min_epochs
        settings.tft_max_epochs = self.trainer.max_epochs
        settings.tft_batch_size = self.trainer.batch_size
        settings.tft_early_stop_patience = self.trainer.early_stop_patience

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["splits"] = {key: list(value) for key, value in self.splits.items()}
        return payload

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="ascii",
        )

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_tuple(value: Any, name: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"splits.{name} must be a [start, end] pair")
    return str(value[0]), str(value[1])
