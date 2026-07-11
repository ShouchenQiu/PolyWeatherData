"""Discrete settlement-temperature distributions for daily-high models."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


PMF_ARTIFACT_SCHEMA_VERSION = 1
SUPPORT_MIN_F = -100
SUPPORT_MAX_F = 160


class DailyHighPmfNetwork(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, class_count),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


@dataclass(frozen=True)
class TemperaturePmf:
    support_f: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        support = np.asarray(self.support_f, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if support.ndim != 1 or len(support) == 0:
            raise ValueError("PMF support must be a non-empty vector")
        if probabilities.shape != support.shape:
            raise ValueError("PMF probabilities must match support")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("PMF probabilities must be finite and non-negative")
        if not np.all(np.diff(support) == 1.0):
            raise ValueError("PMF support must use consecutive Fahrenheit integers")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise ValueError("PMF probabilities must sum to one")
        object.__setattr__(self, "support_f", support)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def from_logits(
        cls,
        logits: np.ndarray,
        *,
        support_min_f: int = -100,
        observed_high_f: float | None = None,
    ) -> "TemperaturePmf":
        values = np.asarray(logits, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("PMF logits must be one finite non-empty vector")
        support = np.arange(
            int(support_min_f), int(support_min_f) + len(values), dtype=float
        )
        if observed_high_f is not None and np.isfinite(observed_high_f):
            minimum = float(np.floor(float(observed_high_f) + 0.5))
            values = values.copy()
            values[support < minimum] = -np.inf
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError("observed high excludes the entire PMF support")
        shifted = values - np.max(values[finite])
        weights = np.zeros_like(values)
        weights[finite] = np.exp(shifted[finite])
        return cls(support, weights / weights.sum())

    def probability_for_bucket(self, lo_f: float, hi_f: float) -> float:
        if lo_f >= hi_f:
            raise ValueError("bucket lower bound must be below upper bound")
        mask = (self.support_f >= lo_f) & (self.support_f < hi_f)
        return float(self.probabilities[mask].sum())

    def quantile(self, q: float) -> float:
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError("quantile must be between zero and one")
        index = int(np.searchsorted(np.cumsum(self.probabilities), q, side="left"))
        return float(self.support_f[min(index, len(self.support_f) - 1)])

    @property
    def p10_f(self) -> float:
        return self.quantile(0.1)

    @property
    def p50_f(self) -> float:
        return self.quantile(0.5)

    @property
    def p90_f(self) -> float:
        return self.quantile(0.9)

    def samples(self, *, seed: int, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("sample count must be positive")
        return np.random.default_rng(seed).choice(
            self.support_f, size=count, p=self.probabilities
        )


class DailyHighPmfModel:
    """Fixed-contract MLP that predicts settlement-temperature PMFs."""

    def __init__(
        self,
        *,
        network: DailyHighPmfNetwork,
        feature_columns: tuple[str, ...],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        tft_checkpoint_sha256: str,
        training_manifest_sha256: str,
    ) -> None:
        self._network = network.eval()
        self.feature_columns = tuple(feature_columns)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.tft_checkpoint_sha256 = _sha256_value(
            tft_checkpoint_sha256, "TFT checkpoint"
        )
        self.training_manifest_sha256 = _sha256_value(
            training_manifest_sha256, "training manifest"
        )
        if len(self.feature_columns) == 0:
            raise ValueError("PMF model requires at least one feature")
        if self.feature_mean.shape != (len(self.feature_columns),):
            raise ValueError("feature mean shape does not match feature contract")
        if self.feature_scale.shape != self.feature_mean.shape:
            raise ValueError("feature scale shape does not match feature contract")
        if not np.isfinite(self.feature_mean).all():
            raise ValueError("feature mean must be finite")
        if not np.isfinite(self.feature_scale).all() or np.any(
            self.feature_scale <= 0
        ):
            raise ValueError("feature scale must be finite and positive")

    def predict(self, row: Mapping[str, object]) -> TemperaturePmf:
        missing = [name for name in self.feature_columns if name not in row]
        if missing:
            raise ValueError("PMF inference features missing: " + ", ".join(missing))
        values = np.array(
            [float(row[name]) for name in self.feature_columns], dtype=np.float32
        )
        if not np.isfinite(values).all():
            raise ValueError("PMF inference features must be finite")
        normalized = (values - self.feature_mean) / self.feature_scale
        with torch.no_grad():
            logits = self._network(torch.from_numpy(normalized)[None, :])[0]
        observed_high = None
        if float(row.get("observed_high_mask", 0.0)) > 0:
            observed_high = float(row["observed_high_f"])
        return TemperaturePmf.from_logits(
            logits.numpy(),
            support_min_f=SUPPORT_MIN_F,
            observed_high_f=observed_high,
        )

    def save(self, root: Path | str) -> dict[str, object]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        weights_path = directory / "daily_high_pmf.pt"
        standardizer_path = directory / "standardizer.npz"
        torch.save(self._network.state_dict(), weights_path)
        np.savez_compressed(
            standardizer_path,
            mean=self.feature_mean,
            scale=self.feature_scale,
        )
        metadata = {
            "schema_version": PMF_ARTIFACT_SCHEMA_VERSION,
            "distribution_source": "daily_high_pmf_pilot",
            "support_min_f": SUPPORT_MIN_F,
            "support_max_f": SUPPORT_MAX_F,
            "feature_columns": list(self.feature_columns),
            "feature_contract_sha256": _json_sha256(list(self.feature_columns)),
            "tft_checkpoint_sha256": self.tft_checkpoint_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
            "sha256": {
                weights_path.name: _file_sha256(weights_path),
                standardizer_path.name: _file_sha256(standardizer_path),
            },
        }
        (directory / "daily_high_pmf.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="ascii"
        )
        return metadata

    @classmethod
    def load(
        cls,
        root: Path | str,
        *,
        expected_tft_checkpoint_sha256: str,
    ) -> "DailyHighPmfModel":
        directory = Path(root)
        metadata = json.loads(
            (directory / "daily_high_pmf.json").read_text(encoding="ascii")
        )
        if metadata.get("schema_version") != PMF_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported daily-high PMF artifact schema")
        if metadata.get("tft_checkpoint_sha256") != expected_tft_checkpoint_sha256:
            raise ValueError("TFT checkpoint hash mismatch")
        features = tuple(str(value) for value in metadata["feature_columns"])
        if metadata.get("feature_contract_sha256") != _json_sha256(list(features)):
            raise ValueError("PMF feature contract hash mismatch")
        for filename, expected in metadata["sha256"].items():
            if _file_sha256(directory / filename) != expected:
                raise ValueError(f"PMF artifact checksum mismatch: {filename}")
        with np.load(directory / "standardizer.npz", allow_pickle=False) as arrays:
            mean = arrays["mean"]
            scale = arrays["scale"]
        network = DailyHighPmfNetwork(
            len(features), SUPPORT_MAX_F - SUPPORT_MIN_F + 1
        )
        network.load_state_dict(
            torch.load(
                directory / "daily_high_pmf.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        return cls(
            network=network,
            feature_columns=features,
            feature_mean=mean,
            feature_scale=scale,
            tft_checkpoint_sha256=str(metadata["tft_checkpoint_sha256"]),
            training_manifest_sha256=str(metadata["training_manifest_sha256"]),
        )


def _sha256_value(value: str, name: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{name} hash must be a SHA-256 hex digest")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
