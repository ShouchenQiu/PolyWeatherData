"""Low-dimensional future-maximum distributions for point-in-time daily highs."""
from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from torch import nn


SUPPORT_MIN_F = -100.0
SUPPORT_MAX_F = 160.0
_NORMAL = NormalDist()


class GaussianFutureMaxNetwork(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


@dataclass(frozen=True)
class TruncatedNormal:
    """A normal future-maximum distribution truncated to physical support."""

    mean_f: float
    sigma_f: float
    lower_f: float = SUPPORT_MIN_F
    upper_f: float = SUPPORT_MAX_F

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.mean_f, self.sigma_f, self.lower_f, self.upper_f
        )):
            raise ValueError("truncated-normal parameters must be finite")
        if self.sigma_f <= 0 or self.lower_f >= self.upper_f:
            raise ValueError("truncated-normal bounds and sigma are invalid")

    def cdf(self, value_f: float) -> float:
        if value_f <= self.lower_f:
            return 0.0
        if value_f >= self.upper_f:
            return 1.0
        lower = _NORMAL.cdf((self.lower_f - self.mean_f) / self.sigma_f)
        upper = _NORMAL.cdf((self.upper_f - self.mean_f) / self.sigma_f)
        raw = _NORMAL.cdf((float(value_f) - self.mean_f) / self.sigma_f)
        return float(np.clip((raw - lower) / (upper - lower), 0.0, 1.0))

    def quantile(self, probability: float) -> float:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("quantile must be between zero and one")
        lower = _NORMAL.cdf((self.lower_f - self.mean_f) / self.sigma_f)
        upper = _NORMAL.cdf((self.upper_f - self.mean_f) / self.sigma_f)
        raw = lower + probability * (upper - lower)
        raw = float(np.clip(raw, 1e-15, 1.0 - 1e-15))
        return float(np.clip(
            self.mean_f + self.sigma_f * _NORMAL.inv_cdf(raw),
            self.lower_f,
            self.upper_f,
        ))

    def samples(self, *, seed: int, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("sample count must be positive")
        rng = np.random.default_rng(seed)
        return np.array(
            [self.quantile(float(probability)) for probability in rng.random(count)],
            dtype=float,
        )


@dataclass(frozen=True)
class DailyHighGaussian:
    """Final-high distribution created from observed and future maxima."""

    future_max: TruncatedNormal | None
    observed_high_f: float | None = None

    def __post_init__(self) -> None:
        if self.future_max is None and self.observed_high_f is None:
            raise ValueError("daily high needs a future distribution or observed high")
        if self.observed_high_f is not None and not math.isfinite(self.observed_high_f):
            raise ValueError("observed high must be finite")

    def cdf(self, value_f: float) -> float:
        if self.future_max is None:
            return float(value_f >= float(self.observed_high_f))
        if self.observed_high_f is not None and value_f < self.observed_high_f:
            return 0.0
        return self.future_max.cdf(value_f)

    def probability_for_bucket(self, lo_f: float, hi_f: float) -> float:
        if lo_f >= hi_f:
            raise ValueError("bucket lower bound must be below upper bound")
        lower = float("-inf") if math.isinf(lo_f) and lo_f < 0 else lo_f - 0.5
        upper = float("inf") if math.isinf(hi_f) and hi_f > 0 else hi_f - 0.5
        return float(np.clip(self.cdf(upper) - self.cdf(lower), 0.0, 1.0))

    def quantile(self, probability: float) -> float:
        if self.future_max is None:
            return _round_half_up(float(self.observed_high_f))
        if self.observed_high_f is not None and probability <= self.future_max.cdf(
            self.observed_high_f
        ):
            return _round_half_up(self.observed_high_f)
        return _round_half_up(self.future_max.quantile(probability))

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
        if self.future_max is None:
            return np.full(count, _round_half_up(float(self.observed_high_f)))
        future = self.future_max.samples(seed=seed, count=count)
        if self.observed_high_f is not None:
            future = np.maximum(future, self.observed_high_f)
        return np.floor(future + 0.5)


def compose_daily_high(
    future_max: TruncatedNormal | None,
    *,
    observed_high_f: float | None = None,
) -> DailyHighGaussian:
    return DailyHighGaussian(future_max, observed_high_f)


def _round_half_up(value: float) -> float:
    return float(math.floor(value + 0.5))


class GaussianFutureMaxModel:
    """Versioned low-capacity predictor for a future-maximum distribution."""

    def __init__(
        self,
        *,
        network: GaussianFutureMaxNetwork,
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
        if len(self.feature_columns) == 0 or self.feature_mean.shape != (len(self.feature_columns),):
            raise ValueError("Gaussian feature contract is invalid")
        if self.feature_scale.shape != self.feature_mean.shape or np.any(self.feature_scale <= 0):
            raise ValueError("Gaussian feature scale is invalid")
        if len(tft_checkpoint_sha256) != 64 or len(training_manifest_sha256) != 64:
            raise ValueError("Gaussian artifact hashes must be SHA-256 digests")
        self.tft_checkpoint_sha256 = tft_checkpoint_sha256
        self.training_manifest_sha256 = training_manifest_sha256

    def predict_future(self, row: Mapping[str, object]) -> TruncatedNormal:
        missing = [key for key in self.feature_columns if key not in row]
        if missing:
            raise ValueError("Gaussian inference features missing: " + ", ".join(missing))
        values = np.array([float(row[key]) for key in self.feature_columns], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Gaussian inference features must be finite")
        normalized = (values - self.feature_mean) / self.feature_scale
        with torch.no_grad():
            output = self._network(torch.from_numpy(normalized)[None, :])[0].numpy()
        base = float(row["tft_future_peak_p50_f"])
        mean = base + float(output[0])
        sigma = float(np.log1p(np.exp(output[1])) + 0.25)
        return TruncatedNormal(mean, sigma)

    def save(self, root: Path | str) -> dict[str, object]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        weights = directory / "future_max_gaussian.pt"
        standardizer = directory / "standardizer.npz"
        torch.save(self._network.state_dict(), weights)
        np.savez_compressed(standardizer, mean=self.feature_mean, scale=self.feature_scale)
        metadata = {
            "schema_version": 1,
            "distribution_source": "future_max_truncated_gaussian_pilot",
            "feature_columns": list(self.feature_columns),
            "feature_contract_sha256": _json_sha256(list(self.feature_columns)),
            "tft_checkpoint_sha256": self.tft_checkpoint_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
            "sha256": {weights.name: _sha256(weights), standardizer.name: _sha256(standardizer)},
        }
        (directory / "future_max_gaussian.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="ascii")
        return metadata

    @classmethod
    def load(cls, root: Path | str, *, expected_tft_checkpoint_sha256: str) -> "GaussianFutureMaxModel":
        directory = Path(root)
        metadata = json.loads((directory / "future_max_gaussian.json").read_text(encoding="ascii"))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported future-max Gaussian schema")
        if metadata.get("tft_checkpoint_sha256") != expected_tft_checkpoint_sha256:
            raise ValueError("TFT checkpoint hash mismatch")
        features = tuple(str(item) for item in metadata["feature_columns"])
        if metadata.get("feature_contract_sha256") != _json_sha256(list(features)):
            raise ValueError("Gaussian feature contract hash mismatch")
        for name, expected in metadata["sha256"].items():
            if _sha256(directory / name) != expected:
                raise ValueError(f"Gaussian artifact checksum mismatch: {name}")
        with np.load(directory / "standardizer.npz", allow_pickle=False) as values:
            mean, scale = values["mean"], values["scale"]
        network = GaussianFutureMaxNetwork(len(features))
        network.load_state_dict(torch.load(directory / "future_max_gaussian.pt", map_location="cpu", weights_only=True))
        return cls(network=network, feature_columns=features, feature_mean=mean, feature_scale=scale, tft_checkpoint_sha256=str(metadata["tft_checkpoint_sha256"]), training_manifest_sha256=str(metadata["training_manifest_sha256"]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
