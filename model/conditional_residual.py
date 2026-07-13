"""Training-only conditional residual-path dependence for daily-high MC."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


_RESIDUAL_COLUMNS = tuple(f"residual_{hour:02d}" for hour in range(1, 49))


class InsufficientConditionalPathSupport(ValueError):
    """Raised when a state has too little historical path support."""


@dataclass(frozen=True)
class ConditionalResidualPathLibrary:
    """Validated, standardized historical residual paths and state scaler."""

    paths: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ood_distance_threshold: float
    manifest_sha256: str

    @classmethod
    def fit(
        cls, paths: pd.DataFrame, *, feature_columns: Sequence[str]
    ) -> "ConditionalResidualPathLibrary":
        features = tuple(feature_columns)
        required = {"path_id", "init_time", "target_date", *features, *_RESIDUAL_COLUMNS}
        missing = sorted(required - set(paths.columns))
        if missing:
            if any(name.startswith("residual_") for name in missing):
                raise ValueError("conditional path library requires 48 residual columns")
            raise ValueError("conditional path library missing columns: " + ", ".join(missing))
        if not features:
            raise ValueError("conditional path library requires state features")
        provenance = [name for name in ("source_fold_id", "prediction_sha256", "calibration_sha256") if name in paths]
        work = paths.loc[:, ["path_id", "init_time", "target_date", *provenance, *features, *_RESIDUAL_COLUMNS]].copy()
        work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
        if work["path_id"].duplicated().any():
            raise ValueError("conditional path library contains duplicate path_id")
        numeric = [*features, *_RESIDUAL_COLUMNS]
        if not np.isfinite(work[numeric].to_numpy(dtype=float)).all():
            raise ValueError("conditional path library contains non-finite values")
        mean = work.loc[:, features].to_numpy(float).mean(axis=0)
        scale = work.loc[:, features].to_numpy(float).std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        standardized = (work.loc[:, features].to_numpy(float) - mean) / scale
        if len(standardized) < 2:
            ood_threshold = float("inf")
        else:
            distances = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            ood_threshold = float(np.quantile(distances.min(axis=1), 0.95))
        payload = {
            "feature_columns": features,
            "paths": _hash_frame(work),
            "feature_mean": mean.tolist(), "feature_scale": scale.tolist(), "ood_distance_threshold": ood_threshold,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        return cls(work.reset_index(drop=True), features, mean, scale, ood_threshold, digest)

    def standardized_state(self, state: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in self.feature_columns if name not in state]
        if missing:
            raise ValueError("conditional state missing features: " + ", ".join(missing))
        values = np.asarray([state[name] for name in self.feature_columns], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("conditional state contains non-finite values")
        return (values - self.feature_mean) / self.feature_scale

    def save(self, root: Path | str, *, lineage: Mapping[str, str]) -> dict[str, object]:
        """Persist one immutable training-only library with its feature contract."""
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "paths.parquet"
        self.paths.to_parquet(path, index=False)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "data_class": "conditional_cross_hour_residual_path_library",
            "feature_columns": list(self.feature_columns),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "ood_distance_threshold": self.ood_distance_threshold,
            "library_sha256": self.manifest_sha256,
            "paths_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path_count": len(self.paths),
            "lineage": dict(sorted(lineage.items())),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii")
        return manifest

    @classmethod
    def load(cls, root: Path | str, *, expected_lineage: Mapping[str, str] | None = None) -> "ConditionalResidualPathLibrary":
        directory = Path(root)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        if manifest.get("schema_version") != 1 or manifest.get("data_class") != "conditional_cross_hour_residual_path_library":
            raise ValueError("unsupported conditional residual path library manifest")
        if expected_lineage is not None and dict(manifest.get("lineage", {})) != dict(expected_lineage):
            raise ValueError("conditional residual path library lineage mismatch")
        path = directory / "paths.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("paths_sha256"):
            raise ValueError("conditional residual path library path hash mismatch")
        result = cls.fit(pd.read_parquet(path), feature_columns=tuple(manifest["feature_columns"]))
        if result.manifest_sha256 != manifest.get("library_sha256"):
            raise ValueError("conditional residual path library feature contract mismatch")
        if not np.isclose(result.ood_distance_threshold, float(manifest.get("ood_distance_threshold")), equal_nan=False):
            raise ValueError("conditional residual path library OOD threshold mismatch")
        return result

    @property
    def standardized_features(self) -> np.ndarray:
        values = self.paths.loc[:, self.feature_columns].to_numpy(dtype=float)
        return (values - self.feature_mean) / self.feature_scale


@dataclass(frozen=True)
class ConditionalPathDiagnostics:
    nearest_distance: float
    effective_sample_size: float
    candidate_path_count: int
    is_ood: bool
    path_peak_slot_entropy: float
    path_peak_slot_mode_share: float


class ConditionalPathSampler:
    """Resample complete standardized residual paths by forecast-state similarity."""

    def __init__(
        self, library: ConditionalResidualPathLibrary, *, n_samp: int = 10_000,
        rng_seed: int = 42, neighbors: int = 128, min_effective_samples: float = 32.0,
    ) -> None:
        if n_samp <= 0 or neighbors <= 0 or min_effective_samples <= 0:
            raise ValueError("conditional sampler settings must be positive")
        self.n_samp = int(n_samp)
        self.rng_seed = int(rng_seed)
        self.neighbors = int(neighbors)
        self.min_effective_samples = float(min_effective_samples)
        self.library = library
        self._rng = np.random.default_rng(rng_seed)
        self._standardized_features = library.standardized_features
        self._residual_paths = library.paths.loc[:, _RESIDUAL_COLUMNS].to_numpy(float)
        self.last_diagnostics: ConditionalPathDiagnostics | None = None
        self.last_state: dict[str, float] | None = None

    def sample_daily_high(
        self, mu_vec: np.ndarray, sigma_vec: np.ndarray, hour_indices: Sequence[int],
        *, state: Mapping[str, float], max_bias: float = 0.0,
    ) -> np.ndarray:
        mu = np.asarray(mu_vec, dtype=float)
        sigma = np.asarray(sigma_vec, dtype=float)
        indices = np.asarray(hour_indices, dtype=int)
        if len(mu) == 0 or len(mu) != len(sigma) or len(mu) != len(indices):
            raise ValueError("conditional sampler vectors must be non-empty and aligned")
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or not (sigma > 0).all():
            raise ValueError("conditional sampler requires finite positive sigma")
        if (indices < 0).any() or (indices >= 48).any() or len(np.unique(indices)) != len(indices):
            raise ValueError("conditional sampler hour_indices must be unique values in 0..47")
        state_vector = self.library.standardized_state(state)
        self.last_state = {name: float(state[name]) for name in self.library.feature_columns}
        distances = np.linalg.norm(self._standardized_features - state_vector, axis=1)
        order = np.argsort(distances, kind="stable")[: min(self.neighbors, len(distances))]
        selected_distance = distances[order]
        bandwidth = max(float(np.median(selected_distance)), 1e-6)
        weights = np.exp(-0.5 * (selected_distance / bandwidth) ** 2)
        weights /= weights.sum()
        ess = float(1.0 / np.square(weights).sum())
        if "tft_peak_slot" in self.library.paths:
            path_peak_slots = self.library.paths.iloc[order]["tft_peak_slot"].to_numpy(dtype=int)
            peak_weights = np.bincount(path_peak_slots, weights=weights, minlength=48)
            peak_weights = peak_weights[peak_weights > 0]
            peak_entropy = float(-(peak_weights * np.log(peak_weights)).sum())
            peak_mode_share = float(peak_weights.max())
        else:
            peak_entropy = float("nan")
            peak_mode_share = float("nan")
        self.last_diagnostics = ConditionalPathDiagnostics(
            nearest_distance=float(selected_distance[0]), effective_sample_size=ess,
            candidate_path_count=len(order),
            is_ood=bool(selected_distance[0] > self.library.ood_distance_threshold),
            path_peak_slot_entropy=peak_entropy,
            path_peak_slot_mode_share=peak_mode_share,
        )
        if ess < self.min_effective_samples:
            raise InsufficientConditionalPathSupport(f"conditional path ESS={ess:.2f} < {self.min_effective_samples:.2f}")
        chosen = self._rng.choice(order, size=self.n_samp, replace=True, p=weights)
        residuals = self._residual_paths[chosen][:, indices]
        temperature_c = mu[None, :] + residuals * sigma[None, :]
        return temperature_c.max(axis=1) * 1.8 + 32.0 - float(max_bias)


def _hash_frame(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()
