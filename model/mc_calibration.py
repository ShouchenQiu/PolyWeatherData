"""Version-bound Monte Carlo calibration artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_TARGET_WINDOW = "local_calendar_day"
FXX_BANDS: tuple[tuple[int, int], ...] = (
    (1, 6),
    (7, 12),
    (13, 24),
    (25, 36),
    (37, 48),
)
CALIBRATION_ARRAYS_FILE = "mc_calibration.npz"
CALIBRATION_METADATA_FILE = "mc_calibration.json"


@dataclass(frozen=True)
class McCalibration:
    """Validated MC correlation, scale, and daily-high bias parameters."""

    correlation: np.ndarray
    sigma_scales: np.ndarray
    max_bias_f: float
    model_sha256: str
    fit_split: str
    fit_start: str
    fit_end: str
    sample_count: int
    station_timezone: str = "UTC"
    target_window: str = CALIBRATION_TARGET_WINDOW

    def __post_init__(self) -> None:
        correlation = np.asarray(self.correlation, dtype=float)
        sigma_scales = np.asarray(self.sigma_scales, dtype=float)
        if correlation.shape != (48, 48):
            raise ValueError(
                "MC calibration correlation must have shape (48, 48), "
                f"got {correlation.shape}"
            )
        if not np.all(np.isfinite(correlation)):
            raise ValueError("MC calibration correlation must be finite")
        if not np.allclose(correlation, correlation.T, atol=1e-8):
            raise ValueError("MC calibration correlation must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-6):
            raise ValueError("MC calibration correlation must have unit diagonal")
        if np.linalg.eigvalsh(correlation).min() <= 0:
            raise ValueError(
                "MC calibration correlation must be positive-definite"
            )
        if sigma_scales.shape != (len(FXX_BANDS),):
            raise ValueError(
                "MC calibration sigma_scales must have shape "
                f"({len(FXX_BANDS)},), got {sigma_scales.shape}"
            )
        if not np.all(np.isfinite(sigma_scales)) or np.any(sigma_scales <= 0):
            raise ValueError(
                "MC calibration sigma_scales must be finite and positive"
            )
        if not np.isfinite(self.max_bias_f):
            raise ValueError("MC calibration max_bias_f must be finite")
        if len(self.model_sha256) != 64:
            raise ValueError("MC calibration model_sha256 must be a SHA-256 hex digest")
        if self.sample_count <= 0:
            raise ValueError("MC calibration sample_count must be positive")
        if not self.station_timezone:
            raise ValueError("MC calibration station_timezone must be non-empty")
        if self.target_window != CALIBRATION_TARGET_WINDOW:
            raise ValueError(
                "MC calibration target_window must be local_calendar_day"
            )
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "sigma_scales", sigma_scales)

    def save(self, directory: Path | str) -> tuple[Path, Path]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        arrays_path = root / CALIBRATION_ARRAYS_FILE
        metadata_path = root / CALIBRATION_METADATA_FILE
        np.savez_compressed(
            arrays_path,
            correlation=self.correlation,
            sigma_scales=self.sigma_scales,
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "model_sha256": self.model_sha256,
                    "fxx_bands": [list(band) for band in FXX_BANDS],
                    "max_bias_f": self.max_bias_f,
                    "fit_split": self.fit_split,
                    "fit_start": self.fit_start,
                    "fit_end": self.fit_end,
                    "sample_count": self.sample_count,
                    "station_timezone": self.station_timezone,
                    "target_window": self.target_window,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="ascii",
        )
        return arrays_path, metadata_path

    @classmethod
    def load(
        cls,
        directory: Path | str,
        checkpoint_path: Path | str,
    ) -> "McCalibration":
        root = Path(directory)
        arrays_path = root / CALIBRATION_ARRAYS_FILE
        metadata_path = root / CALIBRATION_METADATA_FILE
        if not arrays_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "v2 model requires mc_calibration.npz and mc_calibration.json"
            )
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        if metadata.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported MC calibration schema_version="
                f"{metadata.get('schema_version')!r}"
            )
        if metadata.get("fxx_bands") != [list(band) for band in FXX_BANDS]:
            raise ValueError("MC calibration fxx_bands mismatch")
        expected_hash = cls.sha256(checkpoint_path)
        if metadata.get("model_sha256") != expected_hash:
            raise ValueError(
                "MC calibration checkpoint hash mismatch: "
                f"expected {expected_hash}, got {metadata.get('model_sha256')}"
            )
        with np.load(arrays_path, allow_pickle=False) as arrays:
            correlation = arrays["correlation"]
            sigma_scales = arrays["sigma_scales"]
        return cls(
            correlation=correlation,
            sigma_scales=sigma_scales,
            max_bias_f=float(metadata["max_bias_f"]),
            model_sha256=str(metadata["model_sha256"]),
            fit_split=str(metadata["fit_split"]),
            fit_start=str(metadata["fit_start"]),
            fit_end=str(metadata["fit_end"]),
            sample_count=int(metadata["sample_count"]),
            station_timezone=str(metadata["station_timezone"]),
            target_window=str(metadata["target_window"]),
        )

    @staticmethod
    def sha256(path: Path | str) -> str:
        hasher = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def scale_sigma(
        self,
        sigma: np.ndarray,
        hour_indices: list[int] | None = None,
    ) -> np.ndarray:
        values = np.asarray(sigma, dtype=float)
        indices = (
            list(range(len(values)))
            if hour_indices is None
            else list(hour_indices)
        )
        if len(indices) != len(values):
            raise ValueError("hour_indices length must match sigma length")
        scales = np.array(
            [self.sigma_scales[_band_index(index + 1)] for index in indices],
            dtype=float,
        )
        return values * scales


def _band_index(fxx: int) -> int:
    for index, (start, end) in enumerate(FXX_BANDS):
        if start <= fxx <= end:
            return index
    raise ValueError(f"fxx must be in 1..48, got {fxx}")
