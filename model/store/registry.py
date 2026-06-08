"""Model version registry.

Manages checkpoint publication, discovery of the latest model version,
and rollback to a previous version.

Layout on disk:
    <store_dir>/
        <version>/
            model.ckpt
            training_dataset.pkl   (optional, passed through)
            meta.json
        active_version.txt         (tracks the pinned/active version)
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from db.schema import DbClient
    from model.tft import TftModel


class ModelRegistry:
    """Publish and discover TFT model checkpoints.

    Parameters
    ----------
    store_dir:
        Root directory where versioned checkpoints are stored.
    db:
        DbClient instance for persisting version metadata.
    """

    _ACTIVE_VERSION_FILE = "active_version.txt"

    def __init__(self, store_dir: Path, db: "DbClient") -> None:
        self.store_dir = Path(store_dir)
        self.db        = db
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self,
        version: str,
        ckpt_path: Path,
        metrics: dict,
        training_dataset_pkl: Path | None = None,
        activate: bool = True,
        extra_artifacts: list[Path] | None = None,
    ) -> None:
        """Copy checkpoint to the registry and register it.

        Parameters
        ----------
        version:
            Unique identifier string, e.g. "v20240101_1200".
        ckpt_path:
            Source .ckpt file.
        metrics:
            Dictionary with at minimum "train_mae" and "test_mae" keys.
        training_dataset_pkl:
            Optional pickled TimeSeriesDataSet to bundle with the checkpoint.
        """
        version_dir = self.store_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        dest_ckpt = version_dir / "model.ckpt"
        shutil.copy2(ckpt_path, dest_ckpt)

        if training_dataset_pkl is not None:
            dest_pkl = version_dir / "training_dataset.pkl"
            shutil.copy2(training_dataset_pkl, dest_pkl)
        for artifact in extra_artifacts or []:
            shutil.copy2(artifact, version_dir / artifact.name)

        created_at = datetime.now(timezone.utc).isoformat()

        meta = {
            "version":         version,
            "checkpoint_path": str(dest_ckpt),
            "metrics":         metrics,
            "created_at":      created_at,
        }
        with open(version_dir / "meta.json", "w", encoding="ascii") as fh:
            json.dump(meta, fh, indent=2)

        self.db.register_model_version(
            version         = version,
            checkpoint_path = str(dest_ckpt),
            train_mae       = float(metrics.get("train_mae", 0.0)),
            test_mae        = float(metrics.get("test_mae", 0.0)),
            created_at      = created_at,
        )

        if activate:
            self._write_active_version(version)

    # ------------------------------------------------------------------
    # Load latest
    # ------------------------------------------------------------------

    def load_latest(self) -> "TftModel":
        """Return a loaded TftModel for the currently active version.

        The active version is determined by:
        1. active_version.txt (set by publish or rollback).
        2. If no file, the newest entry in model_versions by created_at.

        Raises
        ------
        RuntimeError
            If no checkpoint is registered.
        """
        from model.tft import TftModel

        version = self._read_active_version()

        if version is None:
            # Fall back to DB ordering
            row = self._db_latest_row()
            if row is None:
                raise RuntimeError(
                    "No model version registered. Call publish() first."
                )
            version = row["version"]

        meta = self._load_meta(version)
        ckpt_path = Path(meta["checkpoint_path"])

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"model.ckpt not found at {ckpt_path}. "
                "The checkpoint was deleted or moved after publication."
            )

        pkl_path  = ckpt_path.parent / "training_dataset.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"training_dataset.pkl not found at {pkl_path}. "
                "Bundle it during publish() or provide it manually."
            )
        return TftModel(ckpt_path, pkl_path)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, version: str) -> None:
        """Pin a specific version as the active model.

        Parameters
        ----------
        version:
            Version string that was previously published.

        Raises
        ------
        ValueError
            If the requested version does not exist in the registry.
        """
        version_dir = self.store_dir / version
        if not version_dir.exists():
            raise ValueError(
                f"Version '{version}' not found in store_dir={self.store_dir}. "
                "Make sure publish() was called for this version."
            )
        self._write_active_version(version)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_active_version(self, version: str) -> None:
        active_file = self.store_dir / self._ACTIVE_VERSION_FILE
        active_file.write_text(version, encoding="ascii")

    def _read_active_version(self) -> str | None:
        active_file = self.store_dir / self._ACTIVE_VERSION_FILE
        if active_file.exists():
            v = active_file.read_text(encoding="ascii").strip()
            return v or None
        return None

    def _db_latest_row(self):
        rows = self.db.query(
            "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT 1"
        )
        return rows[0] if rows else None

    def _load_meta(self, version: str) -> dict:
        meta_path = self.store_dir / version / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"meta.json not found for version '{version}' "
                f"at {meta_path}."
            )
        with open(meta_path, encoding="ascii") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Max-bias calibration
    # ------------------------------------------------------------------

    def store_max_bias(self, station_id: str, max_bias: float) -> None:
        """Persist per-station max_bias correction."""
        self.db.execute(
            """
            INSERT INTO station_max_bias (station_id, max_bias)
            VALUES (?, ?)
            ON CONFLICT(station_id) DO UPDATE SET max_bias=excluded.max_bias
            """,
            (station_id, max_bias),
        )

    def load_max_bias(self, station_id: str) -> float:
        """Return stored max_bias for station, or 0.0 if not yet calibrated."""
        rows = self.db.query(
            "SELECT max_bias FROM station_max_bias WHERE station_id=?",
            (station_id,),
        )
        if not rows:
            return 0.0
        return float(rows[0]["max_bias"])

    def calibrate_max_bias(
        self,
        station_id: str,
        val_residuals: np.ndarray,
    ) -> float:
        """Estimate and store max_bias from validation-set residuals.

        max_bias = E[max(samples)] - mean(observed daily max)

        Here val_residuals is a (n_val_cycles, n_hours) array of
        (predicted - observed) residuals per decoder hour.  The mean of each
        row maximum approximates the systematic over-prediction of the daily
        high.

        Parameters
        ----------
        station_id:
            Station to calibrate.
        val_residuals:
            (n_val_cycles, n_hours) residual array from the validation set.

        Returns
        -------
        Stored max_bias value.
        """
        import numpy as _np
        max_bias = float(_np.mean(val_residuals.max(axis=1)))
        self.store_max_bias(station_id, max_bias)
        return max_bias

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    def load_covariance(
        self,
        version: str | None = None,
    ) -> "np.ndarray | None":
        """Load the empirical covariance matrix for the given model version.

        Looks for a ``covariance.npy`` file in the version directory alongside
        the model checkpoint.  If no version is specified the currently active
        version is used.

        Parameters
        ----------
        version:
            Version string to load from.  When None the active version is used.

        Returns
        -------
        (n_hours, n_hours) ndarray when present, otherwise None.  A WARNING is
        logged when the file is absent; the caller (bot.py) will fall back to the
        diagonal covariance built from per-hour sigma_vec.
        """
        from loguru import logger

        if version is None:
            version = self._read_active_version()
            if version is None:
                row = self._db_latest_row()
                version = row["version"] if row else None

        if version is None:
            logger.warning(
                "ModelRegistry: no active version -- "
                "covariance.npy not found; MC will use diagonal (uncorrelated) assumption"
            )
            return None

        cov_path = self.store_dir / version / "covariance.npy"
        if not cov_path.exists():
            logger.warning(
                "ModelRegistry: covariance.npy not found at {} -- "
                "MC will use diagonal (uncorrelated) assumption",
                cov_path,
            )
            return None

        cov = np.load(str(cov_path))
        _COV_DIM = 48
        if cov.shape != (_COV_DIM, _COV_DIM):
            raise ValueError(
                f"covariance.npy at {cov_path} has shape {cov.shape}; "
                f"expected ({_COV_DIM}, {_COV_DIM}). "
                "Re-run fit_cov() with a 48-hour residual window and re-publish."
            )
        return cov
