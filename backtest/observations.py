"""Immutable point-in-time hourly observation evidence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


_COLUMNS = {"valid_time", "available_at", "temperature_c"}


@dataclass(frozen=True)
class PointInTimeObservationDataset:
    observations: pd.DataFrame
    station_id: str
    station_timezone: str

    def __post_init__(self) -> None:
        missing = _COLUMNS - set(self.observations)
        if missing:
            raise ValueError(
                "observations missing columns: " + ", ".join(sorted(missing))
            )
        frame = self.observations.copy()
        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
        frame["available_at"] = pd.to_datetime(
            frame["available_at"], utc=True
        )
        frame["temperature_c"] = pd.to_numeric(
            frame["temperature_c"], errors="raise"
        )
        if frame["valid_time"].duplicated().any():
            raise ValueError("observations contain duplicate valid_time")
        if not np.isfinite(frame["temperature_c"].to_numpy(dtype=float)).all():
            raise ValueError("observation temperatures must be finite")
        if (frame["available_at"] < frame["valid_time"]).any():
            raise ValueError("observation available_at precedes valid_time")
        if not self.station_id or not self.station_timezone:
            raise ValueError("station id and timezone are required")
        object.__setattr__(
            self,
            "observations",
            frame.sort_values("valid_time").reset_index(drop=True),
        )

    def save(
        self,
        root: Path | str,
        *,
        source_sha256: dict[str, str],
    ) -> dict:
        artifact = Path(root)
        artifact.mkdir(parents=True, exist_ok=False)
        payload = artifact / "observations.parquet"
        self.observations.to_parquet(payload, index=False)
        manifest = {
            "schema_version": 1,
            "data_class": "point_in_time_hourly_observations",
            "station_id": self.station_id,
            "station_timezone": self.station_timezone,
            "row_count": len(self.observations),
            "start_at": self.observations["valid_time"].min().isoformat(),
            "end_at": self.observations["valid_time"].max().isoformat(),
            "source_sha256": dict(sorted(source_sha256.items())),
            "sha256": {
                payload.name: hashlib.sha256(payload.read_bytes()).hexdigest()
            },
        }
        (artifact / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, root: Path | str) -> "PointInTimeObservationDataset":
        artifact = Path(root)
        manifest = json.loads(
            (artifact / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("data_class")
            != "point_in_time_hourly_observations"
        ):
            raise RuntimeError("unsupported point-in-time observation manifest")
        payload = artifact / "observations.parquet"
        expected = manifest.get("sha256", {}).get(payload.name)
        actual = hashlib.sha256(payload.read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError("point-in-time observation checksum mismatch")
        return cls(
            observations=pd.read_parquet(payload),
            station_id=str(manifest["station_id"]),
            station_timezone=str(manifest["station_timezone"]),
        )
