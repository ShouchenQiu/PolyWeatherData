"""Immutable point-in-time model forecasts for deterministic replay."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_SNAPSHOT_COLUMNS = {
    "snapshot_id",
    "station_id",
    "init_time",
    "target_date",
    "available_at",
    "p10_f",
    "p50_f",
    "p90_f",
    "lineage_id",
    "coverage_type",
    "observed_high_f",
    "observed_hour_count",
    "observed_through",
    "forecast_hour_count",
    "forecast_start_at",
    "forecast_through",
    "target_hour_count",
}
_PROBABILITY_COLUMNS = {
    "snapshot_id",
    "condition_id",
    "yes_probability",
}
_LINEAGE_COLUMNS = {
    "lineage_id",
    "checkpoint_sha256",
    "calibration_sha256",
    "split_role",
    "mc_seed",
    "mc_samples",
}
_SPLIT_ROLES = {"oof_development", "model_test", "post_test_holdout"}


@dataclass(frozen=True)
class ForecastCache:
    """Forecast snapshots plus market probabilities and model lineage."""

    snapshots: pd.DataFrame
    probabilities: pd.DataFrame
    lineages: pd.DataFrame

    def __post_init__(self) -> None:
        snapshots = self.snapshots.copy()
        defaults = {
            "coverage_type": "forecast_only",
            "observed_high_f": float("nan"),
            "observed_hour_count": 0,
            "observed_through": pd.NaT,
            "forecast_hour_count": 24,
            "forecast_start_at": pd.NaT,
            "forecast_through": pd.NaT,
            "target_hour_count": 24,
        }
        for column, default in defaults.items():
            if column not in snapshots:
                snapshots[column] = default
        probabilities = self.probabilities.copy()
        lineages = self.lineages.copy()
        _require_columns(snapshots, _SNAPSHOT_COLUMNS, "snapshots")
        _require_columns(probabilities, _PROBABILITY_COLUMNS, "probabilities")
        _require_columns(lineages, _LINEAGE_COLUMNS, "lineages")

        snapshots["init_time"] = pd.to_datetime(
            snapshots["init_time"], utc=True
        )
        snapshots["available_at"] = pd.to_datetime(
            snapshots["available_at"], utc=True
        )
        snapshots["target_date"] = pd.to_datetime(
            snapshots["target_date"]
        ).dt.date.astype(str)
        for column in (
            "observed_through",
            "forecast_start_at",
            "forecast_through",
        ):
            snapshots[column] = pd.to_datetime(
                snapshots[column], utc=True, errors="coerce"
            )
        for column in (
            "observed_hour_count",
            "forecast_hour_count",
            "target_hour_count",
        ):
            snapshots[column] = snapshots[column].astype(int)
        if not set(snapshots["coverage_type"]).issubset(
            {"forecast_only", "observed_plus_forecast", "observed_only"}
        ):
            raise ValueError("unsupported daily-high coverage_type")
        if (
            snapshots["observed_hour_count"]
            + snapshots["forecast_hour_count"]
            != snapshots["target_hour_count"]
        ).any():
            raise ValueError("snapshot observation/forecast coverage is incomplete")
        if snapshots["snapshot_id"].duplicated().any():
            raise ValueError("snapshots contain duplicate snapshot_id")
        if lineages["lineage_id"].duplicated().any():
            raise ValueError("lineages contain duplicate lineage_id")
        unknown_lineage = set(snapshots["lineage_id"]) - set(
            lineages["lineage_id"]
        )
        if unknown_lineage:
            raise ValueError(
                f"snapshots reference unknown lineage: {sorted(unknown_lineage)}"
            )
        unknown_snapshot = set(probabilities["snapshot_id"]) - set(
            snapshots["snapshot_id"]
        )
        if unknown_snapshot:
            raise ValueError(
                "probabilities reference unknown snapshot: "
                f"{sorted(unknown_snapshot)}"
            )
        if not probabilities["yes_probability"].between(0.0, 1.0).all():
            raise ValueError("yes_probability must be between zero and one")
        if (
            probabilities.duplicated(["snapshot_id", "condition_id"]).any()
        ):
            raise ValueError(
                "probabilities contain duplicate snapshot/condition pairs"
            )
        if not set(lineages["split_role"]).issubset(_SPLIT_ROLES):
            raise ValueError(
                f"split_role must be one of {sorted(_SPLIT_ROLES)}"
            )
        if (lineages["mc_samples"].astype(int) <= 0).any():
            raise ValueError("mc_samples must be positive")
        invalid_intervals = (
            (snapshots["p10_f"] > snapshots["p50_f"])
            | (snapshots["p50_f"] > snapshots["p90_f"])
        )
        if invalid_intervals.any():
            raise ValueError("forecast quantiles must satisfy p10 <= p50 <= p90")

        object.__setattr__(
            self,
            "snapshots",
            snapshots.sort_values(
                ["available_at", "target_date", "init_time"]
            ).reset_index(drop=True),
        )
        object.__setattr__(
            self,
            "probabilities",
            probabilities.sort_values(
                ["snapshot_id", "condition_id"]
            ).reset_index(drop=True),
        )
        object.__setattr__(
            self,
            "lineages",
            lineages.sort_values("lineage_id").reset_index(drop=True),
        )

    def latest_at(
        self,
        decision_time: str | pd.Timestamp,
        *,
        station_id: str,
        target_date: str | None = None,
    ) -> pd.DataFrame:
        """Return the latest actually available snapshot per target date."""
        tick = _utc(decision_time)
        eligible = self.snapshots[
            (self.snapshots["station_id"] == station_id)
            & (self.snapshots["available_at"] <= tick)
        ]
        if target_date is not None:
            normalized = pd.Timestamp(target_date).date().isoformat()
            eligible = eligible[eligible["target_date"] == normalized]
        if eligible.empty:
            return eligible.copy()
        return (
            eligible.sort_values(["available_at", "init_time"])
            .groupby(["station_id", "target_date"], sort=True)
            .tail(1)
            .sort_values("target_date")
            .reset_index(drop=True)
        )

    def probabilities_at(
        self,
        decision_time: str | pd.Timestamp,
        *,
        station_id: str,
        target_date: str | None = None,
    ) -> pd.DataFrame:
        """Return market probabilities belonging to visible snapshots only."""
        visible = self.latest_at(
            decision_time,
            station_id=station_id,
            target_date=target_date,
        )
        columns = [
            "snapshot_id",
            "station_id",
            "init_time",
            "target_date",
            "available_at",
            "lineage_id",
            "condition_id",
            "yes_probability",
        ]
        if visible.empty:
            return pd.DataFrame(columns=columns)
        return visible.merge(
            self.probabilities,
            on="snapshot_id",
            how="inner",
            validate="one_to_many",
        )[columns].sort_values(
            ["target_date", "condition_id"]
        ).reset_index(drop=True)

    def save(
        self,
        root: Path | str,
        *,
        created_at: str | pd.Timestamp | None = None,
        source_sha256: dict[str, str] | None = None,
    ) -> dict:
        """Write an immutable, hash-verified forecast artifact."""
        artifact_dir = Path(root)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        frames = {
            "snapshots.parquet": self.snapshots,
            "probabilities.parquet": self.probabilities,
            "lineages.parquet": self.lineages,
        }
        hashes: dict[str, str] = {}
        for filename, frame in frames.items():
            path = artifact_dir / filename
            frame.to_parquet(path, index=False)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        created = _utc(created_at or pd.Timestamp.now(tz="UTC"))
        manifest = {
            "schema_version": 2,
            "data_class": "point_in_time_forecast_cache",
            "target_window": "point_in_time_local_calendar_day",
            "resolution_rounding": "half_up_fahrenheit",
            "partial_day_policy": (
                "observed_plus_trusted_forecast_full_coverage"
            ),
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "station_ids": sorted(self.snapshots["station_id"].unique()),
            "split_roles": sorted(self.lineages["split_role"].unique()),
            "row_counts": {
                "snapshots": len(self.snapshots),
                "probabilities": len(self.probabilities),
                "lineages": len(self.lineages),
            },
            "source_sha256": dict(sorted((source_sha256 or {}).items())),
            "sha256": hashes,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, root: Path | str) -> "ForecastCache":
        artifact_dir = Path(root)
        try:
            manifest = json.loads(
                (artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid forecast cache manifest: {exc}") from exc
        if manifest.get("schema_version") not in {1, 2}:
            raise RuntimeError(
                "unsupported forecast cache schema_version="
                f"{manifest.get('schema_version')!r}"
            )
        for filename, expected in manifest.get("sha256", {}).items():
            path = artifact_dir / filename
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise RuntimeError(
                    f"missing forecast cache file {filename}"
                ) from exc
            if actual != expected:
                raise RuntimeError(
                    f"forecast cache checksum mismatch: {filename}"
                )
        return cls(
            snapshots=pd.read_parquet(artifact_dir / "snapshots.parquet"),
            probabilities=pd.read_parquet(
                artifact_dir / "probabilities.parquet"
            ),
            lineages=pd.read_parquet(artifact_dir / "lineages.parquet"),
        )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(missing)}")


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
