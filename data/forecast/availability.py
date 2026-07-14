"""Point-in-time HRRR object availability calculations."""
from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from data.forecast.hrrr_fill import MAJOR_HOURS


_HRRR_KEY = re.compile(
    r"hrrr\.(?P<date>\d{8})/conus/"
    r"hrrr\.t(?P<hour>\d{2})z\.wrfsfcf(?P<fxx>\d{2})\.grib2$"
)


class TextTransport(Protocol):
    def get_text(self, url: str, params: dict) -> str: ...


class RequestsTextTransport:
    def get_text(self, url: str, params: dict) -> str:
        import requests

        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
            time.sleep(2**attempt)
        raise RuntimeError("unreachable S3 listing retry state")


class HrrrS3ObjectSource:
    """List HRRR GRIB objects and original S3 upload timestamps by day."""

    def __init__(
        self,
        *,
        transport: TextTransport | None = None,
        bucket_url: str = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
    ) -> None:
        self._transport = transport or RequestsTextTransport()
        self._bucket_url = bucket_url.rstrip("/")

    def fetch_day(self, day: str | pd.Timestamp) -> pd.DataFrame:
        stamp = pd.Timestamp(day)
        prefix = f"hrrr.{stamp.strftime('%Y%m%d')}/conus/"
        rows: list[dict] = []
        continuation: str | None = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if continuation is not None:
                params["continuation-token"] = continuation
            xml_text = self._transport.get_text(
                f"{self._bucket_url}/",
                params,
            )
            root = ET.fromstring(xml_text)
            for content in root.findall(".//{*}Contents"):
                key_node = content.find("{*}Key")
                modified_node = content.find("{*}LastModified")
                if key_node is None or modified_node is None:
                    continue
                key = key_node.text or ""
                if _HRRR_KEY.search(key) is None:
                    continue
                rows.append(
                    {
                        "key": key,
                        "last_modified": pd.Timestamp(modified_node.text),
                    }
                )
            truncated = (
                root.findtext("{*}IsTruncated", default="false").lower()
                == "true"
            )
            if not truncated:
                break
            continuation = root.findtext("{*}NextContinuationToken")
            if not continuation:
                raise RuntimeError(
                    f"S3 listing for {prefix} is truncated without a token"
                )
        if not rows:
            return pd.DataFrame(columns=["key", "last_modified"])
        return pd.DataFrame(rows).sort_values("key").reset_index(drop=True)


@dataclass(frozen=True)
class HrrrAvailabilityDataset:
    """Immutable HRRR S3 object and cycle-availability evidence."""

    objects: pd.DataFrame
    cycles: pd.DataFrame
    start_date: str
    end_date: str
    compute_budget: str
    decision_tick: str
    provider: str = "noaa_hrrr_bdp_pds_s3"

    def save(
        self,
        root: Path | str,
        *,
        fetched_at: str | None = None,
    ) -> dict:
        artifact_dir = Path(root)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        frames = {
            "objects.parquet": self.objects,
            "cycles.parquet": self.cycles,
        }
        hashes: dict[str, str] = {}
        for filename, frame in frames.items():
            path = artifact_dir / filename
            frame.to_parquet(path, index=False)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        fetched = pd.Timestamp(fetched_at or pd.Timestamp.now(tz="UTC"))
        fetched = (
            fetched.tz_localize("UTC")
            if fetched.tzinfo is None
            else fetched.tz_convert("UTC")
        )
        incomplete = self.cycles.loc[
            ~self.cycles["complete"], "init_time"
        ]
        manifest = {
            "schema_version": 1,
            "data_class": "hrrr_point_in_time_availability",
            "provider": self.provider,
            "requested_range": [self.start_date, self.end_date],
            "compute_budget": self.compute_budget,
            "decision_tick": self.decision_tick,
            "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
            "row_counts": {
                "objects": len(self.objects),
                "cycles": len(self.cycles),
            },
            "incomplete_init_times": [
                pd.Timestamp(value).isoformat() for value in incomplete
            ],
            "sha256": hashes,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, root: Path | str) -> "HrrrAvailabilityDataset":
        artifact_dir = Path(root)
        try:
            manifest = json.loads(
                (artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid HRRR availability manifest: {exc}") from exc
        if manifest.get("schema_version") != 1:
            raise RuntimeError(
                "unsupported HRRR availability schema_version="
                f"{manifest.get('schema_version')!r}"
            )
        for filename, expected in manifest.get("sha256", {}).items():
            path = artifact_dir / filename
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise RuntimeError(
                    f"missing HRRR availability file {filename}"
                ) from exc
            if actual != expected:
                raise RuntimeError(
                    f"HRRR availability checksum mismatch: {filename}"
                )
        start_date, end_date = manifest["requested_range"]
        return cls(
            objects=pd.read_parquet(artifact_dir / "objects.parquet"),
            cycles=pd.read_parquet(artifact_dir / "cycles.parquet"),
            start_date=str(start_date),
            end_date=str(end_date),
            compute_budget=str(manifest["compute_budget"]),
            decision_tick=str(manifest["decision_tick"]),
            provider=str(manifest.get("provider", "noaa_hrrr_bdp_pds_s3")),
        )


def build_hrrr_availability(
    objects: pd.DataFrame,
    *,
    init_times: Iterable[pd.Timestamp],
    compute_budget: str | pd.Timedelta = "5min",
    decision_tick: str = "5min",
) -> pd.DataFrame:
    """Map S3 object timestamps to decoder-complete forecast availability."""
    lookup: dict[tuple[pd.Timestamp, int], pd.Timestamp] = {}
    for row in objects.itertuples(index=False):
        match = _HRRR_KEY.search(str(row.key))
        if match is None:
            continue
        init = pd.Timestamp(
            f"{match.group('date')}T{match.group('hour')}:00:00Z"
        )
        modified = pd.Timestamp(row.last_modified)
        if modified.tzinfo is None:
            modified = modified.tz_localize("UTC")
        else:
            modified = modified.tz_convert("UTC")
        lookup[(init, int(match.group("fxx")))] = modified

    budget = pd.Timedelta(compute_budget)
    rows: list[dict] = []
    for value in init_times:
        init = pd.Timestamp(value)
        if init.tzinfo is None:
            init = init.tz_localize("UTC")
        else:
            init = init.tz_convert("UTC")
        is_major = init.hour in MAJOR_HOURS
        own_max_fxx = 48 if is_major else 18
        own_keys = [(init, fxx) for fxx in range(1, own_max_fxx + 1)]
        dependency = pd.NaT if is_major else init.floor("6h")
        dependency_keys = (
            []
            if is_major
            else [(dependency, fxx) for fxx in range(1, 49)]
        )
        required = own_keys + dependency_keys
        missing = [
            f"{key[0].isoformat()}/f{key[1]:02d}"
            for key in required
            if key not in lookup
        ]
        complete = not missing
        ready_at = (
            max(lookup[key] for key in required)
            if complete
            else pd.NaT
        )
        available_at = (
            (ready_at + budget).ceil(decision_tick)
            if complete
            else pd.NaT
        )
        rows.append(
            {
                "init_time": init,
                "is_major": is_major,
                "own_max_fxx": own_max_fxx,
                "dependency_major_init": dependency,
                "source_ready_at": ready_at,
                "available_at": available_at,
                "required_object_count": len(required),
                "missing_objects": missing,
                "complete": complete,
            }
        )
    return pd.DataFrame(rows).sort_values("init_time").reset_index(drop=True)


def build_virtual_tft_availability(
    *,
    init_times: Iterable[pd.Timestamp],
    latency: str | pd.Timedelta = "2h",
    decision_tick: str = "5min",
) -> pd.DataFrame:
    """Build an explicitly approximate fixed-latency TFT visibility clock."""
    delay = pd.Timedelta(latency)
    if delay < pd.Timedelta(0):
        raise ValueError("virtual TFT latency must be non-negative")
    rows: list[dict] = []
    for value in init_times:
        init = pd.Timestamp(value)
        init = init.tz_localize("UTC") if init.tzinfo is None else init.tz_convert("UTC")
        rows.append(
            {
                "init_time": init,
                "is_major": pd.NA,
                "own_max_fxx": pd.NA,
                "dependency_major_init": pd.NaT,
                "source_ready_at": init,
                "available_at": (init + delay).ceil(decision_tick),
                "required_object_count": 0,
                "missing_objects": [],
                "complete": True,
            }
        )
    return pd.DataFrame(rows).sort_values("init_time").reset_index(drop=True)
