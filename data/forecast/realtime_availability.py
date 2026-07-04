"""Forward-only probes for real HRRR publication and decoder latency."""
from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from data.forecast.hrrr_fill import MAJOR_HOURS


@dataclass(frozen=True)
class CycleRequirement:
    """Objects required to construct one production decoder."""

    init_time: pd.Timestamp
    kind: str
    own_fxx: tuple[int, ...]
    dependency_init: pd.Timestamp | None

    @property
    def terminal_fxx(self) -> int:
        return self.own_fxx[-1]


@dataclass(frozen=True)
class SourceSpec:
    """One public HRRR mirror with the standard object-key layout."""

    name: str
    base_url: str

    def idx_url(self, init_time: pd.Timestamp, fxx: int) -> str:
        return (
            f"{self.base_url.rstrip('/')}/hrrr.{init_time:%Y%m%d}/conus/"
            f"hrrr.t{init_time:%H}z.wrfsfcf{fxx:02d}.grib2.idx"
        )


DEFAULT_SOURCES = (
    SourceSpec(
        "aws",
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
    ),
    SourceSpec(
        "nomads",
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod",
    ),
    SourceSpec(
        "azure",
        "https://noaahrrr.blob.core.windows.net/hrrr",
    ),
)


@dataclass(frozen=True)
class HeadResult:
    status_code: int | None
    elapsed_ms: float
    last_modified: str
    http_date: str
    etag: str
    content_length: int | None
    error: str
    observed_at: str = ""

    @property
    def available(self) -> bool:
        return self.status_code == 200 and not self.error


class HeadClient(Protocol):
    def head(self, url: str) -> HeadResult: ...


class RequestsHeadClient:
    """Thread-local HTTP HEAD client; retries are left to the next poll."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._timeout = timeout_seconds
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {"User-Agent": "PolyWeatherBot-HRRR-Availability-Probe/1.0"}
            )
            self._local.session = session
        return session

    def head(self, url: str) -> HeadResult:
        started = time.perf_counter()
        try:
            response = self._session().head(
                url,
                timeout=self._timeout,
                allow_redirects=True,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return HeadResult(
                status_code=int(response.status_code),
                elapsed_ms=elapsed_ms,
                last_modified=str(response.headers.get("Last-Modified", "")),
                http_date=str(response.headers.get("Date", "")),
                etag=str(response.headers.get("ETag", "")),
                content_length=_optional_int(
                    response.headers.get("Content-Length")
                ),
                error="",
                observed_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )
        except requests.RequestException as exc:
            return HeadResult(
                status_code=None,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                last_modified="",
                http_date="",
                etag="",
                content_length=None,
                error=f"{type(exc).__name__}: {exc}",
                observed_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )


@dataclass(frozen=True)
class ProbeEvent:
    event_type: str
    source: str
    init_time: str
    fxx: int | None
    observed_at: str
    latency_seconds: float
    detail: str = ""


def cycle_requirement(init_time: str | pd.Timestamp) -> CycleRequirement:
    """Return the existing major/non-major decoder dependency contract."""
    init = pd.Timestamp(init_time)
    init = (
        init.tz_localize("UTC")
        if init.tzinfo is None
        else init.tz_convert("UTC")
    ).floor("h")
    is_major = init.hour in MAJOR_HOURS
    return CycleRequirement(
        init_time=init,
        kind="major" if is_major else "non_major",
        own_fxx=tuple(range(1, 49 if is_major else 19)),
        dependency_init=None if is_major else init.floor("6h"),
    )


class RealtimeAvailabilityProbe:
    """Poll mirrors and persist first-observed publication evidence in SQLite."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        sources: list[SourceSpec] | tuple[SourceSpec, ...] = DEFAULT_SOURCES,
        head_client: HeadClient | None = None,
        poll_interval_seconds: float = 30.0,
        lookback_hours: float = 0.0,
        max_cycle_age_hours: float = 8.0,
        rewrite_check_seconds: float = 300.0,
        workers: int = 24,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sources = tuple(sources)
        self.head_client = head_client or RequestsHeadClient()
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.lookback_hours = float(lookback_hours)
        self.max_cycle_age_hours = float(max_cycle_age_hours)
        self.rewrite_check_seconds = float(rewrite_check_seconds)
        self.workers = max(1, int(workers))
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self._db.close()

    def poll_once(
        self,
        *,
        now: str | pd.Timestamp | None = None,
        init_times: list[pd.Timestamp] | None = None,
    ) -> list[ProbeEvent]:
        observed = _utc(now or pd.Timestamp.now(tz="UTC"))
        if init_times is None:
            first = (observed - pd.Timedelta(hours=self.lookback_hours)).floor(
                "h"
            )
            init_values = list(
                pd.date_range(first, observed.floor("h"), freq="h")
            )
        else:
            init_values = [_utc(value).floor("h") for value in init_times]
        if init_times is None:
            oldest = observed - pd.Timedelta(
                hours=self.max_cycle_age_hours
            )
            existing = self._db.execute(
                "SELECT init_time FROM cycles"
            ).fetchall()
            init_values.extend(
                timestamp
                for timestamp in (
                    _utc(row["init_time"]) for row in existing
                )
                if oldest <= timestamp <= observed
            )
        requirements = {
            requirement.init_time: requirement
            for requirement in map(cycle_requirement, init_values)
        }
        for requirement in list(requirements.values()):
            if requirement.dependency_init is not None:
                dependency = cycle_requirement(requirement.dependency_init)
                requirements.setdefault(dependency.init_time, dependency)

        events: list[ProbeEvent] = []
        for requirement in sorted(
            requirements.values(),
            key=lambda item: item.init_time,
        ):
            self._ensure_cycle(requirement, observed)
            self._poll_requirement(requirement, observed, events)
        self._db.commit()
        return events

    def source_cycle_status(self) -> pd.DataFrame:
        rows = self._db.execute(
            """
            SELECT sc.source, sc.init_time, c.kind, c.left_censored,
                   sc.first_object_at, sc.terminal_seen_at,
                   sc.decoder_ready_at, sc.last_modified,
                   ROUND(
                     (julianday(sc.decoder_ready_at) - julianday(sc.init_time))
                     * 86400.0, 3
                   ) AS decoder_latency_seconds
            FROM source_cycles sc
            JOIN cycles c USING (init_time)
            ORDER BY sc.init_time, sc.source
            """
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def cycle_status(self) -> pd.DataFrame:
        rows = self._db.execute(
            "SELECT * FROM cycles ORDER BY init_time"
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def summary(self, *, include_censored: bool = False) -> pd.DataFrame:
        status = self.source_cycle_status()
        ready = (
            status[status["decoder_ready_at"].notna()].copy()
            if not status.empty
            else pd.DataFrame()
        )
        if not include_censored and not ready.empty:
            ready = ready[ready["left_censored"] == 0]
        frames: list[pd.DataFrame] = []
        if not ready.empty:
            ready["decoder_latency_minutes"] = (
                ready["decoder_latency_seconds"] / 60.0
            )
            frames.append(
                ready[["kind", "source", "decoder_latency_minutes"]]
            )
        cycles = self.cycle_status()
        if not cycles.empty:
            fastest = cycles[cycles["winner_at"].notna()].copy()
            if not include_censored:
                fastest = fastest[fastest["left_censored"] == 0]
            if not fastest.empty:
                fastest["source"] = "fastest_any"
                fastest["decoder_latency_minutes"] = [
                    (
                        _utc(winner_at) - _utc(init_time)
                    ).total_seconds()
                    / 60.0
                    for winner_at, init_time in zip(
                        fastest["winner_at"], fastest["init_time"]
                    )
                ]
                frames.append(
                    fastest[
                        ["kind", "source", "decoder_latency_minutes"]
                    ]
                )
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return (
            combined.groupby(["kind", "source"])["decoder_latency_minutes"]
            .agg(
                cycles="count",
                min="min",
                median="median",
                p90=lambda values: values.quantile(0.9),
                max="max",
            )
            .reset_index()
        )

    def recent_attempt_counts(
        self,
        *,
        since: str | pd.Timestamp,
    ) -> dict[str, int]:
        row = self._db.execute(
            """
            SELECT COUNT(*) AS requests,
                   SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN error <> '' THEN 1 ELSE 0 END) AS errors
            FROM attempts WHERE checked_at >= ?
            """,
            (_utc(since).isoformat(),),
        ).fetchone()
        return {
            "requests": int(row["requests"] or 0),
            "ok": int(row["ok"] or 0),
            "errors": int(row["errors"] or 0),
        }

    def revisions(self) -> pd.DataFrame:
        rows = self._db.execute(
            """
            SELECT source, init_time, fxx, observed_at,
                   previous_last_modified, new_last_modified,
                   previous_etag, new_etag
            FROM object_revisions
            ORDER BY observed_at
            """
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS cycles (
                init_time TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                dependency_init TEXT,
                tracking_started_at TEXT NOT NULL,
                tracking_delay_seconds REAL NOT NULL,
                left_censored INTEGER NOT NULL,
                winner_source TEXT,
                winner_at TEXT
            );
            CREATE TABLE IF NOT EXISTS source_cycles (
                source TEXT NOT NULL,
                init_time TEXT NOT NULL,
                first_object_at TEXT,
                terminal_seen_at TEXT,
                decoder_ready_at TEXT,
                last_modified TEXT,
                PRIMARY KEY (source, init_time)
            );
            CREATE TABLE IF NOT EXISTS objects (
                source TEXT NOT NULL,
                init_time TEXT NOT NULL,
                fxx INTEGER NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_modified TEXT,
                latest_last_modified TEXT,
                http_date TEXT,
                etag TEXT,
                latest_etag TEXT,
                content_length INTEGER,
                last_checked_at TEXT NOT NULL,
                PRIMARY KEY (source, init_time, fxx)
            );
            CREATE TABLE IF NOT EXISTS object_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                init_time TEXT NOT NULL,
                fxx INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                previous_last_modified TEXT,
                new_last_modified TEXT,
                previous_etag TEXT,
                new_etag TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checked_at TEXT NOT NULL,
                source TEXT NOT NULL,
                init_time TEXT NOT NULL,
                fxx INTEGER NOT NULL,
                status_code INTEGER,
                elapsed_ms REAL NOT NULL,
                error TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_attempts_checked
                ON attempts(checked_at);
            """
        )
        self._db.commit()

    def _ensure_cycle(
        self,
        requirement: CycleRequirement,
        observed: pd.Timestamp,
    ) -> None:
        init_iso = requirement.init_time.isoformat()
        delay = max(0.0, (observed - requirement.init_time).total_seconds())
        censor_after = max(120.0, self.poll_interval_seconds * 2.0)
        self._db.execute(
            """
            INSERT OR IGNORE INTO cycles
              (init_time, kind, dependency_init, tracking_started_at,
               tracking_delay_seconds, left_censored)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                init_iso,
                requirement.kind,
                (
                    requirement.dependency_init.isoformat()
                    if requirement.dependency_init is not None
                    else None
                ),
                observed.isoformat(),
                delay,
                int(delay > censor_after),
            ),
        )
        for source in self.sources:
            self._db.execute(
                """
                INSERT OR IGNORE INTO source_cycles (source, init_time)
                VALUES (?, ?)
                """,
                (source.name, init_iso),
            )

    def _poll_requirement(
        self,
        requirement: CycleRequirement,
        observed: pd.Timestamp,
        events: list[ProbeEvent],
    ) -> None:
        revision_tasks: list[tuple[SourceSpec, int, str]] = []
        for source in self.sources:
            if self._source_ready(
                source.name, requirement.init_time
            ) and self._revision_due(
                source.name,
                requirement.init_time,
                requirement.terminal_fxx,
                observed,
            ):
                revision_tasks.append(
                    (
                        source,
                        requirement.terminal_fxx,
                        source.idx_url(
                            requirement.init_time,
                            requirement.terminal_fxx,
                        ),
                    )
                )
        self._run_tasks(requirement, revision_tasks, observed, events)

        initial_tasks: list[tuple[SourceSpec, int, str]] = []
        for source in self.sources:
            if self._source_ready(source.name, requirement.init_time):
                continue
            for fxx in (1, requirement.terminal_fxx):
                if not self._object_seen(
                    source.name, requirement.init_time, fxx
                ):
                    initial_tasks.append(
                        (
                            source,
                            fxx,
                            source.idx_url(requirement.init_time, fxx),
                        )
                    )
        self._run_tasks(requirement, initial_tasks, observed, events)

        verification_tasks: list[tuple[SourceSpec, int, str]] = []
        for source in self.sources:
            if not self._object_seen(
                source.name,
                requirement.init_time,
                requirement.terminal_fxx,
            ):
                continue
            for fxx in requirement.own_fxx:
                if not self._object_seen(
                    source.name, requirement.init_time, fxx
                ):
                    verification_tasks.append(
                        (
                            source,
                            fxx,
                            source.idx_url(requirement.init_time, fxx),
                        )
                    )
        self._run_tasks(requirement, verification_tasks, observed, events)

        for source in self.sources:
            if self._source_ready(source.name, requirement.init_time):
                continue
            own_complete = self._object_count(
                source.name, requirement.init_time
            ) == len(requirement.own_fxx)
            dependency_complete = (
                requirement.dependency_init is None
                or self._source_ready(
                    source.name, requirement.dependency_init
                )
            )
            if own_complete and dependency_complete:
                ready_at = self._latest_relevant_seen(
                    source.name,
                    requirement,
                )
                self._mark_ready(source, requirement, ready_at, events)
        self._refresh_winner(requirement, events)

    def _run_tasks(
        self,
        requirement: CycleRequirement,
        tasks: list[tuple[SourceSpec, int, str]],
        observed: pd.Timestamp,
        events: list[ProbeEvent],
    ) -> None:
        if not tasks:
            return
        with ThreadPoolExecutor(
            max_workers=min(self.workers, len(tasks))
        ) as pool:
            futures = {
                pool.submit(self.head_client.head, url): (source, fxx)
                for source, fxx, url in tasks
            }
            for future in as_completed(futures):
                source, fxx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = HeadResult(
                        status_code=None,
                        elapsed_ms=0.0,
                        last_modified="",
                        http_date="",
                        etag="",
                        content_length=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                seen_at = _utc(result.observed_at or observed)
                self._record_attempt(
                    source,
                    requirement.init_time,
                    fxx,
                    seen_at,
                    result,
                )
                state = (
                    self._record_object(
                        source,
                        requirement,
                        fxx,
                        seen_at,
                        result,
                    )
                    if result.available
                    else "unavailable"
                )
                if state == "new":
                    event_type = (
                        "first_object"
                        if fxx == 1
                        else (
                            "terminal_seen"
                            if fxx == requirement.terminal_fxx
                            else "object_seen"
                        )
                    )
                    if event_type != "object_seen":
                        events.append(
                            _event(
                                event_type,
                                source,
                                requirement,
                                seen_at,
                                fxx=fxx,
                                detail=(
                                    f"Last-Modified={result.last_modified}"
                                ),
                            )
                        )
                elif state == "revision":
                    events.append(
                        _event(
                            "rewrite_detected",
                            source,
                            requirement,
                            seen_at,
                            fxx=fxx,
                            detail=(
                                f"new Last-Modified={result.last_modified} "
                                f"ETag={result.etag}"
                            ),
                        )
                    )

    def _record_attempt(
        self,
        source: SourceSpec,
        init_time: pd.Timestamp,
        fxx: int,
        checked_at: pd.Timestamp,
        result: HeadResult,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO attempts
              (checked_at, source, init_time, fxx, status_code,
               elapsed_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checked_at.isoformat(),
                source.name,
                init_time.isoformat(),
                fxx,
                result.status_code,
                result.elapsed_ms,
                result.error,
            ),
        )

    def _record_object(
        self,
        source: SourceSpec,
        requirement: CycleRequirement,
        fxx: int,
        seen_at: pd.Timestamp,
        result: HeadResult,
    ) -> str:
        existing = self._db.execute(
            """
            SELECT latest_last_modified, latest_etag
            FROM objects
            WHERE source = ? AND init_time = ? AND fxx = ?
            """,
            (
                source.name,
                requirement.init_time.isoformat(),
                fxx,
            ),
        ).fetchone()
        if existing is not None:
            previous_modified = str(
                existing["latest_last_modified"] or ""
            )
            previous_etag = str(existing["latest_etag"] or "")
            modified_changed = bool(
                result.last_modified
                and previous_modified
                and result.last_modified != previous_modified
            )
            etag_changed = bool(
                result.etag
                and previous_etag
                and result.etag != previous_etag
            )
            self._db.execute(
                """
                UPDATE objects
                SET last_checked_at = ?,
                    latest_last_modified = CASE WHEN ? <> '' THEN ?
                                                ELSE latest_last_modified END,
                    latest_etag = CASE WHEN ? <> '' THEN ?
                                       ELSE latest_etag END
                WHERE source = ? AND init_time = ? AND fxx = ?
                """,
                (
                    seen_at.isoformat(),
                    result.last_modified,
                    result.last_modified,
                    result.etag,
                    result.etag,
                    source.name,
                    requirement.init_time.isoformat(),
                    fxx,
                ),
            )
            if modified_changed or etag_changed:
                self._db.execute(
                    """
                    INSERT INTO object_revisions
                      (source, init_time, fxx, observed_at,
                       previous_last_modified, new_last_modified,
                       previous_etag, new_etag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source.name,
                        requirement.init_time.isoformat(),
                        fxx,
                        seen_at.isoformat(),
                        previous_modified,
                        result.last_modified,
                        previous_etag,
                        result.etag,
                    ),
                )
                return "revision"
            return "unchanged"
        cursor = self._db.execute(
            """
            INSERT OR IGNORE INTO objects
              (source, init_time, fxx, first_seen_at, last_modified,
               latest_last_modified, http_date, etag, latest_etag,
               content_length, last_checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source.name,
                requirement.init_time.isoformat(),
                fxx,
                seen_at.isoformat(),
                result.last_modified,
                result.last_modified,
                result.http_date,
                result.etag,
                result.etag,
                result.content_length,
                seen_at.isoformat(),
            ),
        )
        if cursor.rowcount == 0:
            return "unchanged"
        field = "first_object_at" if fxx == 1 else None
        if fxx == requirement.terminal_fxx:
            field = "terminal_seen_at"
        if field is not None:
            self._db.execute(
                f"""
                UPDATE source_cycles
                SET {field} = COALESCE({field}, ?),
                    last_modified = CASE WHEN ? = 'terminal_seen_at'
                                         THEN ? ELSE last_modified END
                WHERE source = ? AND init_time = ?
                """,
                (
                    seen_at.isoformat(),
                    field,
                    result.last_modified,
                    source.name,
                    requirement.init_time.isoformat(),
                ),
            )
        return "new"

    def _mark_ready(
        self,
        source: SourceSpec,
        requirement: CycleRequirement,
        ready_at: pd.Timestamp,
        events: list[ProbeEvent],
    ) -> None:
        self._db.execute(
            """
            UPDATE source_cycles SET decoder_ready_at = ?
            WHERE source = ? AND init_time = ? AND decoder_ready_at IS NULL
            """,
            (
                ready_at.isoformat(),
                source.name,
                requirement.init_time.isoformat(),
            ),
        )
        event = _event(
            "decoder_ready",
            source,
            requirement,
            ready_at,
            fxx=requirement.terminal_fxx,
        )
        events.append(event)

    def _refresh_winner(
        self,
        requirement: CycleRequirement,
        events: list[ProbeEvent],
    ) -> None:
        cycle = self._db.execute(
            "SELECT winner_source FROM cycles WHERE init_time = ?",
            (requirement.init_time.isoformat(),),
        ).fetchone()
        if cycle["winner_source"] is not None:
            return
        fastest = self._db.execute(
            """
            SELECT source, decoder_ready_at FROM source_cycles
            WHERE init_time = ? AND decoder_ready_at IS NOT NULL
            ORDER BY decoder_ready_at, source
            LIMIT 1
            """,
            (requirement.init_time.isoformat(),),
        ).fetchone()
        if fastest is None:
            return
        self._db.execute(
            """
            UPDATE cycles SET winner_source = ?, winner_at = ?
            WHERE init_time = ? AND winner_source IS NULL
            """,
            (
                fastest["source"],
                fastest["decoder_ready_at"],
                requirement.init_time.isoformat(),
            ),
        )
        source = next(
            item for item in self.sources if item.name == fastest["source"]
        )
        events.append(
            _event(
                "winner",
                source,
                requirement,
                _utc(fastest["decoder_ready_at"]),
                fxx=requirement.terminal_fxx,
            )
        )

    def _latest_relevant_seen(
        self,
        source: str,
        requirement: CycleRequirement,
    ) -> pd.Timestamp:
        timestamps: list[str] = []
        own = self._db.execute(
            """
            SELECT MAX(first_seen_at) AS seen FROM objects
            WHERE source = ? AND init_time = ?
            """,
            (source, requirement.init_time.isoformat()),
        ).fetchone()["seen"]
        if own:
            timestamps.append(str(own))
        if requirement.dependency_init is not None:
            dependency = self._db.execute(
                """
                SELECT decoder_ready_at AS seen FROM source_cycles
                WHERE source = ? AND init_time = ?
                """,
                (source, requirement.dependency_init.isoformat()),
            ).fetchone()
            if dependency and dependency["seen"]:
                timestamps.append(str(dependency["seen"]))
        return max(map(_utc, timestamps))

    def _object_seen(
        self,
        source: str,
        init_time: pd.Timestamp,
        fxx: int,
    ) -> bool:
        return (
            self._db.execute(
                """
                SELECT 1 FROM objects
                WHERE source = ? AND init_time = ? AND fxx = ?
                """,
                (source, init_time.isoformat(), fxx),
            ).fetchone()
            is not None
        )

    def _object_count(
        self,
        source: str,
        init_time: pd.Timestamp,
    ) -> int:
        return int(
            self._db.execute(
                """
                SELECT COUNT(*) AS count FROM objects
                WHERE source = ? AND init_time = ?
                """,
                (source, init_time.isoformat()),
            ).fetchone()["count"]
        )

    def _source_ready(
        self,
        source: str,
        init_time: pd.Timestamp,
    ) -> bool:
        row = self._db.execute(
            """
            SELECT decoder_ready_at FROM source_cycles
            WHERE source = ? AND init_time = ?
            """,
            (source, init_time.isoformat()),
        ).fetchone()
        return bool(row and row["decoder_ready_at"])

    def _revision_due(
        self,
        source: str,
        init_time: pd.Timestamp,
        fxx: int,
        observed: pd.Timestamp,
    ) -> bool:
        row = self._db.execute(
            """
            SELECT last_checked_at FROM objects
            WHERE source = ? AND init_time = ? AND fxx = ?
            """,
            (source, init_time.isoformat(), fxx),
        ).fetchone()
        if row is None:
            return False
        return (
            observed - _utc(row["last_checked_at"])
        ).total_seconds() >= self.rewrite_check_seconds


def _event(
    event_type: str,
    source: SourceSpec,
    requirement: CycleRequirement,
    observed_at: pd.Timestamp,
    *,
    fxx: int | None,
    detail: str = "",
) -> ProbeEvent:
    return ProbeEvent(
        event_type=event_type,
        source=source.name,
        init_time=requirement.init_time.isoformat(),
        fxx=fxx,
        observed_at=observed_at.isoformat(),
        latency_seconds=(
            observed_at - requirement.init_time
        ).total_seconds(),
        detail=detail,
    )


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None
