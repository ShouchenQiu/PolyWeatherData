from __future__ import annotations

import pandas as pd

from data.forecast.realtime_availability import (
    HeadResult,
    RealtimeAvailabilityProbe,
    SourceSpec,
    cycle_requirement,
)


def test_cycle_requirement_matches_major_and_non_major_decoder_contract() -> None:
    major = cycle_requirement(pd.Timestamp("2026-07-04T12:00:00Z"))
    non_major = cycle_requirement(pd.Timestamp("2026-07-04T13:00:00Z"))

    assert major.kind == "major"
    assert major.own_fxx == tuple(range(1, 49))
    assert major.dependency_init is None
    assert non_major.kind == "non_major"
    assert non_major.own_fxx == tuple(range(1, 19))
    assert non_major.dependency_init == pd.Timestamp(
        "2026-07-04T12:00:00Z"
    )


class _AvailableHeadClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_modified = "Sat, 04 Jul 2026 13:45:00 GMT"

    def head(self, url: str) -> HeadResult:
        self.calls.append(url)
        return HeadResult(
            status_code=200,
            elapsed_ms=12.0,
            last_modified=self.last_modified,
            http_date="Sat, 04 Jul 2026 13:50:00 GMT",
            etag="etag",
            content_length=100,
            error="",
        )


def test_probe_records_decoder_ready_once_and_resumes_without_reprobing(
    tmp_path,
) -> None:
    client = _AvailableHeadClient()
    probe = RealtimeAvailabilityProbe(
        db_path=tmp_path / "probe.sqlite",
        sources=[SourceSpec("aws", "https://example.test")],
        head_client=client,
        poll_interval_seconds=30,
        lookback_hours=3,
        workers=4,
    )
    init = pd.Timestamp("2026-07-04T12:00:00Z")
    observed = pd.Timestamp("2026-07-04T13:50:00Z")

    first_events = probe.poll_once(
        now=observed,
        init_times=[init],
    )
    calls_after_first = len(client.calls)
    second_events = probe.poll_once(
        now=observed + pd.Timedelta(seconds=30),
        init_times=[init],
    )

    status = probe.source_cycle_status()
    assert calls_after_first == 48
    assert len(client.calls) == calls_after_first
    assert second_events == []
    assert status.loc[0, "source"] == "aws"
    assert status.loc[0, "decoder_ready_at"] == observed.isoformat()
    assert status.loc[0, "decoder_latency_seconds"] == 6600.0
    assert any(event.event_type == "decoder_ready" for event in first_events)


def test_non_major_ready_waits_for_own_f18_and_prior_major_f48(
    tmp_path,
) -> None:
    client = _AvailableHeadClient()
    probe = RealtimeAvailabilityProbe(
        db_path=tmp_path / "probe.sqlite",
        sources=[SourceSpec("nomads", "https://example.test")],
        head_client=client,
        workers=4,
    )

    probe.poll_once(
        now=pd.Timestamp("2026-07-04T13:50:00Z"),
        init_times=[pd.Timestamp("2026-07-04T13:00:00Z")],
    )

    status = probe.source_cycle_status().set_index("init_time")
    dependency = status.loc["2026-07-04T12:00:00+00:00"]
    non_major = status.loc["2026-07-04T13:00:00+00:00"]
    assert dependency["kind"] == "major"
    assert non_major["kind"] == "non_major"
    assert dependency["decoder_ready_at"] is not None
    assert non_major["decoder_ready_at"] is not None
    assert len(client.calls) == 48 + 18


def test_probe_records_later_terminal_object_rewrite(tmp_path) -> None:
    client = _AvailableHeadClient()
    probe = RealtimeAvailabilityProbe(
        db_path=tmp_path / "probe.sqlite",
        sources=[SourceSpec("azure", "https://example.test")],
        head_client=client,
        rewrite_check_seconds=300,
        workers=4,
    )
    init = pd.Timestamp("2026-07-04T12:00:00Z")
    first_seen = pd.Timestamp("2026-07-04T13:50:00Z")
    probe.poll_once(now=first_seen, init_times=[init])
    client.last_modified = "Sat, 04 Jul 2026 14:10:00 GMT"

    events = probe.poll_once(
        now=first_seen + pd.Timedelta(seconds=301),
        init_times=[init],
    )

    revisions = probe.revisions()
    assert len(client.calls) == 49
    assert [event.event_type for event in events] == ["rewrite_detected"]
    assert revisions.loc[0, "source"] == "azure"
    assert revisions.loc[0, "previous_last_modified"].endswith("13:45:00 GMT")
    assert revisions.loc[0, "new_last_modified"].endswith("14:10:00 GMT")


def test_summary_separates_each_source_and_fastest_available(tmp_path) -> None:
    probe = RealtimeAvailabilityProbe(
        db_path=tmp_path / "probe.sqlite",
        sources=[
            SourceSpec("aws", "https://aws.test"),
            SourceSpec("nomads", "https://nomads.test"),
        ],
        head_client=_AvailableHeadClient(),
        workers=8,
    )
    init = pd.Timestamp("2026-07-04T12:00:00Z")
    probe.poll_once(
        now=init + pd.Timedelta(seconds=60),
        init_times=[init],
    )

    summary = probe.summary()

    assert set(summary["source"]) == {"aws", "nomads", "fastest_any"}
    assert set(summary["kind"]) == {"major"}
    assert set(summary["median"]) == {1.0}
