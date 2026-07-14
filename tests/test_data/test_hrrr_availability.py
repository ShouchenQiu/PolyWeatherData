from __future__ import annotations

import pandas as pd

from data.forecast.availability import (
    HrrrAvailabilityDataset,
    HrrrS3ObjectSource,
    build_hrrr_availability,
    build_virtual_tft_availability,
)


def _cycle_objects(
    init: str,
    max_fxx: int,
    *,
    last_fxx_at: str,
) -> list[dict]:
    stamp = pd.Timestamp(init)
    date = stamp.strftime("%Y%m%d")
    hour = stamp.strftime("%H")
    rows = []
    for fxx in range(1, max_fxx + 1):
        modified = (
            pd.Timestamp(last_fxx_at)
            if fxx == max_fxx
            else stamp + pd.Timedelta(minutes=50)
        )
        rows.append(
            {
                "key": (
                    f"hrrr.{date}/conus/"
                    f"hrrr.t{hour}z.wrfsfcf{fxx:02d}.grib2"
                ),
                "last_modified": modified,
            }
        )
    return rows


def test_availability_waits_for_major_and_non_major_decoder_dependencies() -> None:
    objects = pd.DataFrame(
        [
            *_cycle_objects(
                "2026-06-30T12:00:00Z",
                48,
                last_fxx_at="2026-06-30T13:46:50Z",
            ),
            *_cycle_objects(
                "2026-06-30T13:00:00Z",
                18,
                last_fxx_at="2026-06-30T14:24:43Z",
            ),
        ]
    )

    availability = build_hrrr_availability(
        objects,
        init_times=pd.to_datetime(
            ["2026-06-30T12:00:00Z", "2026-06-30T13:00:00Z"]
        ),
        compute_budget="5min",
        decision_tick="5min",
    )

    major = availability.iloc[0]
    assert major["complete"]
    assert major["own_max_fxx"] == 48
    assert pd.isna(major["dependency_major_init"])
    assert major["source_ready_at"] == pd.Timestamp("2026-06-30T13:46:50Z")
    assert major["available_at"] == pd.Timestamp("2026-06-30T13:55:00Z")

    non_major = availability.iloc[1]
    assert non_major["complete"]
    assert non_major["own_max_fxx"] == 18
    assert non_major["dependency_major_init"] == pd.Timestamp(
        "2026-06-30T12:00:00Z"
    )
    assert non_major["source_ready_at"] == pd.Timestamp(
        "2026-06-30T14:24:43Z"
    )
    assert non_major["available_at"] == pd.Timestamp(
        "2026-06-30T14:30:00Z"
    )


def test_availability_marks_cycle_incomplete_when_required_object_is_missing() -> None:
    objects = pd.DataFrame(
        _cycle_objects(
            "2026-06-30T12:00:00Z",
            47,
            last_fxx_at="2026-06-30T13:45:00Z",
        )
    )

    availability = build_hrrr_availability(
        objects,
        init_times=[pd.Timestamp("2026-06-30T12:00:00Z")],
    )

    row = availability.iloc[0]
    assert not row["complete"]
    assert pd.isna(row["source_ready_at"])
    assert pd.isna(row["available_at"])
    assert row["missing_objects"] == [
        "2026-06-30T12:00:00+00:00/f48"
    ]


def test_virtual_tft_availability_uses_fixed_latency_without_hrrr_dependencies() -> None:
    availability = build_virtual_tft_availability(
        init_times=pd.to_datetime(
            ["2026-06-30T12:01:00Z", "2026-06-30T13:00:00Z"]
        ),
        latency="2h",
        decision_tick="5min",
    )

    assert availability["available_at"].tolist() == [
        pd.Timestamp("2026-06-30T14:05:00Z"),
        pd.Timestamp("2026-06-30T15:00:00Z"),
    ]
    assert availability["complete"].tolist() == [True, True]
    assert availability["required_object_count"].tolist() == [0, 0]
    assert availability["missing_objects"].tolist() == [[], []]
    assert availability["source_ready_at"].tolist() == [
        pd.Timestamp("2026-06-30T12:01:00Z"),
        pd.Timestamp("2026-06-30T13:00:00Z"),
    ]


class _XmlTransport:
    def __init__(self, pages: list[str]) -> None:
        self.pages = list(pages)
        self.calls: list[dict] = []

    def get_text(self, url: str, params: dict) -> str:
        self.calls.append(dict(params))
        return self.pages.pop(0)


def test_s3_object_source_follows_list_objects_continuation_token() -> None:
    page_one = """\
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>next-page</NextContinuationToken>
  <Contents>
    <Key>hrrr.20260630/conus/hrrr.t12z.wrfsfcf01.grib2</Key>
    <LastModified>2026-06-30T12:53:11.000Z</LastModified>
  </Contents>
  <Contents>
    <Key>hrrr.20260630/conus/hrrr.t12z.wrfsfcf01.grib2.idx</Key>
    <LastModified>2026-06-30T12:53:10.000Z</LastModified>
  </Contents>
</ListBucketResult>
"""
    page_two = """\
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>hrrr.20260630/conus/hrrr.t12z.wrfsfcf48.grib2</Key>
    <LastModified>2026-06-30T13:46:50.000Z</LastModified>
  </Contents>
</ListBucketResult>
"""
    transport = _XmlTransport([page_one, page_two])

    objects = HrrrS3ObjectSource(transport=transport).fetch_day("2026-06-30")

    assert objects["key"].tolist() == [
        "hrrr.20260630/conus/hrrr.t12z.wrfsfcf01.grib2",
        "hrrr.20260630/conus/hrrr.t12z.wrfsfcf48.grib2",
    ]
    assert objects["last_modified"].tolist() == [
        pd.Timestamp("2026-06-30T12:53:11Z"),
        pd.Timestamp("2026-06-30T13:46:50Z"),
    ]
    assert transport.calls == [
        {
            "list-type": "2",
            "prefix": "hrrr.20260630/conus/",
        },
        {
            "list-type": "2",
            "prefix": "hrrr.20260630/conus/",
            "continuation-token": "next-page",
        },
    ]


def test_availability_dataset_persists_and_verifies_manifest(tmp_path) -> None:
    objects = pd.DataFrame(
        _cycle_objects(
            "2026-06-30T12:00:00Z",
            48,
            last_fxx_at="2026-06-30T13:46:50Z",
        )
    )
    cycles = build_hrrr_availability(
        objects,
        init_times=[pd.Timestamp("2026-06-30T12:00:00Z")],
    )
    dataset = HrrrAvailabilityDataset(
        objects=objects,
        cycles=cycles,
        start_date="2026-06-30",
        end_date="2026-06-30",
        compute_budget="5min",
        decision_tick="5min",
    )
    output = tmp_path / "availability"

    manifest = dataset.save(
        output,
        fetched_at="2026-07-02T12:00:00Z",
    )
    loaded = HrrrAvailabilityDataset.load(output)

    assert manifest["data_class"] == "hrrr_point_in_time_availability"
    assert manifest["row_counts"] == {"objects": 48, "cycles": 1}
    assert manifest["incomplete_init_times"] == []
    assert loaded.cycles["available_at"].tolist() == [
        pd.Timestamp("2026-06-30T13:55:00Z")
    ]
