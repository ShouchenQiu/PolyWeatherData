from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.stations import STATIONS
from data.storage.parquet_store import ParquetStore
from db.schema import DbClient
from scripts import seed_demo_data


class FakeDiscovery:
    last_params = None

    def __init__(self, db, settings, **_kwargs):
        self._db = db

    def discover(self, params=None):
        FakeDiscovery.last_params = params
        self._db.upsert_event(
            "cid-demo",
            "Will KLGA be warm?",
            "will-the-high-temp-at-lga-be-70f-or-higher",
            "2026-06-04",
            "KLGA",
            "discovered",
        )
        self._db.upsert_market(
            "tok-demo",
            "cid-demo",
            "will-the-high-temp-at-lga-be-70f-or-higher",
            70.0,
            float("inf"),
            "Yes",
        )
        return [{"condition_id": "cid-demo"}]


class FakeZarrDownloader:
    def __init__(self, store, station_config, **_kwargs):
        self._store = store
        self._station = station_config

    def run(self, start_date, end_date, init_hours=None):
        init = pd.Timestamp(f"{start_date} 00:00:00", tz="UTC")
        rows = [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": 273.15 + fxx,
            }
            for fxx in range(0, 49)
        ]
        return self._store.append(self._station.station_id, pd.DataFrame(rows))


class FakeIemDownloader:
    def __init__(self, store, station_config, **_kwargs):
        self._store = store
        self._station = station_config

    def run(self, start_date, end_date, station="LGA", network="NY_ASOS"):
        start = pd.Timestamp(f"{start_date} 00:00:00", tz="UTC")
        rows = [
            {"valid_time": start + pd.Timedelta(hours=i), "raw_tmpc_tenths": 20.0}
            for i in range(24)
        ]
        return self._store.append(self._station.station_id, pd.DataFrame(rows))


def test_seed_demo_dry_run_does_not_create_files(tmp_path):
    demo_dir = tmp_path / "demo"

    summary = seed_demo_data.seed_demo(
        demo_dir=demo_dir,
        dry_run=True,
        seed_date="2026-06-04",
    )

    assert summary["dry_run"] is True
    assert summary["db_path"] == str(demo_dir / "polyweather.db")
    assert summary["polymarket_slug"] is None
    assert not demo_dir.exists()


def test_seed_demo_writes_demo_db_and_bootstrap_stores(tmp_path):
    main_db = tmp_path / "polyweather.db"
    main_db.write_text("do-not-touch", encoding="ascii")
    demo_dir = tmp_path / "demo"

    summary = seed_demo_data.seed_demo(
        demo_dir=demo_dir,
        seed_date="2026-06-04",
        init_hour=0,
        discovery_cls=FakeDiscovery,
        zarr_downloader_cls=FakeZarrDownloader,
        iem_downloader_cls=FakeIemDownloader,
    )

    assert main_db.read_text(encoding="ascii") == "do-not-touch"
    assert summary["tables"]["events"] == 1
    assert summary["tables"]["markets"] == 1
    assert summary["stores"]["hrrr"]["rows"] == 1
    assert summary["stores"]["obs"]["rows"] == 24

    db = DbClient(demo_dir / "polyweather.db")
    try:
        assert db.query("SELECT COUNT(*) FROM stations")[0][0] == 1
        assert db.query("SELECT COUNT(*) FROM events")[0][0] == 1
        assert db.query("SELECT COUNT(*) FROM markets")[0][0] == 1
        assert db.query("SELECT status FROM events WHERE condition_id='cid-demo'")[0][0] == "active"
    finally:
        db.close()

    hrrr = ParquetStore(demo_dir / "data_store" / "hrrr").read(
        STATIONS["KLGA"].station_id,
        pd.Timestamp("2026-06-04", tz="UTC"),
        pd.Timestamp("2026-06-05", tz="UTC"),
    )
    assert list(hrrr["fxx"]) == [1]


def test_seed_demo_passes_exact_polymarket_slug(tmp_path):
    FakeDiscovery.last_params = None
    slug = "highest-temperature-in-nyc-on-june-5-2026"

    summary = seed_demo_data.seed_demo(
        demo_dir=tmp_path / "demo",
        seed_date="2026-06-04",
        polymarket_slug=slug,
        skip_hrrr=True,
        skip_iem=True,
        discovery_cls=FakeDiscovery,
    )

    assert FakeDiscovery.last_params == {"slug": slug}
    assert summary["polymarket_slug"] == slug
    assert summary["tables"]["events"] == 1


def test_seed_demo_can_skip_network_sections(tmp_path):
    class ShouldNotCall:
        def __init__(self, *args, **kwargs):
            raise AssertionError("network section should have been skipped")

    summary = seed_demo_data.seed_demo(
        demo_dir=tmp_path / "demo",
        seed_date="2026-06-04",
        skip_polymarket=True,
        skip_hrrr=True,
        skip_iem=True,
        discovery_cls=ShouldNotCall,
        zarr_downloader_cls=ShouldNotCall,
        iem_downloader_cls=ShouldNotCall,
    )

    assert summary["tables"]["stations"] == 1
    assert summary["discovered"] == 0
    assert summary["hrrr_written"] == 0
    assert summary["obs_written"] == 0
    assert summary["skipped"] == ["polymarket", "hrrr", "iem"]
