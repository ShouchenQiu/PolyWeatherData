"""Seed a standalone demo DB and tiny weather stores for UI smoke testing.

This script is intentionally separate from the production bot.  It writes only
under the provided demo directory, so the root ``polyweather.db`` remains clean.

Example:
    python3 -m scripts.seed_demo_data --demo-dir .demo

Then run the UI against the demo DB:
    POLYWEATHER_DB=.demo/polyweather.db \
    POLYWEATHER_DATA_STORE=.demo/data_store \
    python3 -m streamlit run ui/app.py
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings
from config.stations import STATIONS
from data.clob.orderbook import OrderbookFetcher
from data.storage.parquet_store import ParquetStore
from db.schema import DbClient
from events.discovery import EventDiscovery
from scripts.batch_download_iem import BatchIemDownloader
from scripts.batch_download_zarr import BatchZarrDownloader

_TABLES = (
    "stations",
    "events",
    "markets",
    "forecast_snapshots",
    "forecast_bucket_probs",
    "signals",
    "orders",
    "pnl_snapshots",
    "loop_ticks",
)
_CLOB_BOOK_TIMEOUT_S = 10.0


def seed_demo(
    demo_dir: Path | str = Path(".demo"),
    station_id: str = "KLGA",
    seed_date: str | None = None,
    init_hour: int | None = 0,
    polymarket_slug: str | None = None,
    dry_run: bool = False,
    skip_polymarket: bool = False,
    skip_hrrr: bool = False,
    skip_iem: bool = False,
    discovery_cls: Any = EventDiscovery,
    zarr_downloader_cls: Any = BatchZarrDownloader,
    iem_downloader_cls: Any = BatchIemDownloader,
    public_client_cls: Any | None = None,
) -> dict:
    """Create a demo DB plus tiny HRRR/IEM stores and return a row-count summary."""
    demo_root = Path(demo_dir)
    station = STATIONS[station_id]
    day = seed_date or _default_seed_date()
    db_path = demo_root / "polyweather.db"
    data_store_dir = demo_root / "data_store"
    settings = Settings(data_store_dir=data_store_dir)

    plan = {
        "dry_run": dry_run,
        "demo_dir": str(demo_root),
        "db_path": str(db_path),
        "data_store_dir": str(data_store_dir),
        "station_id": station_id,
        "seed_date": day,
        "init_hour": init_hour,
        "polymarket_slug": polymarket_slug,
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan

    demo_root.mkdir(parents=True, exist_ok=True)
    db = DbClient(db_path)
    try:
        db.upsert_station(station.station_id, station.lat, station.lon, station.timezone)

        hrrr_store = ParquetStore(settings.forecast_store_dir)
        obs_store = ParquetStore(settings.obs_store_dir)
        skipped = []
        timings: dict[str, dict[str, Any]] = {}

        if skip_polymarket:
            discovered = []
            skipped.append("polymarket")
            timings["polymarket"] = _timing(0.0, 0, 0)
        else:
            t0 = time.perf_counter()
            try:
                discovery = discovery_cls(db, settings)
                discovery_params = (
                    {"slug": polymarket_slug}
                    if polymarket_slug
                    else {"active": "true"}
                )
                discovered = discovery.discover(discovery_params)
                timings["polymarket"] = _timing(time.perf_counter() - t0, len(discovered), 0)
            except Exception:
                timings["polymarket"] = _timing(time.perf_counter() - t0, 0, 1)
                raise
            _activate_demo_events(db)

        if skip_polymarket:
            clob_written = 0
            timings["clob"] = _timing(0.0, 0, 0)
        else:
            clob_written, clob_errors, clob_elapsed = _seed_clob_snapshots(
                db,
                public_client_cls=public_client_cls,
            )
            timings["clob"] = _timing(clob_elapsed, clob_written, clob_errors)

        if skip_hrrr:
            hrrr_written = 0
            skipped.append("hrrr")
            timings["hrrr"] = _timing(0.0, 0, 0)
        else:
            t0 = time.perf_counter()
            zarr = zarr_downloader_cls(
                hrrr_store,
                station_config=station,
                max_workers=1,
            )
            hrrr_init_hours = None if init_hour is None else [init_hour]
            hrrr_written = zarr.run(day, day, init_hours=hrrr_init_hours)
            timings["hrrr"] = _timing(time.perf_counter() - t0, hrrr_written, 0)

        if skip_iem:
            obs_written = 0
            skipped.append("iem")
            timings["iem"] = _timing(0.0, 0, 0)
        else:
            t0 = time.perf_counter()
            iem = iem_downloader_cls(
                obs_store,
                station_config=station,
                delay=0.0,
            )
            iem_station = station.station_id[1:] if station.station_id.startswith("K") else station.station_id
            obs_written = iem.run(day, day, station=iem_station, network=station.iem_network)
            timings["iem"] = _timing(time.perf_counter() - t0, obs_written, 0)

        summary = {
            **plan,
            "discovered": len(discovered),
            "clob_written": clob_written,
            "hrrr_written": hrrr_written,
            "obs_written": obs_written,
            "skipped": skipped,
            "timings": timings,
            "tables": _table_counts(db),
            "stores": {
                "hrrr": _store_summary(hrrr_store, station.station_id),
                "obs": _store_summary(obs_store, station.station_id),
            },
        }
        (demo_root / "seed_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="ascii",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        db.close()


def _default_seed_date() -> str:
    return (pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=1)).date().isoformat()


def _table_counts(db: DbClient) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _TABLES:
        counts[table] = int(db.query(f"SELECT COUNT(*) FROM {table}")[0][0])
    return counts


def _activate_demo_events(db: DbClient) -> None:
    """Make discovered demo markets visible on the Overview page."""
    for ev in db.get_non_terminal_events():
        if ev.get("status") == "discovered":
            db.update_event_status(str(ev["condition_id"]), "active")


def _timing(elapsed_seconds: float, rows_written: int, errors: int) -> dict[str, Any]:
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "rows_written": int(rows_written),
        "errors": int(errors),
        "rows_per_second": (
            float(rows_written) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
    }


def _seed_clob_snapshots(
    db: DbClient,
    public_client_cls: Any | None = None,
) -> tuple[int, int, float]:
    t0 = time.perf_counter()
    try:
        client_cls = public_client_cls
        if client_cls is None:
            try:
                from polymarket import PublicClient as client_cls  # type: ignore[assignment,import-untyped]
            except ImportError:
                from polymarket_client import PublicClient as client_cls  # type: ignore[assignment,import-untyped]
        fetcher = OrderbookFetcher(client_cls())
    except Exception:
        return 0, 1, time.perf_counter() - t0

    rows = db.query(
        "SELECT token_id, condition_id FROM markets WHERE discovery_status='parsed'"
    )
    tick_utc = datetime.now(timezone.utc).isoformat()
    written = 0
    errors = 0
    for row in rows:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(fetcher.get_book, str(row["token_id"]))
            book = fut.result(timeout=_CLOB_BOOK_TIMEOUT_S)
            spread = (
                book["best_ask"] - book["best_bid"]
                if book.get("best_bid") is not None and book.get("best_ask") is not None
                else None
            )
            db.record_market_price_snapshot(
                str(row["token_id"]),
                str(row["condition_id"]),
                tick_utc,
                best_bid=book.get("best_bid"),
                best_ask=book.get("best_ask"),
                mid_price=book.get("mid_price"),
                spread=spread,
                depth_bid=sum(float(level.get("size", 0.0)) for level in book.get("bids", [])),
                depth_ask=sum(float(level.get("size", 0.0)) for level in book.get("asks", [])),
            )
            written += 1
        except TimeoutError:
            errors += 1
        except Exception:
            errors += 1
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    return written, errors, time.perf_counter() - t0


def _store_summary(store: ParquetStore, station_id: str) -> dict[str, Any]:
    df = store.read(
        station_id,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        end=pd.Timestamp("2100-01-01", tz="UTC"),
    )
    latest = None
    if not df.empty:
        time_col = "valid_time" if "valid_time" in df.columns else "init_time"
        latest = str(pd.to_datetime(df[time_col], utc=True).max())
    return {"rows": int(len(df)), "latest_time": latest}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Seed standalone demo data for the Streamlit UI")
    parser.add_argument("--demo-dir", default=".demo")
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS.keys()))
    parser.add_argument("--date", default=None, help="UTC date YYYY-MM-DD; default yesterday")
    parser.add_argument("--init-hour", type=int, default=0)
    parser.add_argument(
        "--all-init-hours",
        action="store_true",
        help="Download all 24 HRRR init hours for the selected date",
    )
    parser.add_argument(
        "--polymarket-slug",
        default=None,
        help="Optional exact Gamma event slug for deterministic Polymarket seeding",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-polymarket", action="store_true")
    parser.add_argument("--skip-hrrr", action="store_true")
    parser.add_argument("--skip-iem", action="store_true")
    args = parser.parse_args()

    seed_demo(
        demo_dir=Path(args.demo_dir),
        station_id=args.station,
        seed_date=args.date,
        init_hour=None if args.all_init_hours else args.init_hour,
        polymarket_slug=args.polymarket_slug,
        dry_run=args.dry_run,
        skip_polymarket=args.skip_polymarket,
        skip_hrrr=args.skip_hrrr,
        skip_iem=args.skip_iem,
    )


if __name__ == "__main__":
    _cli()
