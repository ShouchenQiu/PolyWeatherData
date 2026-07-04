"""Continuously measure first-observed HRRR publication latency by mirror."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from data.forecast.realtime_availability import (
    DEFAULT_SOURCES,
    ProbeEvent,
    RealtimeAvailabilityProbe,
    RequestsHeadClient,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the forward-only probe")
    run.add_argument(
        "--db",
        type=Path,
        default=Path("data_store/probes/hrrr_availability.sqlite"),
    )
    run.add_argument(
        "--log",
        type=Path,
        help="rotating log path (default: <db>.log)",
    )
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--heartbeat-seconds", type=float, default=300.0)
    run.add_argument(
        "--lookback-hours",
        type=float,
        default=0.0,
        help="backfill hours on first start; 0 avoids censored startup traffic",
    )
    run.add_argument("--max-cycle-age-hours", type=float, default=8.0)
    run.add_argument("--rewrite-check-seconds", type=float, default=300.0)
    run.add_argument("--workers", type=int, default=24)
    run.add_argument("--timeout-seconds", type=float, default=10.0)
    run.add_argument(
        "--sources",
        default="aws,nomads,azure",
        help="comma-separated subset of aws,nomads,azure",
    )
    run.add_argument(
        "--once",
        action="store_true",
        help="perform one poll and exit (useful for connectivity checks)",
    )

    summary = subparsers.add_parser(
        "summary",
        help="print accumulated major/non-major latency statistics",
    )
    summary.add_argument(
        "--db",
        type=Path,
        default=Path("data_store/probes/hrrr_availability.sqlite"),
    )
    summary.add_argument(
        "--include-censored",
        action="store_true",
        help="include cycles that were already in progress when tracking began",
    )
    summary.add_argument(
        "--export-dir",
        type=Path,
        help="write cycles.csv, source_cycles.csv, and summary.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summary":
        return _summary(args)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive")
    if args.lookback_hours < 0:
        raise ValueError("--lookback-hours must be non-negative")
    if args.max_cycle_age_hours <= 0:
        raise ValueError("--max-cycle-age-hours must be positive")
    if args.rewrite_check_seconds <= 0:
        raise ValueError("--rewrite-check-seconds must be positive")
    source_names = [
        item.strip().lower()
        for item in str(args.sources).split(",")
        if item.strip()
    ]
    known = {source.name: source for source in DEFAULT_SOURCES}
    unknown = sorted(set(source_names) - set(known))
    if unknown:
        raise ValueError("unknown --sources: " + ", ".join(unknown))
    if not source_names:
        raise ValueError("--sources must contain at least one source")
    sources = [known[name] for name in source_names]

    log_path = args.log or args.db.with_suffix(".log")
    logger = _configure_logging(log_path)
    probe = RealtimeAvailabilityProbe(
        db_path=args.db,
        sources=sources,
        head_client=RequestsHeadClient(args.timeout_seconds),
        poll_interval_seconds=args.poll_seconds,
        lookback_hours=args.lookback_hours,
        max_cycle_age_hours=args.max_cycle_age_hours,
        rewrite_check_seconds=args.rewrite_check_seconds,
        workers=args.workers,
    )
    stop = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("[STOP] received signal=%s; finishing current poll", signum)
        stop.set()

    previous_handlers: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    logger.info(
        "[START] db=%s log=%s poll=%.1fs heartbeat=%.1fs "
        "lookback=%.1fh max_age=%.1fh rewrite_check=%.1fs "
        "workers=%d sources=%s",
        args.db,
        log_path,
        args.poll_seconds,
        args.heartbeat_seconds,
        args.lookback_hours,
        args.max_cycle_age_hours,
        args.rewrite_check_seconds,
        args.workers,
        ",".join(source_names),
    )
    for source in sources:
        logger.info("[SOURCE] %s %s", source.name, source.base_url)
    logger.info(
        "[NOTE] first-observed times are accurate to roughly one poll interval; "
        "left-censored cycles are excluded from the default summary"
    )

    last_heartbeat = pd.Timestamp.now(tz="UTC")
    known_cycles = set(probe.cycle_status().get("init_time", []))
    try:
        while not stop.is_set():
            poll_started = pd.Timestamp.now(tz="UTC")
            try:
                events = probe.poll_once(now=poll_started)
                cycles = probe.cycle_status()
                for row in cycles.itertuples(index=False):
                    if row.init_time in known_cycles:
                        continue
                    known_cycles.add(row.init_time)
                    logger.info(
                        "[TRACK] init=%s kind=%s delay=%.1fs censored=%s "
                        "dependency=%s",
                        row.init_time,
                        row.kind,
                        row.tracking_delay_seconds,
                        "yes" if row.left_censored else "no",
                        (
                            row.dependency_init
                            if pd.notna(row.dependency_init)
                            else "-"
                        ),
                    )
                for event in events:
                    logger.info(_format_event(event))
            except Exception:  # noqa: BLE001
                logger.exception("[POLL-ERROR] probe iteration failed; retrying")

            now = pd.Timestamp.now(tz="UTC")
            if (
                now - last_heartbeat
                >= pd.Timedelta(seconds=args.heartbeat_seconds)
                or args.once
            ):
                _log_heartbeat(probe, logger, last_heartbeat, now)
                last_heartbeat = now
            if args.once:
                break
            elapsed = (
                pd.Timestamp.now(tz="UTC") - poll_started
            ).total_seconds()
            stop.wait(max(0.0, args.poll_seconds - elapsed))
    finally:
        probe.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        logger.info("[STOPPED] probe database safely closed")
    return 0


def _summary(args: argparse.Namespace) -> int:
    probe = RealtimeAvailabilityProbe(db_path=args.db)
    try:
        summary = probe.summary(include_censored=args.include_censored)
        cycles = probe.cycle_status()
        source_cycles = probe.source_cycle_status()
        revisions = probe.revisions()
        if summary.empty:
            print("No decoder-ready uncensored cycles yet.")
        else:
            print(summary.to_string(index=False, float_format="%.2f"))
        if args.export_dir is not None:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            cycles.to_csv(args.export_dir / "cycles.csv", index=False)
            source_cycles.to_csv(
                args.export_dir / "source_cycles.csv",
                index=False,
            )
            summary.to_csv(args.export_dir / "summary.csv", index=False)
            revisions.to_csv(
                args.export_dir / "object_revisions.csv",
                index=False,
            )
            print(f"Exported probe evidence to {args.export_dir}")
    finally:
        probe.close()
    return 0


def _format_event(event: ProbeEvent) -> str:
    labels = {
        "first_object": "FIRST",
        "terminal_seen": "TAIL",
        "decoder_ready": "READY",
        "winner": "WINNER",
        "rewrite_detected": "REWRITE",
    }
    label = labels.get(event.event_type, event.event_type.upper())
    return (
        f"[{label}] source={event.source} init={event.init_time} "
        f"fxx={event.fxx if event.fxx is not None else '-'} "
        f"observed={event.observed_at} "
        f"lag={event.latency_seconds / 60.0:.2f}min "
        f"{event.detail}"
    ).rstrip()


def _log_heartbeat(
    probe: RealtimeAvailabilityProbe,
    logger: logging.Logger,
    since: pd.Timestamp,
    now: pd.Timestamp,
) -> None:
    counts = probe.recent_attempt_counts(since=since)
    status = probe.source_cycle_status()
    ready = (
        int(status["decoder_ready_at"].notna().sum())
        if not status.empty
        else 0
    )
    pending = len(status) - ready
    logger.info(
        "[HEARTBEAT] utc=%s requests=%d ok=%d unavailable=%d errors=%d "
        "source_cycles_ready=%d pending=%d",
        now.isoformat(),
        counts["requests"],
        counts["ok"],
        counts["requests"] - counts["ok"] - counts["errors"],
        counts["errors"],
        ready,
        pending,
    )
    summary = probe.summary()
    if not summary.empty:
        for row in summary.itertuples(index=False):
            logger.info(
                "[SUMMARY] kind=%s source=%s n=%d median=%.2fmin "
                "p90=%.2fmin min=%.2fmin max=%.2fmin",
                row.kind,
                row.source,
                row.cycles,
                row.median,
                row.p90,
                row.min,
                row.max,
            )


def _configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hrrr-availability-probe")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    rotating = RotatingFileHandler(
        path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
