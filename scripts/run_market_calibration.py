"""Run the governed KLGA historical transaction-calibration study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.market_calibration import (
    run_market_calibration_analysis,
    write_market_calibration_artifact,
)
from backtest.synthetic_quotes import SyntheticQuoteDataset
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset, HistoricalDataIncomplete
from data.clob.quality import build_clob_audit_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-event-days", type=int, default=30)
    parser.add_argument("--min-condition-cells", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(
            f"immutable market-calibration output exists: {args.output}"
        )
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if min(args.min_event_days, args.min_condition_cells) <= 0:
        raise ValueError("minimum inference counts must be positive")
    print(
        f"[start] market calibration station={args.station} output={args.output}",
        flush=True,
    )
    validated = _load_and_audit_inputs(
        clob_root=args.clob_root,
        actual_highs=args.actual_highs,
        quotes_root=args.quotes_root,
        station_id=args.station,
    )
    clob, actuals, truth_exclusions, quotes, audit, source_sha256 = validated
    print(
        f"[complete] input audit events={len(clob.events)} "
        f"markets={len(clob.markets)} trades={len(clob.trades)} "
        f"truth_exclusions={len(truth_exclusions)}",
        flush=True,
    )
    events, markets, trades = _eligible_clob_frames(clob, audit.daily_coverage)
    analysis = run_market_calibration_analysis(
        events=events,
        markets=markets,
        trades=trades,
        actuals=actuals,
        quotes=quotes.quotes,
        timezone=STATIONS[args.station].timezone,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_event_days=args.min_event_days,
        min_condition_cells=args.min_condition_cells,
        progress_callback=lambda stage, rows: print(
            f"[complete] {stage} rows={rows}", flush=True
        ),
    )
    frames = {
        **analysis.frames,
        "input_audit_checks": audit.checks,
        "input_audit_event_checks": audit.event_checks,
        "input_daily_coverage": audit.daily_coverage,
        "input_quote_checks": audit.quote_checks,
        "input_truth_exclusions": truth_exclusions,
    }
    report = {
        **analysis.report,
        "input_audit": {
            "passed": audit.passed,
            "failures": list(audit.failures),
            "summary": audit.summary,
        },
        "eligible_input_rows": {
            "events": len(events),
            "markets": len(markets),
            "trades": len(trades),
        },
        "truth_exclusions": {
            "count": len(truth_exclusions),
            "target_dates": truth_exclusions["target_date"].astype(str).tolist(),
        },
    }
    truth_limitations = [
        {
            "artifact": "input_truth_exclusions.parquet",
            "excluded_event_days": int(len(group)),
            "reason": str(reason),
        }
        for reason, group in truth_exclusions.groupby(
            "exclusion_reason",
            sort=True,
        )
    ]
    question_ledger = {
        question: {
            **entry,
            "data_contract_limitations": truth_limitations,
        }
        for question, entry in analysis.question_ledger.items()
    }
    manifest = write_market_calibration_artifact(
        output=args.output,
        frames=frames,
        report=report,
        question_ledger=question_ledger,
        source_sha256=source_sha256,
        protocol={
            "station_id": args.station,
            "split_role": "oof_development",
            "holdout_consumed": False,
            "settlement_boundary": "end_of_station_local_target_day",
            "market_close_gate": "timestamp_before_contract_closed_at",
            "price_bin_widths": [0.10, 0.05, 0.02, 0.01],
            "primary_weighting": "event_contract_equal",
            "paper_comparator_weighting": "fill",
            "multiplicity": "bonferroni_event_cluster_bootstrap",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "min_event_days": args.min_event_days,
            "min_condition_cells": args.min_condition_cells,
            "execution_evidence": "observed_buy_fills_only",
            "executable_edge": "not_measured_without_historical_l2",
            "truth_coverage": (
                "complete_station_local_day_or_explicit_exclusion"
            ),
        },
        html_artifacts=analysis.html_artifacts,
    )
    print(
        f"[complete] artifact files={len(manifest['sha256'])} "
        f"output={args.output}",
        flush=True,
    )
    print(f"[done] market calibration output={args.output}", flush=True)
    return 0


def _load_and_audit_inputs(
    *,
    clob_root: Path,
    actual_highs: Path,
    quotes_root: Path,
    station_id: str,
) -> tuple[
    HistoricalClobDataset,
    pd.DataFrame,
    pd.DataFrame,
    SyntheticQuoteDataset,
    object,
    dict[str, str],
]:
    clob_manifest_path = clob_root / "manifest.json"
    clob_manifest = _json(clob_manifest_path)
    if clob_manifest.get("schema_version") != 2:
        raise HistoricalDataIncomplete(
            "formal market calibration requires schema-v2 CLOB resolution fields"
        )
    actual_manifest_path = actual_highs.with_suffix(
        actual_highs.suffix + ".manifest.json"
    )
    actual_manifest = _json(actual_manifest_path)
    if (
        actual_manifest.get("schema_version") != 2
        or actual_manifest.get("hour_coverage_contract")
        != "complete_station_local_day_or_explicit_exclusion"
        or actual_manifest.get("daily_high_source")
        != "weather_com_raw_observation_max"
    ):
        raise HistoricalDataIncomplete(
            "formal market calibration requires schema-v2 complete local-day truth "
            "with explicit exclusions and raw-observation daily highs"
        )
    if str(actual_manifest.get("station_id")) != station_id:
        raise HistoricalDataIncomplete("actual-high station does not match study")
    if list(actual_manifest.get("requested_range", [])) != list(
        clob_manifest.get("requested_range", [])
    ):
        raise HistoricalDataIncomplete(
            "actual-high and CLOB requested ranges do not match"
        )
    expected_actual_sha = actual_manifest.get("sha256", {}).get(actual_highs.name)
    if expected_actual_sha != _sha256(actual_highs):
        raise HistoricalDataIncomplete("actual-high artifact checksum mismatch")
    exclusions_name = str(actual_manifest.get("truth_exclusions_file", ""))
    if not exclusions_name or Path(exclusions_name).name != exclusions_name:
        raise HistoricalDataIncomplete("invalid truth-exclusions artifact name")
    truth_exclusions_path = actual_highs.parent / exclusions_name
    expected_exclusions_sha = actual_manifest.get("sha256", {}).get(
        exclusions_name
    )
    if (
        not truth_exclusions_path.exists()
        or expected_exclusions_sha != _sha256(truth_exclusions_path)
    ):
        raise HistoricalDataIncomplete("truth-exclusions artifact checksum mismatch")
    quote_manifest_path = quotes_root / "manifest.json"
    quote_manifest = _json(quote_manifest_path)
    clob_manifest_sha = _sha256(clob_manifest_path)
    if quote_manifest.get("source_sha256") != clob_manifest_sha:
        raise HistoricalDataIncomplete(
            "synthetic quotes are not bound to the schema-v2 CLOB manifest"
        )
    clob = HistoricalClobDataset.load(clob_root)
    if clob.station_id != station_id:
        raise HistoricalDataIncomplete("CLOB station does not match study")
    actuals = pd.read_csv(actual_highs)
    if int(actual_manifest.get("row_count", -1)) != len(actuals):
        raise HistoricalDataIncomplete("actual-high manifest row count mismatch")
    truth_exclusions = pd.read_parquet(truth_exclusions_path)
    required_exclusions = {
        "station_id",
        "target_date",
        "observed_hours",
        "expected_hours",
        "exclusion_reason",
    }
    if not required_exclusions.issubset(truth_exclusions.columns):
        raise HistoricalDataIncomplete("truth-exclusions artifact missing columns")
    if int(actual_manifest.get("exclusion_count", -1)) != len(truth_exclusions):
        raise HistoricalDataIncomplete("truth-exclusions manifest row count mismatch")
    if (
        truth_exclusions["station_id"].astype(str).ne(station_id).any()
        or truth_exclusions["exclusion_reason"].astype(str).str.len().eq(0).any()
    ):
        raise HistoricalDataIncomplete("invalid truth-exclusions rows")
    _validate_truth_date_partition(actuals, truth_exclusions, actual_manifest)
    quotes = SyntheticQuoteDataset.load(quotes_root)
    exclusions = {
        str(day): "Polymarket provider has no complete event artifact"
        for day in clob_manifest.get("incomplete_target_dates", [])
    }
    truth_exclusion_reasons = {
        str(row.target_date): str(row.exclusion_reason)
        for row in truth_exclusions.itertuples(index=False)
    }
    audit = build_clob_audit_report(
        clob,
        exclusions=exclusions,
        analysis_exclusions=truth_exclusion_reasons,
        quotes=quotes.quotes,
        actual_highs=actuals,
        clob_manifest_sha256=clob_manifest_sha,
        quote_source_sha256=str(quote_manifest.get("source_sha256")),
        quote_max_staleness=str(quote_manifest.get("max_staleness")),
        expected_row_counts=clob_manifest.get("row_counts"),
    )
    if not audit.passed:
        raise HistoricalDataIncomplete(
            "market-calibration input audit failed: " + ", ".join(audit.failures)
        )
    return (
        clob,
        actuals,
        truth_exclusions,
        quotes,
        audit,
        {
            "clob_manifest": clob_manifest_sha,
            "actuals_manifest": _sha256(actual_manifest_path),
            "quotes_manifest": _sha256(quote_manifest_path),
        },
    )


def _validate_truth_date_partition(
    actuals: pd.DataFrame,
    truth_exclusions: pd.DataFrame,
    manifest: dict,
) -> None:
    if "target_date" not in actuals.columns:
        raise HistoricalDataIncomplete("actual-high artifact missing target_date")
    actual_dates = actuals["target_date"].astype(str)
    exclusion_dates = truth_exclusions["target_date"].astype(str)
    if actual_dates.duplicated().any() or exclusion_dates.duplicated().any():
        raise HistoricalDataIncomplete("truth date partition contains duplicates")
    if set(actual_dates).intersection(exclusion_dates):
        raise HistoricalDataIncomplete("truth date is both eligible and excluded")
    requested_range = list(manifest.get("requested_range", []))
    if len(requested_range) != 2:
        raise HistoricalDataIncomplete("truth manifest requested range is invalid")
    expected_dates = {
        stamp.date().isoformat()
        for stamp in pd.date_range(requested_range[0], requested_range[1], freq="D")
    }
    partitioned_dates = set(actual_dates) | set(exclusion_dates)
    if partitioned_dates != expected_dates:
        raise HistoricalDataIncomplete(
            "truth eligible/excluded dates do not partition requested range"
        )
    if int(manifest.get("requested_day_count", -1)) != len(expected_dates):
        raise HistoricalDataIncomplete("truth manifest requested day count mismatch")


def _eligible_clob_frames(
    clob: HistoricalClobDataset,
    daily_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible_dates = set(
        daily_coverage.loc[
            daily_coverage["audit_status"].eq("ELIGIBLE"), "target_date"
        ].astype(str)
    )
    events = clob.events[
        clob.events["target_date"].astype(str).isin(eligible_dates)
    ].copy()
    event_ids = set(events["event_id"].astype(str))
    markets = clob.markets[
        clob.markets["event_id"].astype(str).isin(event_ids)
    ].copy()
    condition_ids = set(markets["condition_id"].astype(str))
    trades = clob.trades[
        clob.trades["condition_id"].astype(str).isin(condition_ids)
    ].copy()
    return events, markets, trades


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HistoricalDataIncomplete(f"invalid manifest {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
