"""Signal-only calibration evidence and conservative diagnostic replay."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.study import StudyValidation
from backtest.synthetic_quotes import build_synthetic_book
from config.stations import STATIONS
from core.pricing.fees import taker_fee_usdc
from core.pricing.quote import compute_quote


@dataclass(frozen=True)
class EdgeCalibrator:
    """Transparent residual calibration with conservative probability bounds."""

    uncertainty_cuts: tuple[float, float]
    global_correction: float
    global_lower: float
    global_upper: float
    cells: tuple[dict, ...]
    prior_strength: float
    bootstrap_samples: int
    seed: int

    def calibrate(self, observations: pd.DataFrame) -> pd.DataFrame:
        work = _assign_strata(observations, self.uncertainty_cuts)
        cell_by_key = {str(cell["cell"]): cell for cell in self.cells}
        corrections: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        counts: list[int] = []
        for key in work["calibration_cell"]:
            cell = cell_by_key.get(str(key))
            corrections.append(
                float(
                    cell["correction"]
                    if cell is not None
                    else self.global_correction
                )
            )
            lowers.append(
                float(cell["lower"] if cell is not None else self.global_lower)
            )
            uppers.append(
                float(cell["upper"] if cell is not None else self.global_upper)
            )
            counts.append(int(cell["event_count"]) if cell is not None else 0)
        probability = work["yes_probability"].astype(float).to_numpy()
        work["calibration_correction"] = corrections
        work["calibration_event_count"] = counts
        work["calibrated_yes_probability"] = np.clip(
            probability + np.asarray(corrections), 0.0, 1.0
        )
        work["yes_probability_lower"] = np.clip(
            probability + np.asarray(lowers), 0.0, 1.0
        )
        work["yes_probability_upper"] = np.clip(
            probability + np.asarray(uppers), 0.0, 1.0
        )
        return work

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "data_class": "edge_residual_calibration",
            "uncertainty_cuts": list(self.uncertainty_cuts),
            "global_correction": self.global_correction,
            "global_lower": self.global_lower,
            "global_upper": self.global_upper,
            "cells": list(self.cells),
            "prior_strength": self.prior_strength,
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EdgeCalibrator":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported edge calibration schema_version")
        return cls(
            uncertainty_cuts=tuple(
                float(value) for value in payload["uncertainty_cuts"]
            ),
            global_correction=float(payload["global_correction"]),
            global_lower=float(payload["global_lower"]),
            global_upper=float(payload["global_upper"]),
            cells=tuple(dict(cell) for cell in payload["cells"]),
            prior_strength=float(payload["prior_strength"]),
            bootstrap_samples=int(payload["bootstrap_samples"]),
            seed=int(payload["seed"]),
        )


@dataclass(frozen=True)
class CrossFitEdgeCalibration:
    observations: pd.DataFrame
    folds: pd.DataFrame
    frozen_calibrator: EdgeCalibrator


@dataclass(frozen=True)
class DiagnosticReplayResult:
    trades: pd.DataFrame
    settlements: pd.DataFrame
    decisions: pd.DataFrame


@dataclass(frozen=True)
class EdgeStudyReport:
    signal_summary: dict[str, float | int]
    money_summary: dict[str, float | int]
    by_stratum: pd.DataFrame
    paper_eligible: bool
    gate_reasons: tuple[str, ...]


def save_edge_study_artifact(
    root: Path | str,
    *,
    study_id: str,
    split_role: str,
    input_sha256: dict[str, str],
    observations: pd.DataFrame,
    calibrator: EdgeCalibrator,
    replay: DiagnosticReplayResult,
    report: EdgeStudyReport,
    folds: pd.DataFrame | None = None,
) -> dict:
    """Persist immutable, hash-verified edge-study evidence."""
    artifact_dir = Path(root)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "observations.parquet": observations,
        "trades.parquet": replay.trades,
        "settlements.parquet": replay.settlements,
        "decisions.parquet": replay.decisions,
        "by_stratum.parquet": report.by_stratum,
    }
    hashes: dict[str, str] = {}
    for filename, frame in frames.items():
        path = artifact_dir / filename
        frame.to_parquet(path, index=False)
        hashes[filename] = _file_sha256(path)
    calibration_path = artifact_dir / "edge_calibration.json"
    calibration_path.write_text(
        json.dumps(calibrator.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    hashes[calibration_path.name] = _file_sha256(calibration_path)
    summary = {
        "signal": report.signal_summary,
        "money": report.money_summary,
        "paper_eligible": report.paper_eligible,
        "gate_reasons": list(report.gate_reasons),
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    hashes[summary_path.name] = _file_sha256(summary_path)
    if folds is not None:
        folds_path = artifact_dir / "folds.json"
        fold_rows = [
            {
                "fold": int(row.fold),
                "fit_dates": list(row.fit_dates),
                "held_out_dates": list(row.held_out_dates),
            }
            for row in folds.itertuples(index=False)
        ]
        folds_path.write_text(
            json.dumps(fold_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        hashes[folds_path.name] = _file_sha256(folds_path)
    manifest = {
        "schema_version": 1,
        "data_class": "backtest_edge_study",
        "study_id": study_id,
        "split_role": split_role,
        "input_sha256": dict(sorted(input_sha256.items())),
        "calibration_sha256": hashes["edge_calibration.json"],
        "sha256": hashes,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_edge_calibrator(root: Path | str) -> EdgeCalibrator:
    artifact_dir = Path(root)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    path = artifact_dir / "edge_calibration.json"
    if _file_sha256(path) != manifest.get("calibration_sha256"):
        raise ValueError("edge calibration checksum mismatch")
    return EdgeCalibrator.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def build_edge_study_report(
    calibrated_observations: pd.DataFrame,
    replay: DiagnosticReplayResult,
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
    min_traded_events: int = 20,
    min_event_days: int = 10,
) -> EdgeStudyReport:
    """Report signal evidence separately from fixed-risk money outcomes."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    observations = calibrated_observations.copy()
    event_counts = observations.groupby("event_id")["event_id"].transform("size")
    weights = 1.0 / event_counts.astype(float)
    weights /= weights.sum()
    outcome = observations["yes_outcome"].astype(float)
    raw_probability = observations["yes_probability"].astype(float)
    calibrated_probability = observations[
        "calibrated_yes_probability"
    ].astype(float)
    raw_brier = float((((raw_probability - outcome) ** 2) * weights).sum())
    calibrated_brier = float(
        (((calibrated_probability - outcome) ** 2) * weights).sum()
    )
    calibration_residual = float(
        ((outcome - calibrated_probability) * weights).sum()
    )
    if "calibration_cell" in observations:
        stratum_rows: list[dict] = []
        for cell, rows in observations.groupby("calibration_cell", sort=True):
            counts = rows.groupby("event_id")["event_id"].transform("size")
            cell_weights = 1.0 / counts.astype(float)
            cell_weights /= cell_weights.sum()
            residual = (
                rows["yes_outcome"].astype(float)
                - rows["calibrated_yes_probability"].astype(float)
            )
            stratum_rows.append(
                {
                    "calibration_cell": str(cell),
                    "event_count": int(rows["event_id"].nunique()),
                    "row_count": len(rows),
                    "mean_calibration_residual": float(
                        (residual * cell_weights).sum()
                    ),
                }
            )
        by_stratum = pd.DataFrame(stratum_rows)
    else:
        by_stratum = pd.DataFrame(
            columns=[
                "calibration_cell",
                "event_count",
                "row_count",
                "mean_calibration_residual",
            ]
        )

    trades = replay.trades.copy()
    settlements = replay.settlements.copy()
    event_days = int(trades["target_date"].nunique()) if len(trades) else 0
    realized_lcb, pnl_lcb = _diagnostic_bootstrap_lcbs(
        trades,
        settlements,
        samples=bootstrap_samples,
        seed=seed,
    )
    net_pnl = float(
        settlements.get(
            "realized_pnl_usd", pd.Series(dtype=float)
        ).sum()
    )
    daily_pnl = (
        settlements.groupby("target_date", sort=True)["realized_pnl_usd"].sum()
        if not settlements.empty
        else pd.Series(dtype=float)
    )
    cumulative = daily_pnl.cumsum()
    max_drawdown = (
        float((cumulative.cummax().clip(lower=0.0) - cumulative).max())
        if len(cumulative)
        else float("nan")
    )
    signal_summary: dict[str, float | int] = {
        "observation_events": int(observations["event_id"].nunique()),
        "traded_events": int(trades["event_id"].nunique()) if len(trades) else 0,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "mean_calibration_residual": calibration_residual,
        "mean_realized_executable_edge": (
            float(trades["realized_executable_edge"].mean())
            if len(trades)
            else float("nan")
        ),
        "realized_edge_lcb_90": realized_lcb,
    }
    money_summary: dict[str, float | int] = {
        "event_days": event_days,
        "trade_count": len(trades),
        "turnover_usd": float(
            trades.get("size_usd", pd.Series(dtype=float)).sum()
        ),
        "fees_usd": float(
            trades.get("fee_usd", pd.Series(dtype=float)).sum()
        ),
        "net_pnl_usd": net_pnl,
        "mean_event_day_pnl_usd": (
            float(daily_pnl.mean()) if len(daily_pnl) else float("nan")
        ),
        "event_day_pnl_lcb_90": pnl_lcb,
        "max_drawdown_usd": max_drawdown,
    }
    reasons: list[str] = []
    if signal_summary["traded_events"] < min_traded_events:
        reasons.append("insufficient_traded_events")
    if event_days < min_event_days:
        reasons.append("insufficient_event_days")
    if not np.isfinite(realized_lcb) or realized_lcb <= 0.0:
        reasons.append("non_positive_realized_edge_lcb")
    if not np.isfinite(pnl_lcb) or pnl_lcb <= 0.0:
        reasons.append("non_positive_event_day_pnl_lcb")
    return EdgeStudyReport(
        signal_summary=signal_summary,
        money_summary=money_summary,
        by_stratum=by_stratum,
        paper_eligible=not reasons,
        gate_reasons=tuple(reasons),
    )


def run_diagnostic_replay(
    calibrated_observations: pd.DataFrame,
    *,
    min_edge: float = 0.04,
    risk_usd: float = 50.0,
) -> DiagnosticReplayResult:
    """Take only the first qualifying fixed-risk signal for each Event."""
    if risk_usd <= 0:
        raise ValueError("risk_usd must be positive")
    work = calibrated_observations.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    trade_rows: list[dict] = []
    settlement_rows: list[dict] = []
    decision_rows: list[dict] = []
    for event_id, event_rows in work.groupby("event_id", sort=True):
        entered = False
        for timestamp, tick_rows in event_rows.groupby("timestamp", sort=True):
            candidates: list[dict] = []
            for row in tick_rows.to_dict("records"):
                choices = [
                    {
                        "side": "YES",
                        "token_id": row["yes_token_id"],
                        "probability_bound": float(
                            row["yes_probability_lower"]
                        ),
                        "effective_price": float(row["yes_effective_price"]),
                        "vwap": float(row["yes_vwap"]),
                        "fee_usd": float(row["yes_fee_usd"]),
                        "outcome": float(row["yes_outcome"]),
                        "outcome_rank": 0,
                    },
                    {
                        "side": "NO",
                        "token_id": row["no_token_id"],
                        "probability_bound": 1.0
                        - float(row["yes_probability_upper"]),
                        "effective_price": float(row["no_effective_price"]),
                        "vwap": float(row["no_vwap"]),
                        "fee_usd": float(row["no_fee_usd"]),
                        "outcome": 1.0 - float(row["yes_outcome"]),
                        "outcome_rank": 1,
                    },
                ]
                for choice in choices:
                    edge = (
                        choice["probability_bound"]
                        - choice["effective_price"]
                    )
                    decision_rows.append(
                        {
                            "timestamp": timestamp,
                            "event_id": str(event_id),
                            "condition_id": str(row["condition_id"]),
                            "side": choice["side"],
                            "conservative_executable_edge": edge,
                            "qualified": edge >= min_edge,
                        }
                    )
                    if edge >= min_edge:
                        candidates.append(
                            {
                                **choice,
                                "edge": edge,
                                "condition_id": str(row["condition_id"]),
                                "target_date": str(row["target_date"]),
                            }
                        )
            if not candidates:
                continue
            chosen = sorted(
                candidates,
                key=lambda item: (
                    -item["edge"],
                    item["condition_id"],
                    item["outcome_rank"],
                ),
            )[0]
            shares = risk_usd / chosen["vwap"]
            payout = shares * chosen["outcome"]
            total_cost = risk_usd + chosen["fee_usd"]
            realized_pnl = payout - total_cost
            trade_rows.append(
                {
                    "timestamp": timestamp,
                    "event_id": str(event_id),
                    "target_date": chosen["target_date"],
                    "condition_id": chosen["condition_id"],
                    "token_id": chosen["token_id"],
                    "side": chosen["side"],
                    "size_usd": float(risk_usd),
                    "shares": shares,
                    "vwap_price": chosen["vwap"],
                    "fee_usd": chosen["fee_usd"],
                    "effective_price": chosen["effective_price"],
                    "probability_bound": chosen["probability_bound"],
                    "conservative_executable_edge": chosen["edge"],
                    "token_outcome": chosen["outcome"],
                    "realized_executable_edge": (
                        chosen["outcome"] - chosen["effective_price"]
                    ),
                }
            )
            settlement_rows.append(
                {
                    "event_id": str(event_id),
                    "target_date": chosen["target_date"],
                    "condition_id": chosen["condition_id"],
                    "token_id": chosen["token_id"],
                    "payout_usd": payout,
                    "cost_basis_usd": total_cost,
                    "realized_pnl_usd": realized_pnl,
                }
            )
            entered = True
            break
        if entered:
            continue
    return DiagnosticReplayResult(
        trades=pd.DataFrame(trade_rows, columns=_DIAGNOSTIC_TRADE_COLUMNS),
        settlements=pd.DataFrame(
            settlement_rows,
            columns=_DIAGNOSTIC_SETTLEMENT_COLUMNS,
        ),
        decisions=pd.DataFrame(
            decision_rows,
            columns=_DIAGNOSTIC_DECISION_COLUMNS,
        ),
    )


def _diagnostic_bootstrap_lcbs(
    trades: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if trades.empty or settlements.empty:
        return float("nan"), float("nan")
    days = sorted(trades["target_date"].astype(str).unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(samples, len(days)))
    draw_counts = np.zeros((samples, len(days)), dtype=np.int16)
    sample_rows = np.repeat(np.arange(samples), len(days))
    np.add.at(draw_counts, (sample_rows, draws.ravel()), 1)
    trade_days = trades["target_date"].astype(str)
    edge_sum = np.asarray(
        [
            trades.loc[
                trade_days == day,
                "realized_executable_edge",
            ].sum()
            for day in days
        ],
        dtype=float,
    )
    edge_count = np.asarray(
        [(trade_days == day).sum() for day in days],
        dtype=float,
    )
    pnl_by_day = (
        settlements.assign(
            target_date=settlements["target_date"].astype(str)
        )
        .groupby("target_date")["realized_pnl_usd"]
        .sum()
    )
    pnl = np.asarray([float(pnl_by_day.loc[day]) for day in days])
    edge_means = (draw_counts @ edge_sum) / (draw_counts @ edge_count)
    pnl_means = (draw_counts @ pnl) / len(days)
    return (
        float(np.quantile(edge_means, 0.1)),
        float(np.quantile(pnl_means, 0.1)),
    )


def cross_fit_edge_calibration(
    observations: pd.DataFrame,
    *,
    folds: int = 4,
    prior_strength: float = 20.0,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> CrossFitEdgeCalibration:
    """Calibrate each contiguous event-day block without fitting on it."""
    days = sorted(observations["target_date"].astype(str).unique())
    if folds < 2 or len(days) < 2:
        raise ValueError("cross-fitting requires at least two folds and days")
    fold_days = [
        tuple(str(day) for day in part)
        for part in np.array_split(days, min(folds, len(days)))
        if len(part)
    ]
    source = observations.reset_index(drop=True).copy()
    source["_source_index"] = source.index
    calibrated_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for fold_index, held_dates in enumerate(fold_days):
        held_mask = source["target_date"].astype(str).isin(held_dates)
        training = source.loc[~held_mask].drop(columns="_source_index")
        held = source.loc[held_mask].copy()
        calibrator = fit_edge_calibrator(
            training,
            prior_strength=prior_strength,
            bootstrap_samples=bootstrap_samples,
            seed=seed + fold_index,
        )
        calibrated = calibrator.calibrate(
            held.drop(columns="_source_index")
        )
        calibrated["_source_index"] = held["_source_index"].to_numpy()
        calibrated["calibration_fold"] = fold_index
        calibrated_parts.append(calibrated)
        fit_dates = tuple(
            day for day in days if day not in set(held_dates)
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "fit_dates": fit_dates,
                "held_out_dates": held_dates,
            }
        )
    calibrated_observations = (
        pd.concat(calibrated_parts, ignore_index=True)
        .sort_values("_source_index")
        .drop(columns="_source_index")
        .reset_index(drop=True)
    )
    frozen = fit_edge_calibrator(
        observations,
        prior_strength=prior_strength,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return CrossFitEdgeCalibration(
        observations=calibrated_observations,
        folds=pd.DataFrame(fold_rows),
        frozen_calibrator=frozen,
    )


def fit_edge_calibrator(
    observations: pd.DataFrame,
    *,
    prior_strength: float = 20.0,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> EdgeCalibrator:
    """Fit event-equal residual corrections and event-day bootstrap bounds."""
    required = {
        "event_id",
        "target_date",
        "yes_probability",
        "yes_outcome",
        "yes_effective_price",
        "lead_hours",
        "forecast_width_f",
    }
    missing = required - set(observations)
    if missing:
        raise ValueError(f"calibration observations missing: {sorted(missing)}")
    if observations.empty:
        raise ValueError("cannot calibrate empty observations")
    if prior_strength < 0 or bootstrap_samples <= 0:
        raise ValueError("calibration strengths and samples must be positive")
    widths = observations.groupby("event_id", sort=False)[
        "forecast_width_f"
    ].mean()
    cuts = tuple(float(value) for value in widths.quantile([1 / 3, 2 / 3]))
    work = _assign_strata(observations, cuts)
    global_correction, cells = _corrections(
        work,
        prior_strength=prior_strength,
        event_column="event_id",
    )
    global_samples, samples_by_cell = _bootstrap_corrections(
        work,
        cell_keys=tuple(cells),
        prior_strength=prior_strength,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    global_lower, global_upper = np.quantile(
        global_samples, [0.1, 0.9]
    )
    cell_rows: list[dict] = []
    for key, cell in sorted(cells.items()):
        lower, upper = np.quantile(samples_by_cell[key], [0.1, 0.9])
        cell_rows.append(
            {
                "cell": key,
                "correction": float(cell["correction"]),
                "lower": float(lower),
                "upper": float(upper),
                "event_count": int(cell["event_count"]),
            }
        )
    return EdgeCalibrator(
        uncertainty_cuts=cuts,
        global_correction=float(global_correction),
        global_lower=float(global_lower),
        global_upper=float(global_upper),
        cells=tuple(cell_rows),
        prior_strength=float(prior_strength),
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )


def build_edge_observations(
    validation: StudyValidation,
    *,
    risk_usd: float = 50.0,
) -> pd.DataFrame:
    """Bind each observable quote tick to forecast, truth, and execution."""
    if risk_usd <= 0:
        raise ValueError("risk_usd must be positive")
    events = validation.clob.events.copy()
    events["target_date"] = pd.to_datetime(
        events["target_date"]
    ).dt.date.astype(str)
    events = events[
        events["target_date"].between(
            validation.start_target,
            validation.end_target,
        )
    ].copy()
    station = STATIONS[validation.station_id]
    events["settlement_at"] = events["target_date"].map(
        lambda target_date: (
            pd.Timestamp(str(target_date))
            .tz_localize(station.timezone)
            + pd.DateOffset(days=1)
        ).tz_convert("UTC")
    )
    markets = validation.clob.markets.merge(
        events[
            ["event_id", "target_date", "event_end_at", "settlement_at"]
        ],
        on="event_id",
        how="inner",
        validate="many_to_one",
    )
    markets["market_close_at"] = _market_close(markets)
    actuals = dict(
        zip(
            validation.actuals["target_date"].astype(str),
            validation.actuals["actual_high_f"].astype(float),
        )
    )
    rows: list[dict] = []
    quotes = validation.quotes.quotes.copy()
    quotes = quotes[
        quotes["condition_id"].astype(str).isin(
            markets["condition_id"].astype(str)
        )
    ].copy()
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True)
    evidence = _join_visible_forecasts(
        quotes=quotes,
        markets=markets,
        forecast=validation.forecast,
        station_id=validation.station_id,
    )
    for row in evidence.itertuples(index=False):
            target_date = str(row.target_date)
            settlement_at = _utc(row.settlement_at)
            tick = _utc(row.timestamp)
            quote_payload = row._asdict()
            yes_book = build_synthetic_book(quote_payload, outcome="YES")
            no_book = build_synthetic_book(quote_payload, outcome="NO")
            yes_quote = compute_quote(yes_book, "BUY", risk_usd)
            no_quote = compute_quote(no_book, "BUY", risk_usd)
            if not yes_quote.complete_fill or not no_quote.complete_fill:
                continue
            fee_args = {
                "fees_enabled": bool(getattr(row, "fees_enabled", False)),
                "fee_rate": getattr(row, "fee_rate", None),
                "fee_exponent": getattr(row, "fee_exponent", 1.0),
            }
            yes_fee = taker_fee_usdc(
                shares=yes_quote.filled_shares,
                price=yes_quote.vwap_price,
                **fee_args,
            )
            no_fee = taker_fee_usdc(
                shares=no_quote.filled_shares,
                price=no_quote.vwap_price,
                **fee_args,
            )
            actual = actuals[target_date]
            yes_outcome = float(
                actual >= float(row.bucket_lo_f)
                and actual < float(row.bucket_hi_f)
            )
            rows.append(
                {
                    "timestamp": tick,
                    "event_id": str(row.event_id),
                    "target_date": target_date,
                    "condition_id": str(row.condition_id),
                    "yes_token_id": str(row.yes_token_id),
                    "no_token_id": str(row.no_token_id),
                    "yes_probability": float(row.yes_probability),
                    "yes_outcome": yes_outcome,
                    "p10_f": float(row.p10_f),
                    "p50_f": float(row.p50_f),
                    "p90_f": float(row.p90_f),
                    "forecast_width_f": float(
                        row.p90_f - row.p10_f
                    ),
                    "lead_hours": (
                        settlement_at - tick
                    ).total_seconds()
                    / 3600.0,
                    "estimated_spread": float(row.estimated_spread),
                    "capacity_usdc": float(row.capacity_usdc),
                    "yes_vwap": yes_quote.vwap_price,
                    "yes_fee_usd": yes_fee,
                    "yes_effective_price": (
                        risk_usd + yes_fee
                    )
                    / yes_quote.filled_shares,
                    "no_vwap": no_quote.vwap_price,
                    "no_fee_usd": no_fee,
                    "no_effective_price": (
                        risk_usd + no_fee
                    )
                    / no_quote.filled_shares,
                    "risk_usd": float(risk_usd),
                }
            )
    return pd.DataFrame(rows, columns=_OBSERVATION_COLUMNS)


def _join_visible_forecasts(
    *,
    quotes: pd.DataFrame,
    markets: pd.DataFrame,
    forecast: ForecastCache,
    station_id: str,
) -> pd.DataFrame:
    """Join each quote to the latest condition forecast visible at that tick."""
    condition_ids = set(markets["condition_id"].astype(str))
    snapshots = forecast.snapshots[
        forecast.snapshots["station_id"].astype(str) == station_id
    ].copy()
    probabilities = forecast.probabilities[
        forecast.probabilities["condition_id"].astype(str).isin(condition_ids)
    ].copy()
    visible = snapshots.merge(
        probabilities,
        on="snapshot_id",
        how="inner",
        validate="one_to_many",
    )
    visible["condition_id"] = visible["condition_id"].astype(str)
    visible["available_at"] = pd.to_datetime(
        visible["available_at"], utc=True
    )
    visible["init_time"] = pd.to_datetime(visible["init_time"], utc=True)
    quote_rows = quotes.copy()
    quote_rows["condition_id"] = quote_rows["condition_id"].astype(str)
    quote_rows = quote_rows.sort_values(
        ["timestamp", "condition_id"]
    ).reset_index(drop=True)
    visible = visible.sort_values(
        ["available_at", "init_time", "condition_id"]
    ).reset_index(drop=True)
    evidence = pd.merge_asof(
        quote_rows,
        visible,
        left_on="timestamp",
        right_on="available_at",
        by="condition_id",
        direction="backward",
        allow_exact_matches=True,
    ).dropna(subset=["snapshot_id"])
    market_columns = [
        "condition_id",
        "event_id",
        "target_date",
        "market_close_at",
        "settlement_at",
        "bucket_lo_f",
        "bucket_hi_f",
        "yes_token_id",
        "no_token_id",
    ]
    market_columns.extend(
        column
        for column in ("fees_enabled", "fee_rate", "fee_exponent")
        if column in markets
    )
    market_rows = markets[market_columns].rename(
        columns={"target_date": "market_target_date"}
    )
    evidence = evidence.merge(
        market_rows,
        on="condition_id",
        how="inner",
        validate="many_to_one",
    )
    evidence = evidence[
        evidence["target_date"].astype(str)
        == evidence["market_target_date"].astype(str)
    ]
    evidence = evidence[
        evidence["timestamp"] < evidence["market_close_at"]
    ]
    evidence = evidence[
        evidence["timestamp"] < evidence["settlement_at"]
    ]
    return evidence.sort_values(
        ["timestamp", "condition_id"]
    ).reset_index(drop=True)


def _market_close(markets: pd.DataFrame) -> pd.Series:
    values = pd.Series(pd.NaT, index=markets.index, dtype="datetime64[ns, UTC]")
    for column in ("closed_at", "market_end_at", "event_end_at"):
        if column in markets:
            values = values.fillna(
                pd.to_datetime(markets[column], utc=True, errors="coerce")
            )
    if values.isna().any():
        raise ValueError("every study Market requires a close timestamp")
    return values


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _assign_strata(
    observations: pd.DataFrame,
    uncertainty_cuts: tuple[float, float],
) -> pd.DataFrame:
    work = observations.copy()
    work["price_band"] = np.digitize(
        work["yes_effective_price"].astype(float),
        [0.2, 0.4, 0.6, 0.8],
        right=False,
    )
    work["lead_band"] = np.digitize(
        work["lead_hours"].astype(float),
        [6.0, 12.0, 24.0],
        right=False,
    )
    work["uncertainty_band"] = np.digitize(
        work["forecast_width_f"].astype(float),
        list(uncertainty_cuts),
        right=False,
    )
    work["calibration_cell"] = (
        work["price_band"].astype(str)
        + "|"
        + work["lead_band"].astype(str)
        + "|"
        + work["uncertainty_band"].astype(str)
    )
    return work


def _corrections(
    work: pd.DataFrame,
    *,
    prior_strength: float,
    event_column: str,
) -> tuple[float, dict[str, dict]]:
    weighted = work.copy()
    counts = weighted.groupby(event_column)[event_column].transform("size")
    weighted["_event_weight"] = 1.0 / counts.astype(float)
    weighted["_residual"] = (
        weighted["yes_outcome"].astype(float)
        - weighted["yes_probability"].astype(float)
    )
    total_weight = float(weighted["_event_weight"].sum())
    global_correction = float(
        (weighted["_residual"] * weighted["_event_weight"]).sum()
        / total_weight
    )
    cells: dict[str, dict] = {}
    for key, rows in weighted.groupby("calibration_cell", sort=True):
        weight = float(rows["_event_weight"].sum())
        cell_mean = float(
            (rows["_residual"] * rows["_event_weight"]).sum() / weight
        )
        event_count = int(rows[event_column].nunique())
        correction = (
            event_count * cell_mean + prior_strength * global_correction
        ) / (event_count + prior_strength)
        cells[str(key)] = {
            "correction": float(correction),
            "event_count": event_count,
        }
    return global_correction, cells


def _bootstrap_corrections(
    work: pd.DataFrame,
    *,
    cell_keys: tuple[str, ...],
    prior_strength: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Bootstrap event days from pre-aggregated sufficient statistics."""
    weighted = work.copy()
    weighted["_day"] = weighted["target_date"].astype(str)
    event_counts = weighted.groupby("event_id")["event_id"].transform("size")
    weighted["_event_weight"] = 1.0 / event_counts.astype(float)
    weighted["_weighted_residual"] = (
        weighted["yes_outcome"].astype(float)
        - weighted["yes_probability"].astype(float)
    ) * weighted["_event_weight"]
    days = sorted(weighted["_day"].unique())
    day_index = {day: index for index, day in enumerate(days)}
    cell_index = {cell: index for index, cell in enumerate(cell_keys)}
    global_numerator = np.zeros(len(days), dtype=float)
    global_denominator = np.zeros(len(days), dtype=float)
    cell_numerator = np.zeros((len(days), len(cell_keys)), dtype=float)
    cell_denominator = np.zeros((len(days), len(cell_keys)), dtype=float)
    cell_events = np.zeros((len(days), len(cell_keys)), dtype=float)

    for day, rows in weighted.groupby("_day", sort=False):
        day_offset = day_index[str(day)]
        global_numerator[day_offset] = rows["_weighted_residual"].sum()
        global_denominator[day_offset] = rows["_event_weight"].sum()
        for cell, cell_rows in rows.groupby("calibration_cell", sort=False):
            key = str(cell)
            if key not in cell_index:
                continue
            cell_offset = cell_index[key]
            cell_numerator[day_offset, cell_offset] = cell_rows[
                "_weighted_residual"
            ].sum()
            cell_denominator[day_offset, cell_offset] = cell_rows[
                "_event_weight"
            ].sum()
            cell_events[day_offset, cell_offset] = cell_rows[
                "event_id"
            ].nunique()

    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(days),
        size=(bootstrap_samples, len(days)),
    )
    draw_counts = np.zeros(
        (bootstrap_samples, len(days)),
        dtype=np.int16,
    )
    sample_rows = np.repeat(np.arange(bootstrap_samples), len(days))
    np.add.at(draw_counts, (sample_rows, draws.ravel()), 1)
    global_samples = (
        draw_counts @ global_numerator
        / (draw_counts @ global_denominator)
    )
    sampled_cell_numerator = draw_counts @ cell_numerator
    sampled_cell_denominator = draw_counts @ cell_denominator
    sampled_cell_events = draw_counts @ cell_events
    cell_means = np.divide(
        sampled_cell_numerator,
        sampled_cell_denominator,
        out=np.zeros_like(sampled_cell_numerator),
        where=sampled_cell_denominator > 0,
    )
    corrections = (
        sampled_cell_events * cell_means
        + prior_strength * global_samples[:, None]
    ) / (sampled_cell_events + prior_strength)
    corrections = np.where(
        sampled_cell_denominator > 0,
        corrections,
        global_samples[:, None],
    )
    return global_samples, {
        cell: corrections[:, offset]
        for cell, offset in cell_index.items()
    }


_OBSERVATION_COLUMNS = [
    "timestamp",
    "event_id",
    "target_date",
    "condition_id",
    "yes_token_id",
    "no_token_id",
    "yes_probability",
    "yes_outcome",
    "p10_f",
    "p50_f",
    "p90_f",
    "forecast_width_f",
    "lead_hours",
    "estimated_spread",
    "capacity_usdc",
    "yes_vwap",
    "yes_fee_usd",
    "yes_effective_price",
    "no_vwap",
    "no_fee_usd",
    "no_effective_price",
    "risk_usd",
]

_DIAGNOSTIC_TRADE_COLUMNS = [
    "timestamp",
    "event_id",
    "target_date",
    "condition_id",
    "token_id",
    "side",
    "size_usd",
    "shares",
    "vwap_price",
    "fee_usd",
    "effective_price",
    "probability_bound",
    "conservative_executable_edge",
    "token_outcome",
    "realized_executable_edge",
]
_DIAGNOSTIC_SETTLEMENT_COLUMNS = [
    "event_id",
    "target_date",
    "condition_id",
    "token_id",
    "payout_usd",
    "cost_basis_usd",
    "realized_pnl_usd",
]
_DIAGNOSTIC_DECISION_COLUMNS = [
    "timestamp",
    "event_id",
    "condition_id",
    "side",
    "conservative_executable_edge",
    "qualified",
]
