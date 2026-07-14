"""Past-only synthetic executable quotes derived from public trades."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_REQUIRED_TRADE_COLUMNS = {
    "condition_id",
    "timestamp",
    "outcome",
    "side",
    "price",
    "size",
    "usdc",
}


@dataclass(frozen=True)
class SyntheticQuoteDataset:
    """Immutable synthetic quote evidence derived from one raw CLOB artifact."""

    quotes: pd.DataFrame
    start_at: str
    end_at: str
    decision_tick: str
    lookback: str
    max_staleness: str

    def save(
        self,
        root: Path | str,
        *,
        source_sha256: str,
        created_at: str | pd.Timestamp | None = None,
    ) -> dict:
        artifact_dir = Path(root)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        quotes_path = artifact_dir / "quotes.parquet"
        self.quotes.to_parquet(quotes_path, index=False)
        created = _utc(created_at or pd.Timestamp.now(tz="UTC"))
        manifest = {
            "schema_version": 1,
            "data_class": "past_only_synthetic_quotes",
            "requested_range": [self.start_at, self.end_at],
            "decision_tick": self.decision_tick,
            "lookback": self.lookback,
            "max_staleness": self.max_staleness,
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "row_count": len(self.quotes),
            "condition_count": int(self.quotes["condition_id"].nunique()),
            "source_sha256": source_sha256,
            "sha256": {
                "quotes.parquet": hashlib.sha256(
                    quotes_path.read_bytes()
                ).hexdigest()
            },
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, root: Path | str) -> "SyntheticQuoteDataset":
        artifact_dir = Path(root)
        try:
            manifest = json.loads(
                (artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"invalid synthetic quote manifest: {exc}"
            ) from exc
        if manifest.get("schema_version") != 1:
            raise RuntimeError(
                "unsupported synthetic quote schema_version="
                f"{manifest.get('schema_version')!r}"
            )
        quotes_path = artifact_dir / "quotes.parquet"
        expected = manifest.get("sha256", {}).get("quotes.parquet")
        try:
            actual = hashlib.sha256(quotes_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError("missing synthetic quote parquet") from exc
        if expected != actual:
            raise RuntimeError("synthetic quote checksum mismatch: quotes.parquet")
        start_at, end_at = manifest["requested_range"]
        return cls(
            quotes=pd.read_parquet(quotes_path),
            start_at=str(start_at),
            end_at=str(end_at),
            decision_tick=str(manifest["decision_tick"]),
            lookback=str(manifest["lookback"]),
            max_staleness=str(manifest["max_staleness"]),
        )


def build_synthetic_book(
    quote: Mapping[str, object] | pd.Series,
    *,
    outcome: str,
    levels: int = 5,
) -> dict[str, list[dict[str, float]]]:
    """Expand a synthetic top-of-book quote into a conservative L2 ladder."""
    if levels <= 0:
        raise ValueError("levels must be positive")
    outcome = outcome.upper()
    if outcome not in {"YES", "NO"}:
        raise ValueError("outcome must be YES or NO")
    capacity = max(0.0, float(quote["capacity_usdc"]))
    if capacity == 0:
        return {"bids": [], "asks": []}
    bid = float(quote["bid_price_yes"])
    ask = float(quote["ask_price_yes"])
    impact = max(0.0, float(quote["impact_per_usdc"]))
    usd_per_level = capacity / levels
    yes_bids: list[dict[str, float]] = []
    yes_asks: list[dict[str, float]] = []
    for index in range(levels):
        offset = impact * usd_per_level * index
        bid_price = float(np.clip(bid - offset, 0.001, 0.999))
        ask_price = float(np.clip(ask + offset, 0.001, 0.999))
        yes_bids.append(
            {"price": bid_price, "size": usd_per_level / bid_price}
        )
        yes_asks.append(
            {"price": ask_price, "size": usd_per_level / ask_price}
        )
    if outcome == "YES":
        return {"bids": yes_bids, "asks": yes_asks}
    return {
        "bids": sorted(
            [
                {
                    "price": round(1.0 - level["price"], 12),
                    "size": level["size"],
                }
                for level in yes_asks
            ],
            key=lambda level: level["price"],
            reverse=True,
        ),
        "asks": sorted(
            [
                {
                    "price": round(1.0 - level["price"], 12),
                    "size": level["size"],
                }
                for level in yes_bids
            ],
            key=lambda level: level["price"],
        ),
    }


def build_synthetic_quotes(
    trades: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    decision_tick: str = "5min",
    lookback: str = "60min",
    max_staleness: str = "30min",
    min_tick_size: float = 0.01,
    min_tick_sizes: Mapping[str, float] | None = None,
    progress_callback: Callable[[int, int, str, int], None] | None = None,
) -> pd.DataFrame:
    """Estimate 5-minute quotes using information observable by each tick."""
    missing = sorted(_REQUIRED_TRADE_COLUMNS - set(trades.columns))
    if missing:
        raise ValueError(
            "trades missing columns: " + ", ".join(missing)
        )
    start_at = _utc(start)
    end_at = _utc(end)
    if end_at < start_at:
        raise ValueError("end must not precede start")
    tick_delta = pd.Timedelta(decision_tick)
    lookback_delta = pd.Timedelta(lookback)
    stale_delta = pd.Timedelta(max_staleness)
    if tick_delta <= pd.Timedelta(0) or lookback_delta <= pd.Timedelta(0):
        raise ValueError("decision_tick and lookback must be positive")
    if min_tick_size <= 0:
        raise ValueError("min_tick_size must be positive")

    work = trades.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["outcome"] = work["outcome"].astype(str).str.upper()
    work["side"] = work["side"].astype(str).str.upper()
    if not work["outcome"].isin(["YES", "NO"]).all():
        raise ValueError("trade outcome must be YES or NO")
    if not work["side"].isin(["BUY", "SELL"]).all():
        raise ValueError("trade side must be BUY or SELL")
    work["yes_price"] = np.where(
        work["outcome"] == "YES",
        work["price"].astype(float),
        1.0 - work["price"].astype(float),
    )
    work["yes_side"] = np.where(
        ((work["outcome"] == "YES") & (work["side"] == "BUY"))
        | ((work["outcome"] == "NO") & (work["side"] == "SELL")),
        "BUY",
        "SELL",
    )
    work = work.sort_values(["condition_id", "timestamp"]).reset_index(drop=True)
    ticks = pd.date_range(start_at, end_at, freq=decision_tick)
    rows: list[dict] = []
    grouped_conditions = list(work.groupby(
        "condition_id",
        sort=True,
    ))
    for completed, (condition_id, condition_trades) in enumerate(
        grouped_conditions, start=1
    ):
        condition_trades = condition_trades.sort_values("timestamp")
        condition_tick_size = float(
            (min_tick_sizes or {}).get(str(condition_id), min_tick_size)
        )
        if condition_tick_size <= 0:
            raise ValueError(
                f"min tick size must be positive for {condition_id}"
            )
        first_tick = condition_trades["timestamp"].iloc[0].ceil(decision_tick)
        last_tick = (
            condition_trades["timestamp"].iloc[-1] + stale_delta
        ).floor(decision_tick)
        condition_ticks = ticks[(ticks >= first_tick) & (ticks <= last_tick)]
        trade_time_ns = (
            condition_trades["timestamp"]
            .astype("datetime64[ns, UTC]")
            .astype("int64")
            .to_numpy()
        )
        yes_prices = condition_trades["yes_price"].to_numpy(dtype=float)
        yes_sides = condition_trades["yes_side"].to_numpy(dtype=str)
        trade_usdc = condition_trades["usdc"].to_numpy(dtype=float)
        for tick in condition_ticks:
            right = int(
                np.searchsorted(trade_time_ns, tick.value, side="right")
            )
            if right == 0:
                continue
            last_index = right - 1
            last_trade_at = condition_trades["timestamp"].iloc[last_index]
            staleness = tick - last_trade_at
            if staleness > stale_delta:
                continue
            cutoff_ns = (tick - lookback_delta).value
            left = int(
                np.searchsorted(trade_time_ns, cutoff_ns, side="right")
            )
            if left >= right:
                continue
            recent_prices = yes_prices[left:right]
            recent_sides = yes_sides[left:right]
            recent_usdc = trade_usdc[left:right]
            buys = recent_prices[recent_sides == "BUY"]
            sells = recent_prices[recent_sides == "SELL"]
            price_changes = np.abs(np.diff(recent_prices))
            if len(buys) and len(sells):
                spread = max(
                    condition_tick_size,
                    float(np.median(buys) - np.median(sells)),
                )
                quality = "two_sided_trade_proxy"
            else:
                movement = (
                    float(np.median(price_changes))
                    if len(recent_prices) > 1
                    else 0.0
                )
                spread = max(2.0 * condition_tick_size, 2.0 * movement)
                quality = "one_sided_trade_proxy"
            spread = float(np.clip(spread, condition_tick_size, 0.50))
            mid = float(np.clip(yes_prices[last_index], 0.0, 1.0))
            bid = float(np.clip(mid - spread / 2.0, 0.0, 1.0))
            ask = float(np.clip(mid + spread / 2.0, 0.0, 1.0))
            spread = ask - bid
            rolling_usdc = float(recent_usdc.sum())
            impact_samples = np.divide(
                price_changes,
                recent_usdc[1:],
                out=np.full_like(price_changes, np.nan),
                where=recent_usdc[1:] != 0.0,
            )
            impact_samples = impact_samples[np.isfinite(impact_samples)]
            impact = (
                float(np.median(impact_samples))
                if len(impact_samples)
                else 0.0
            )
            impact_floor = spread / max(rolling_usdc, 1.0)
            impact = max(impact, impact_floor)
            capacity = rolling_usdc * (
                tick_delta.total_seconds() / lookback_delta.total_seconds()
            )
            rows.append(
                {
                    "condition_id": str(condition_id),
                    "timestamp": tick,
                    "mid_price_yes": mid,
                    "bid_price_yes": bid,
                    "ask_price_yes": ask,
                    "estimated_spread": spread,
                    "impact_per_usdc": impact,
                    "capacity_usdc": float(capacity),
                    "rolling_volume_usdc": rolling_usdc,
                    "trade_count": len(recent_prices),
                    "buy_yes_trade_count": len(buys),
                    "sell_yes_trade_count": len(sells),
                    "last_trade_at": last_trade_at,
                    "staleness_seconds": staleness.total_seconds(),
                    "quote_quality": quality,
                    "source": "past_public_trades",
                }
            )
        if progress_callback is not None:
            progress_callback(
                completed,
                len(grouped_conditions),
                str(condition_id),
                len(rows),
            )
    columns = [
        "condition_id",
        "timestamp",
        "mid_price_yes",
        "bid_price_yes",
        "ask_price_yes",
        "estimated_spread",
        "impact_per_usdc",
        "capacity_usdc",
        "rolling_volume_usdc",
        "trade_count",
        "buy_yes_trade_count",
        "sell_yes_trade_count",
        "last_trade_at",
        "staleness_seconds",
        "quote_quality",
        "source",
    ]
    return pd.DataFrame(rows, columns=columns)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
