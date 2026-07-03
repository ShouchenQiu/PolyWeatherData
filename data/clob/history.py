"""Point-in-time historical Polymarket market-data ingestion."""
from __future__ import annotations

import calendar
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from events.discovery import parse_slug


class JsonTransport(Protocol):
    """Minimal HTTP seam used by the historical source."""

    def get_json(self, url: str, params: dict) -> object: ...


class RequestsJsonTransport:
    """Requests-backed JSON transport for public Polymarket APIs."""

    def get_json(self, url: str, params: dict) -> object:
        import requests

        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
            time.sleep(2**attempt)
        raise RuntimeError("unreachable historical API retry state")


class HistoricalDataIncomplete(RuntimeError):
    """Raised when an API boundary would silently truncate historical data."""


@dataclass(frozen=True)
class HistoricalClobDataset:
    """Canonical raw historical events, markets, and public trades."""

    events: pd.DataFrame
    markets: pd.DataFrame
    trades: pd.DataFrame
    start_date: str
    end_date: str
    station_id: str

    @classmethod
    def load(cls, root: Path | str) -> "HistoricalClobDataset":
        """Load an artifact only after all manifest hashes verify."""
        artifact_dir = Path(root)
        manifest_path = artifact_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HistoricalDataIncomplete(
                f"invalid historical manifest at {manifest_path}: {exc}"
            ) from exc
        if manifest.get("schema_version") != 1:
            raise HistoricalDataIncomplete(
                "unsupported historical artifact schema_version="
                f"{manifest.get('schema_version')!r}"
            )
        for filename, expected in manifest.get("sha256", {}).items():
            path = artifact_dir / filename
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise HistoricalDataIncomplete(
                    f"missing historical artifact file {filename}"
                ) from exc
            if actual != expected:
                raise HistoricalDataIncomplete(
                    f"historical artifact checksum mismatch: {filename}"
                )
        start_date, end_date = manifest["requested_range"]
        return cls(
            events=pd.read_parquet(artifact_dir / "events.parquet"),
            markets=pd.read_parquet(artifact_dir / "markets.parquet"),
            trades=pd.read_parquet(artifact_dir / "trades.parquet"),
            start_date=str(start_date),
            end_date=str(end_date),
            station_id=str(manifest["station_id"]),
        )

    def coverage(self) -> pd.DataFrame:
        """Return one completeness row for every requested target date."""
        rows: list[dict] = []
        for stamp in pd.date_range(self.start_date, self.end_date, freq="D"):
            target_date = stamp.date().isoformat()
            day_events = self.events[
                self.events["target_date"] == target_date
            ]
            event_ids = set(day_events["event_id"])
            day_markets = self.markets[
                self.markets["event_id"].isin(event_ids)
            ]
            condition_ids = set(day_markets["condition_id"])
            day_trades = self.trades[
                self.trades["condition_id"].isin(condition_ids)
            ]
            token_count = int(
                day_markets["yes_token_id"].astype(bool).sum()
                + day_markets["no_token_id"].astype(bool).sum()
            )
            event_count = len(day_events)
            market_count = len(day_markets)
            trade_count = len(day_trades)
            rows.append(
                {
                    "target_date": target_date,
                    "event_count": event_count,
                    "market_count": market_count,
                    "token_count": token_count,
                    "trade_count": trade_count,
                    "complete": bool(
                        event_count > 0
                        and market_count > 0
                        and token_count == 2 * market_count
                        and trade_count > 0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def save(
        self,
        root: Path | str,
        *,
        fetched_at: str | None = None,
    ) -> dict:
        """Persist an immutable raw-data artifact and its integrity manifest."""
        artifact_dir = Path(root)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        frames = {
            "events.parquet": self.events,
            "markets.parquet": self.markets,
            "trades.parquet": self.trades,
            "coverage.parquet": self.coverage(),
        }
        hashes: dict[str, str] = {}
        for filename, frame in frames.items():
            path = artifact_dir / filename
            frame.to_parquet(path, index=False)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

        requested_dates = {
            stamp.date().isoformat()
            for stamp in pd.date_range(self.start_date, self.end_date, freq="D")
        }
        observed_dates = set(self.events.get("target_date", pd.Series(dtype=str)))
        coverage = self.coverage()
        fetched = pd.Timestamp(fetched_at or pd.Timestamp.now(tz="UTC"))
        if fetched.tzinfo is None:
            fetched = fetched.tz_localize("UTC")
        else:
            fetched = fetched.tz_convert("UTC")
        manifest = {
            "schema_version": 1,
            "data_class": "raw_point_in_time",
            "provider": "polymarket_public_api",
            "station_id": self.station_id,
            "requested_range": [self.start_date, self.end_date],
            "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
            "row_counts": {
                "events": len(self.events),
                "markets": len(self.markets),
                "trades": len(self.trades),
            },
            "missing_target_dates": sorted(requested_dates - observed_dates),
            "incomplete_target_dates": coverage.loc[
                ~coverage["complete"], "target_date"
            ].tolist(),
            "sha256": hashes,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest


class HistoricalClobSource:
    """Download canonical NYC daily-high events and their public trades."""

    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        gamma_api: str = "https://gamma-api.polymarket.com",
        data_api: str = "https://data-api.polymarket.com",
        trade_page_size: int = 10_000,
        max_trade_offset: int = 10_000,
        cache_dir: Path | str | None = None,
        progress_callback: Callable[[int, int, str, int, int], None] | None = None,
    ) -> None:
        if not 1 <= trade_page_size <= 10_000:
            raise ValueError("trade_page_size must be between 1 and 10,000")
        if max_trade_offset < 0:
            raise ValueError("max_trade_offset must be non-negative")
        self._transport = transport or RequestsJsonTransport()
        self._gamma_api = gamma_api.rstrip("/")
        self._data_api = data_api.rstrip("/")
        self._trade_page_size = trade_page_size
        self._max_trade_offset = max_trade_offset
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._progress_callback = progress_callback

    def fetch(
        self,
        *,
        start_date: str,
        end_date: str,
        station: object,
    ) -> HistoricalClobDataset:
        """Fetch raw point-in-time records for an inclusive target-date range."""
        days = pd.date_range(start_date, end_date, freq="D")
        event_rows: list[dict] = []
        market_rows: list[dict] = []
        trade_rows: list[dict] = []

        for completed, day in enumerate(days, start=1):
            slug = (
                "highest-temperature-in-nyc-on-"
                f"{calendar.month_name[day.month].lower()}-{day.day}-{day.year}"
            )
            events = self._fetch_events(slug)
            for event in events:
                event_id = str(event.get("id") or "")
                event_rows.append(
                    {
                        "event_id": event_id,
                        "event_slug": str(event.get("slug") or ""),
                        "event_title": str(
                            event.get("title") or event.get("question") or ""
                        ),
                        "target_date": day.date().isoformat(),
                        "event_end_at": pd.to_datetime(
                            event.get("endDate"), utc=True, errors="coerce"
                        ),
                        "station_id": str(getattr(station, "station_id")),
                    }
                )
                for market in event.get("markets", []):
                    normalized = self._normalize_market(event_id, market)
                    if normalized is None:
                        continue
                    market_rows.append(normalized)
                    trade_rows.extend(
                        self._fetch_trades(normalized["condition_id"])
                    )
            if self._progress_callback is not None:
                self._progress_callback(
                    completed,
                    len(days),
                    day.date().isoformat(),
                    len(market_rows),
                    len(trade_rows),
                )

        events_frame = pd.DataFrame(
            event_rows,
            columns=[
                "event_id",
                "event_slug",
                "event_title",
                "target_date",
                "event_end_at",
                "station_id",
            ],
        ).drop_duplicates(subset=["event_id"])
        markets_frame = pd.DataFrame(
            market_rows,
            columns=[
                "event_id",
                "market_id",
                "condition_id",
                "market_slug",
                "yes_token_id",
                "no_token_id",
                "bucket_lo_f",
                "bucket_hi_f",
                "fees_enabled",
                "fee_type",
                "fee_rate",
                "fee_exponent",
                "fee_taker_only",
                "fee_rebate_rate",
                "maker_base_fee",
                "taker_base_fee",
                "min_tick_size",
                "min_order_size",
                "accepting_orders",
                "active",
                "closed",
                "neg_risk",
                "market_start_at",
                "market_end_at",
                "accepting_orders_at",
                "closed_at",
            ],
        ).drop_duplicates(subset=["condition_id"])
        trades_frame = self._trade_frame(trade_rows)
        return HistoricalClobDataset(
            events=events_frame.reset_index(drop=True),
            markets=markets_frame.reset_index(drop=True),
            trades=trades_frame,
            start_date=str(start_date),
            end_date=str(end_date),
            station_id=str(getattr(station, "station_id")),
        )

    @staticmethod
    def _normalize_market(event_id: str, market: dict) -> dict | None:
        token_ids = _json_list(market.get("clobTokenIds"))
        outcomes = _json_list(market.get("outcomes"))
        if len(token_ids) != len(outcomes):
            return None
        tokens = {
            str(outcome).upper(): str(token)
            for outcome, token in zip(outcomes, token_ids)
        }
        if not tokens.get("YES") or not tokens.get("NO"):
            return None
        slug = str(market.get("slug") or "")
        bucket = parse_slug(slug)
        if bucket is None:
            return None
        lo_f, hi_f = bucket
        fee_schedule = _json_object(market.get("feeSchedule"))
        return {
            "event_id": event_id,
            "market_id": str(market.get("id") or ""),
            "condition_id": str(market.get("conditionId") or ""),
            "market_slug": slug,
            "yes_token_id": tokens["YES"],
            "no_token_id": tokens["NO"],
            "bucket_lo_f": float(lo_f),
            "bucket_hi_f": float(hi_f),
            "fees_enabled": bool(market.get("feesEnabled", False)),
            "fee_type": str(market.get("feeType") or ""),
            "fee_rate": _optional_float(fee_schedule.get("rate")),
            "fee_exponent": _optional_float(fee_schedule.get("exponent")),
            "fee_taker_only": bool(fee_schedule.get("takerOnly", False)),
            "fee_rebate_rate": _optional_float(
                fee_schedule.get("rebateRate")
            ),
            "maker_base_fee": _optional_float(market.get("makerBaseFee")),
            "taker_base_fee": _optional_float(market.get("takerBaseFee")),
            "min_tick_size": _optional_float(
                market.get("orderPriceMinTickSize")
            ),
            "min_order_size": _optional_float(market.get("orderMinSize")),
            "accepting_orders": bool(market.get("acceptingOrders", False)),
            "active": bool(market.get("active", False)),
            "closed": bool(market.get("closed", False)),
            "neg_risk": bool(market.get("negRisk", False)),
            "market_start_at": _utc_timestamp(market.get("startDate")),
            "market_end_at": _utc_timestamp(market.get("endDate")),
            "accepting_orders_at": _utc_timestamp(
                market.get("acceptingOrdersTimestamp")
            ),
            "closed_at": _utc_timestamp(market.get("closedTime")),
        }

    def _fetch_trades(self, condition_id: str) -> list[dict]:
        cached = self._read_cache("trades", condition_id)
        if cached is not None:
            return list(cached) if isinstance(cached, list) else []
        rows: list[dict] = []
        offset = 0
        while offset <= self._max_trade_offset:
            payload = self._transport.get_json(
                f"{self._data_api}/trades",
                {
                    "market": condition_id,
                    "limit": self._trade_page_size,
                    "offset": offset,
                    "takerOnly": "true",
                },
            )
            page = list(payload) if isinstance(payload, list) else []
            rows.extend(page)
            if len(page) < self._trade_page_size:
                self._write_cache("trades", condition_id, rows)
                return rows
            offset += self._trade_page_size
        raise HistoricalDataIncomplete(
            "trade history reached the configured API offset boundary for "
            f"condition_id={condition_id}; refusing to mark it complete"
        )

    def _fetch_events(self, slug: str) -> list[dict]:
        cached = self._read_cache("events", slug)
        if cached is None:
            payload = self._transport.get_json(
                f"{self._gamma_api}/events",
                {"slug": slug},
            )
            events = list(payload) if isinstance(payload, list) else []
            self._write_cache("events", slug, events)
        else:
            events = list(cached) if isinstance(cached, list) else []
        if events:
            return events

        recovered = self._read_cache("events-recovered", slug)
        if recovered is not None:
            return list(recovered) if isinstance(recovered, list) else []
        payload = self._transport.get_json(
            f"{self._gamma_api}/public-search",
            {
                "q": slug.replace("-", " "),
                "events_status": "closed",
                "keep_closed_markets": 1,
                "limit_per_type": 10,
            },
        )
        candidates = payload.get("events", []) if isinstance(payload, dict) else []
        matched = [
            event
            for event in candidates or []
            if isinstance(event, dict) and str(event.get("slug") or "") == slug
        ]
        self._write_cache("events-recovered", slug, matched)
        return matched

    def _cache_path(self, category: str, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._cache_dir / category / f"{safe_key}.json"

    def _read_cache(self, category: str, key: str) -> object | None:
        path = self._cache_path(category, key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HistoricalDataIncomplete(
                f"invalid raw API cache file {path}: {exc}"
            ) from exc

    def _write_cache(self, category: str, key: str, payload: object) -> None:
        path = self._cache_path(category, key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _trade_frame(rows: list[dict]) -> pd.DataFrame:
        columns = [
            "condition_id",
            "token_id",
            "timestamp",
            "outcome",
            "side",
            "price",
            "size",
            "usdc",
            "transaction_hash",
            "proxy_wallet",
            "source",
        ]
        canonical = []
        for row in rows:
            price = float(row.get("price") or 0.0)
            size = float(row.get("size") or 0.0)
            canonical.append(
                {
                    "condition_id": str(row.get("conditionId") or ""),
                    "token_id": str(row.get("asset") or ""),
                    "timestamp": pd.to_datetime(
                        row.get("timestamp"), unit="s", utc=True
                    ),
                    "outcome": str(row.get("outcome") or "").upper(),
                    "side": str(row.get("side") or "").upper(),
                    "price": price,
                    "size": size,
                    "usdc": price * size,
                    "transaction_hash": str(
                        row.get("transactionHash") or ""
                    ),
                    "proxy_wallet": str(row.get("proxyWallet") or ""),
                    "source": "data_api_trade",
                }
            )
        if not canonical:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(canonical, columns=columns)
            .drop_duplicates()
            .sort_values(
                ["timestamp", "condition_id", "token_id", "transaction_hash"]
            )
            .reset_index(drop=True)
        )


def _json_list(value: object) -> list:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return list(value) if isinstance(value, (list, tuple)) else []


def _json_object(value: object) -> dict:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")
