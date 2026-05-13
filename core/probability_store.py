"""In-process shared probability store for ForecastLoop and TradingLoop."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BucketPayload:
    """Probability and quantile data for a single temperature bucket.

    Frozen so field reassignment is prevented.  Callers must not mutate
    the nested ``quantiles_by_fxx`` dict; ``ProbabilityStore.write()`` and
    ``read()`` deepcopy payloads on ingress/egress to enforce isolation.

    Fields
    ------
    mc_prob:
        Monte-Carlo bucket probability for the YES outcome (0..1).
    quantiles_by_fxx:
        Dict mapping decoder hour offset (fxx) to a dict of quantile values:
        ``{"p10": float, "p50": float, "p90": float}``.
    last_tick_utc:
        UTC timestamp of the ForecastLoop run that produced this payload.
        Must be identical for every bucket in the same write() call.
    """

    mc_prob:          float
    quantiles_by_fxx: dict[int, dict[str, float]]
    last_tick_utc:    datetime


# Bucket tuple: (lo_f, hi_f)
_BucketKey = tuple[float, float]
_BucketMap = dict[_BucketKey, BucketPayload]

_VALID_DAY_OFFSETS = frozenset({0, 1, 2})


def _validate_last_tick_utc_consistent(buckets: _BucketMap) -> None:
    """Raise ValueError if buckets carry more than one distinct last_tick_utc."""
    if len(buckets) <= 1:
        return
    ticks = {p.last_tick_utc for p in buckets.values()}
    if len(ticks) > 1:
        raise ValueError(
            "All BucketPayload entries in a single write() call must share the "
            f"same last_tick_utc, but found {len(ticks)} distinct values: {ticks}"
        )


class ProbabilityStore:
    """Shared in-process store keyed by (condition_id, day_offset).

    Written by ForecastLoop, read by TradingLoop.
    An asyncio.Lock guards all mutations.

    day_offset domain: {0, 1, 2} -- up to 3 simultaneously active events.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._data: dict[tuple[str, int], _BucketMap] = {}

    @staticmethod
    def _validate_day_offset(day_offset: int) -> None:
        if day_offset not in _VALID_DAY_OFFSETS:
            raise ValueError(
                f"day_offset must be one of {sorted(_VALID_DAY_OFFSETS)}, "
                f"got {day_offset!r}"
            )

    async def write(
        self,
        condition_id: str,
        day_offset: int,
        buckets: _BucketMap,
    ) -> None:
        """Replace all bucket data for (condition_id, day_offset).

        Parameters
        ----------
        condition_id:
            Polymarket condition/event ID.
        day_offset:
            Day offset in {0, 1, 2}.
        buckets:
            Mapping of (lo_f, hi_f) -> BucketPayload for this key.

        Raises
        ------
        ValueError
            If day_offset is not in {0, 1, 2}.
        ValueError
            If bucket payloads carry inconsistent last_tick_utc values.
        """
        self._validate_day_offset(day_offset)
        _validate_last_tick_utc_consistent(buckets)
        snapshot = {k: copy.deepcopy(v) for k, v in buckets.items()}
        async with self._lock:
            self._data[(condition_id, day_offset)] = snapshot

    async def read(
        self,
        condition_id: str,
        day_offset: int,
    ) -> _BucketMap | None:
        """Return the bucket map for (condition_id, day_offset), or None.

        Parameters
        ----------
        condition_id:
            Polymarket condition/event ID.
        day_offset:
            Day offset in {0, 1, 2}.

        Returns
        -------
        A copy of the stored bucket map, or None if the key has not been
        written yet.

        Raises
        ------
        ValueError
            If day_offset is not in {0, 1, 2}.
        """
        self._validate_day_offset(day_offset)
        async with self._lock:
            entry = self._data.get((condition_id, day_offset))
            return {k: copy.deepcopy(v) for k, v in entry.items()} if entry is not None else None
