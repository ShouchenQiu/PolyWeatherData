"""In-process shared probability store for ForecastLoop and TradingLoop."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class BucketPayload:
    """Probability and quantile data for a single temperature bucket.

    Fields
    ------
    mc_prob:
        Monte-Carlo bucket probability for the YES outcome (0..1).
    quantiles_by_fxx:
        Dict mapping decoder hour offset (fxx) to a dict of quantile values:
        ``{"p10": float, "p50": float, "p90": float}``.
    last_tick_utc:
        UTC timestamp of the ForecastLoop run that produced this payload.
        Shared across all buckets for the same (condition_id, day_offset) key.
    """

    mc_prob:          float
    quantiles_by_fxx: dict[int, dict[str, float]]
    last_tick_utc:    datetime


# Bucket tuple: (lo_f, hi_f)
_BucketKey = tuple[float, float]
_BucketMap = dict[_BucketKey, BucketPayload]

_VALID_DAY_OFFSETS = frozenset({0, 1, 2})


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
        """
        self._validate_day_offset(day_offset)
        async with self._lock:
            self._data[(condition_id, day_offset)] = dict(buckets)

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
            return dict(entry) if entry is not None else None
