"""In-process shared probability store for ForecastLoop and TradingLoop."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime, timezone


# Bucket tuple: (lo_f, hi_f)
_BucketKey = tuple[float, float]

_VALID_DAY_OFFSETS = frozenset({0, 1, 2})


@dataclass(frozen=True)
class ProbabilitySnapshot:
    """A complete snapshot for one (condition_id, day_offset) key.

    Written by ForecastLoop via ProbabilityStore.write(); read by TradingLoop
    via ProbabilityStore.read().

    Fields
    ------
    buckets:
        Mapping of (lo_f, hi_f) -> mc_prob (float, 0..1) for this key.
    quantiles_by_fxx:
        Decoder-hour quantile output shared by all buckets in the snapshot:
        ``{fxx: {"p10": float, "p50": float, "p90": float}}``.
    last_tick_utc:
        UTC timestamp of the ForecastLoop run that produced this snapshot.
    """

    buckets:          dict[tuple[float, float], float]
    quantiles_by_fxx: dict[int, dict[str, float]]
    last_tick_utc:    datetime

    def age_minutes(self, now: datetime | None = None) -> float:
        """Return minutes elapsed since last_tick_utc.

        Parameters
        ----------
        now:
            Reference time.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        return (now - self.last_tick_utc).total_seconds() / 60.0

    def is_fresh(self, threshold_min: float, now: datetime | None = None) -> bool:
        """Return True when age_minutes(now) <= threshold_min."""
        return self.age_minutes(now) <= threshold_min


class ProbabilityStore:
    """Shared in-process store keyed by (condition_id, day_offset).

    Written by ForecastLoop, read by TradingLoop.
    An asyncio.Lock guards all mutations.

    day_offset domain: {0, 1, 2} -- up to 3 simultaneously active events.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._data: dict[tuple[str, int], ProbabilitySnapshot] = {}

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
        snapshot: ProbabilitySnapshot,
    ) -> None:
        """Replace the snapshot for (condition_id, day_offset).

        Parameters
        ----------
        condition_id:
            Polymarket condition/event ID.
        day_offset:
            Day offset in {0, 1, 2}.
        snapshot:
            ProbabilitySnapshot to store.  Deep-copied on ingress so
            subsequent mutation of the caller's object has no effect.

        Raises
        ------
        ValueError
            If day_offset is not in {0, 1, 2}.
        """
        self._validate_day_offset(day_offset)
        stored = copy.deepcopy(snapshot)
        async with self._lock:
            self._data[(condition_id, day_offset)] = stored

    async def read(
        self,
        condition_id: str,
        day_offset: int,
    ) -> ProbabilitySnapshot | None:
        """Return a ProbabilitySnapshot for (condition_id, day_offset), or None.

        Parameters
        ----------
        condition_id:
            Polymarket condition/event ID.
        day_offset:
            Day offset in {0, 1, 2}.

        Returns
        -------
        ProbabilitySnapshot with buckets, quantiles_by_fxx, and last_tick_utc,
        or None if the key has not been written yet.

        Raises
        ------
        ValueError
            If day_offset is not in {0, 1, 2}.
        """
        self._validate_day_offset(day_offset)
        async with self._lock:
            entry = self._data.get((condition_id, day_offset))
            if entry is None:
                return None
            return copy.deepcopy(entry)
