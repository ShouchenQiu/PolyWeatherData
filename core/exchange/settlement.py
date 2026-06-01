"""Event settlement abstractions.

SettlementRelayer: abstract base for on-chain settlement relay.
Implementation deferred pending Relayer ABI wiring -- ADR-0007.

The ExitSettler (V1 0.99-fallback sell path) was removed in V2V12.
Settlement is now handled by SettlementLoop + SettlementResolver, which
computes per-order payouts from observed temperature via signed_shares.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.exchange.base import BaseExchange


class SettlementRelayer(ABC):
    """Abstract base for on-chain settlement relay.

    Full implementation is deferred pending Relayer ABI wiring -- ADR-0007.
    """

    @abstractmethod
    def resolve_on_chain(self, event_id: str) -> bool:
        """Resolve the event on-chain via the Relayer contract.

        Returns True if the event was successfully resolved, False otherwise.
        Raises NotImplementedError until the Relayer ABI is wired (ADR-0007).
        """
