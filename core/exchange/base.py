"""BaseExchange ABC and Order helper."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.alerts import AlertBus


class BaseExchange(ABC):
    """Abstract base class for all exchange adapters."""

    def __init__(self, alert_bus: "AlertBus | None" = None) -> None:
        self._alert_bus = alert_bus

    def _emit_alert(
        self,
        alert_type: str,
        token_id:   str,
        reason:     str,
    ) -> None:
        """Emit a trade-related alert via the wired AlertBus.

        No-op when no AlertBus is wired.
        """
        if self._alert_bus is None:
            return
        from core.alerts import AlertEvent
        self._alert_bus.emit(AlertEvent(
            alert_type  = alert_type,
            station     = None,
            message     = f"{type(self).__name__}: {alert_type} for {token_id}: {reason}",
            occurred_at = datetime.now(tz=timezone.utc),
            context     = {"token_id": token_id, "reason": reason},
        ))

    @abstractmethod
    def place_order(
        self,
        token_id: str,
        side:     str,
        size_usd: float,
        price:    float,
    ) -> str:
        """Submit an order and return the exchange order ID.

        Parameters
        ----------
        token_id:
            Polymarket CLOB token identifier.
        side:
            ``"BUY"`` or ``"SELL"``.
        size_usd:
            Nominal USDC amount.
        price:
            Limit price (0..1, binary market convention).

        Returns
        -------
        Order ID string assigned by the exchange (or locally generated for
        paper trading).
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Returns True if the cancellation was accepted, False otherwise.
        """

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return current open positions.

        Each position dict must have at minimum:
        ``{"token_id": str, "size_usd": float, "side": str}``.
        """
