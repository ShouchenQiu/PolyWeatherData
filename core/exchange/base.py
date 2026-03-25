"""BaseExchange ABC and Order helper."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExchange(ABC):
    """Abstract base class for all exchange adapters."""

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
