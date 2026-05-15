"""Event settlement logic.

Settlement strategy:
  1. If Relayer gasless settlement is available, call it.
  2. Otherwise, list all open YES positions for the event and place
     SELL limit orders at 0.99 to exit.

This module intentionally keeps Relayer integration as a stub because
the on-chain Relayer ABI is not yet wired.  The fallback path (0.99
sell) is fully implemented and used in paper trading.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.exchange.base import BaseExchange


# Sell price used when the Relayer is unavailable
_FALLBACK_SELL_PRICE = 0.99


class Settler:
    """Settle expired events by closing all open YES positions.

    Parameters
    ----------
    relayer_client:
        Optional Relayer gasless settlement client.  When None the
        fallback 0.99 sell path is always used.
    fallback_price:
        Limit sell price for the fallback path (default 0.99).
    """

    def __init__(
        self,
        relayer_client: object | None  = None,
        fallback_price: float          = _FALLBACK_SELL_PRICE,
    ) -> None:
        self._relayer        = relayer_client
        self._fallback_price = fallback_price

    def settle(self, event: dict, exchange: "BaseExchange") -> str:
        """Close all open YES positions for *event*.

        Parameters
        ----------
        event:
            Event dict with at minimum ``"condition_id"`` and
            ``"token_ids"`` (list of YES token IDs for this event).
        exchange:
            Exchange adapter used to place exit orders.

        Returns
        -------
        ``"relayer"`` if gasless settlement was used, otherwise
        ``"fallback"`` after placing 0.99 sell orders.
        """
        token_ids = event.get("token_ids") or []

        # --- Try Relayer ---
        if self._relayer is not None:
            try:
                result = self._relayer.settle(event.get("condition_id", ""))
                if result:
                    return "relayer"
            except Exception:
                pass  # Fall through to 0.99 sell

        # --- Fallback: sell at 0.99 ---
        positions = exchange.get_positions()
        for pos in positions:
            if pos["token_id"] in token_ids and pos.get("size_usd", 0) > 0:
                exchange.place_order(
                    token_id = pos["token_id"],
                    side     = "SELL",
                    size_usd = pos["size_usd"],
                    price    = self._fallback_price,
                )

        return "fallback"


class SettlementRelayer:
    """On-chain settlement relay stub.

    Full implementation is deferred pending Relayer ABI wiring -- ADR-0007.
    """

    def resolve_on_chain(self, event_id: str) -> bool:
        raise NotImplementedError(
            "SettlementRelayer.resolve_on_chain is deferred pending "
            "Relayer ABI wiring -- see ADR-0007."
        )
