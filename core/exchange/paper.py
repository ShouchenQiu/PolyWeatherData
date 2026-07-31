"""Paper exchange: accepts validated orders without sending live orders.

Depth validation is the responsibility of TradingLoop (two compute_quote()
passes before this is called). This class accepts the validated marginal limit
price and returns a UUID order_id on any valid order, or an empty string for
degenerate inputs. Paper fill accounting remains upstream at executable VWAP.
No in-memory state is held.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from loguru import logger

from core.exchange.base import BaseExchange

if TYPE_CHECKING:
    from core.alerts import AlertBus


class PaperExchange(BaseExchange):
    """Simulate order fills without sending real orders to the CLOB.

    Depth validation is performed upstream in TradingLoop via two
    ``compute_quote()`` passes.  This adapter trusts the ``price`` already
    validated by the caller and returns a UUID on success.

    Parameters
    ----------
    alert_bus:
        Optional alert bus for exchange-level alerts.
    """

    def __init__(
        self,
        alert_bus: "AlertBus | None" = None,
    ) -> None:
        super().__init__(alert_bus=alert_bus)

    # ------------------------------------------------------------------
    # BaseExchange interface
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side:     str,
        size_usd: float,
        price:    float,
        shares:   float | None = None,
    ) -> str:
        """Accept a caller-validated paper order.

        Returns a UUID order_id on success, or an empty string for
        degenerate inputs (``price <= 0`` or ``size_usd <= 0``).

        Depth has already been verified by TradingLoop via two
        ``compute_quote()`` calls; this adapter trusts the validated marginal
        limit ``price`` and does not re-fetch the book. TradingLoop records the
        simulated fill separately at executable VWAP.
        """
        if price <= 0 or size_usd <= 0 or (shares is not None and shares <= 0):
            logger.warning(
                "[PAPER] Degenerate order rejected: {} {} size={} price={}",
                side.upper(), token_id, size_usd, price,
            )
            return ""

        order_id = str(uuid.uuid4())
        logger.debug(
            "[PAPER] {} {} ${:.2f} @ {:.4f} order_id={}",
            side.upper(), token_id, size_usd, price, order_id,
        )
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """No in-flight orders are tracked; always returns False."""
        return False

    def get_positions(self) -> list[dict]:
        """No in-memory positions; returns empty list.

        Position tracking is the responsibility of PositionTracker.
        """
        return []
