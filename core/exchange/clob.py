"""Live CLOB exchange adapter.

Wraps ``py_clob_client.ClobClient`` for real order submission.
py_clob_client is optional; ImportError is raised at construction time
if the package is missing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from core.exchange.base import BaseExchange

if TYPE_CHECKING:
    from core.alerts import AlertBus


class ClobExchange(BaseExchange):
    """Send real orders via the Polymarket CLOB API.

    Parameters
    ----------
    clob_client:
        A ``py_clob_client.ClobClient`` instance initialised with valid
        API credentials.
    """

    def __init__(
        self,
        clob_client: Any,
        alert_bus: "AlertBus | None" = None,
    ) -> None:
        super().__init__(alert_bus=alert_bus)
        self._client = clob_client

    def place_order(
        self,
        token_id: str,
        side:     str,
        size_usd: float,
        price:    float,
    ) -> str:
        """Build and submit a GTC limit order.

        Uses the SDK's ``create_and_post_order()`` workflow.  Returns the
        exchange-assigned order ID on success.
        """
        try:
            from py_clob_client.clob_types import (  # type: ignore[import-untyped]
                OrderArgs,
                OrderType,
            )
            order_args = OrderArgs(
                token_id   = token_id,
                price      = price,
                size       = size_usd,
                side       = side.upper(),
                order_type = OrderType.GTC,
            )
        except ImportError:
            order_args = {
                "token_id": token_id,
                "price":    price,
                "size":     size_usd,
                "side":     side.upper(),
            }
        try:
            signed = self._client.create_order(order_args)
            resp   = self._client.post_order(signed)
        except Exception as exc:
            logger.error("[CLOB] Order submission failed for {}: {}", token_id, exc)
            self._emit_alert("order_failure", token_id, str(exc))
            raise

        order_id = resp.get("orderID", "") or resp.get("order_id", "")
        if not order_id:
            error_msg = resp.get("errorMsg", "rejected")
            logger.error("[CLOB] Order rejected for {}: {}", token_id, error_msg)
            self._emit_alert("order_failure", token_id, error_msg)
        else:
            logger.info(
                "[CLOB] {} {} ${:.2f} @ {:.4f} order_id={}",
                side.upper(), token_id, size_usd, price, order_id,
            )
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            logger.info("[CLOB] Cancelled order {}", order_id)
            return True
        except Exception:
            logger.warning("[CLOB] Failed to cancel order {}", order_id)
            return False

    def get_positions(self) -> list[dict]:
        """Query the CLOB API for open positions."""
        raw = self._client.get_positions()
        if not raw:
            return []
        positions = []
        for p in raw:
            positions.append({
                "token_id": str(p.get("asset_id") or p.get("token_id") or ""),
                "size_usd": float(p.get("value") or p.get("size_usd") or 0),
                "side":     str(p.get("side", "BUY")),
            })
        return positions
