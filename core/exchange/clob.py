"""Live CLOB exchange adapter.

Wraps the polymarket-client SDK's SecureClient for real order submission.
polymarket-client is optional; the adapter accepts any object exposing
``place_limit_order(*, token_id, price, size, side)`` and
``cancel_orders(*, order_ids)``.
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
        A ``polymarket_client.SecureClient`` instance (or any object
        exposing ``place_limit_order`` and ``cancel_orders``).
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
        """Submit a limit order via the CLOB API.

        Delegates to ``SecureClient.place_limit_order()``.  Returns the
        exchange-assigned order ID, or an empty string on rejection.
        """
        try:
            # NOTE(V2V09): Confirm whether SecureClient.place_limit_order()
            # expects `size` in shares or in notional USDC.  If the SDK
            # interprets `size` as shares, the caller must convert:
            #   shares = size_usd / price
            # before passing.  As of V2F02 we pass `size_usd` directly; fix
            # belongs in V2V09 once SDK contract is verified.
            resp = self._client.place_limit_order(
                token_id=token_id,
                price=price,
                size=size_usd,
                side=side.upper(),
            )
        except Exception as exc:
            logger.error("[CLOB] Order submission failed for {}: {}", token_id, exc)
            self._emit_alert("order_failure", token_id, str(exc))
            raise

        order_id = getattr(resp, "order_id", None) or ""
        if not order_id:
            error_msg = getattr(resp, "error_msg", None) or "rejected"
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
            self._client.cancel_orders(order_ids=[order_id])
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
