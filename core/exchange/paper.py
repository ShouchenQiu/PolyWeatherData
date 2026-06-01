"""Paper exchange: simulates order execution without sending real orders.

Uses L2 order book depth (from a book_fetcher) to compute VWAP fill prices
via compute_quote().  Position and order persistence is the responsibility of
TradingLoop + PositionTracker; this class holds no in-memory state.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.exchange.base import BaseExchange
from core.pricing.quote import compute_quote

if TYPE_CHECKING:
    from core.alerts import AlertBus


class PaperExchange(BaseExchange):
    """Simulate order fills using real L2 CLOB book depth.

    Parameters
    ----------
    book_fetcher:
        Object exposing ``get_book(token_id) -> NormalizedBook`` (dict with
        ``"bids"`` and ``"asks"`` lists).  When None every ``place_order``
        call returns an empty string (cannot simulate without a book).
    alert_bus:
        Optional alert bus for book-fetch failures.
    """

    def __init__(
        self,
        book_fetcher: Any = None,
        alert_bus: "AlertBus | None" = None,
    ) -> None:
        super().__init__(alert_bus=alert_bus)
        self._fetcher = book_fetcher

    # ------------------------------------------------------------------
    # BaseExchange interface
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side:     str,
        size_usd: float,
        price:    float,
    ) -> str:
        """Simulate a fill by walking L2 depth via compute_quote().

        BUY orders consume asks ascending; SELL orders consume bids
        descending.  Returns an order_id string on success, or an empty
        string when depth is insufficient (complete_fill=False) or when
        the book cannot be fetched.

        The ``price`` argument is kept for interface compatibility but is
        not used for fill calculation; the actual fill VWAP comes from the
        live L2 book.
        """
        if self._fetcher is None:
            logger.warning(
                "[PAPER] No book_fetcher; cannot simulate fill for {} {}",
                side.upper(), token_id,
            )
            return ""

        try:
            book = self._fetcher.get_book(token_id)
        except Exception as exc:
            logger.error("[PAPER] get_book() failed for {}: {}", token_id, exc)
            self._emit_alert("book_fetch_failure", token_id, str(exc))
            return ""

        try:
            quote = compute_quote(book, side.upper(), size_usd)
        except Exception as exc:
            logger.error("[PAPER] compute_quote() failed for {}: {}", token_id, exc)
            return ""

        if not quote.complete_fill:
            logger.info(
                "[PAPER] Insufficient depth for {} {} ${:.2f}; "
                "max_fillable=${:.2f}",
                side.upper(), token_id, size_usd, quote.max_fillable_usd,
            )
            return ""

        order_id = str(uuid.uuid4())
        logger.debug(
            "[PAPER] {} {} ${:.2f} @ vwap={:.4f} order_id={}",
            side.upper(), token_id, size_usd, quote.vwap_price, order_id,
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
