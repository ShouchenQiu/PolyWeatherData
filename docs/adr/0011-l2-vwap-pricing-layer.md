# All price consumers share one L2-VWAP pricing module; full-fill-or-skip in v1

**Status**: accepted

Four distinct price concepts previously conflated in strategy, paper exchange, and
backtest must be separated: **mark price** (best-bid/best-ask midpoint, for display and
staleness estimation), **executable price** (VWAP computed by walking L2 depth for a
given side and size, used for edge calculation), **marginal limit price** (the worst
price consumed by that depth walk, used for live limit order submission), and **fill
price** (the price at which an order actually settles; in paper mode equals executable
VWAP; in live mode equals the true CLOB fill). The module `core/pricing/quote.py`
owns the single `compute_quote(book, side, size_usd) -> Quote` function; `Quote` carries
`vwap_price`, `max_fillable_usd`, `filled_shares`, `slippage_bps`, `best_bid`,
`best_ask`, `mid_price`, `limit_price`, and `complete_fill`. BUY walks asks (ascending),
SELL walks bids (descending); `OrderbookFetcher` must defensively sort both sides before
returning. Live submission passes `filled_shares` independently from `limit_price`;
deriving shares as `notional / limit_price` is incorrect for a multi-level fill. When
`complete_fill=False`, callers skip or cancel -- no partial fills in v1.

`EdgeStrategy` replaces `mid_price` with `quote.vwap_price` for edge computation,
using a configurable `quote_size_usd` as a proxy order size at signal time.
`TradingLoop` **re-quotes** after `RiskPipeline.compute_size()` resolves the actual
`size_usd`; if the re-quoted edge no longer clears `min_edge`, the order is skipped.
`PaperExchange` uses the same `compute_quote` for fill simulation; a BUY fills at
asks VWAP, a SELL at bids VWAP; depth-insufficient orders are rejected outright.
`BacktestEngine` uses `compute_quote` when historical L2 book data is available;
rows with only `mid_price_yes` fall back to a `legacy_mid_price` field and must not
be labelled `clob_price`.

**Considered option -- keep mid-price in strategy, VWAP only in execution**: Simpler
short-term, but leaves edge estimates systematically optimistic (ignoring real
slippage) and means signal quality cannot be measured consistently across paper,
backtest, and live. Rejected because it perpetuates the three-price conflation and
makes PnL attribution unreliable.

**Considered option -- introduce fills table and partial-fill semantics now**:
Correct long-term direction, but would require simultaneous changes to
`PositionTracker`, PnL module, settlement, and dashboard. Deferred: v1 uses
full-fill-or-skip; fills table introduced in a subsequent slice once order/fill
semantics are stabilised.

**Consequences**: `core/pricing/quote.py` is a new leaf module with no imports
from other project packages. `OrderbookFetcher` must expose `best_bid` and `best_ask`
explicitly (currently a hidden bug in `SniperLoop` which reads `best_ask` from a
fetcher that does not provide it). All callers that previously passed
`{"mid_price": float}` as a book dict must be updated to pass the full normalized
L2 book (`bids` and `asks`) into `compute_quote`.
`settings.quote_size_usd` is the shared proxy size used by `EdgeStrategy`.
