# Live mode uses L2 ClobClient with ApiCreds; paper mode uses a separate L1-only client wrapped as OrderbookFetcher

In live mode `BotManager` constructs a full L2 `ClobClient` (chain_id + ApiCreds:
api_key, secret, passphrase loaded from environment variables) and passes it to
`ClobExchange` for order submission and position queries. In paper mode a separate
L1-only `ClobClient` (chain_id only, no ApiCreds) is constructed and wrapped in an
`OrderbookFetcher` that exposes only `get_book(token_id)`; this client is passed
exclusively to `PaperExchange` for real-book price reads. L2 credentials are never
loaded, validated, or held in paper mode.

**Considered option ¡ª single shared L2 client**: Construct one L2 client and pass it
to both `ClobExchange` and `PaperExchange`. Rejected because paper mode then holds
full signing credentials even though it never submits orders, widening the credential
exposure surface unnecessarily and making it impossible to run paper mode without
valid L2 keys.

**Consequences**: `BotManager._build()` must branch on `settings.paper_mode` to choose
which client construction path to take. `PaperExchange` depends only on
`OrderbookFetcher.get_book()`, never on `ClobClient` directly, so the interface
boundary is stable regardless of which underlying client backs it.
