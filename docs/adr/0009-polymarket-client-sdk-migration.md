# Replace py_clob_client with polymarket-client; supersedes ADR-0008

**Status**: accepted | supersedes ADR-0008

`py_clob_client` is no longer maintained. The replacement SDK `polymarket-client`
splits the old single `ClobClient` into two distinct classes: `PublicClient`
(unauthenticated, for orderbook reads and market discovery) and `SecureClient`
(authenticated, constructed via `SecureClient.create(private_key=..., wallet=...)` using
an EVM private key instead of the three-part `ApiCreds`). The L1/L2 principle from
ADR-0008 is preserved unchanged: paper mode holds only a `PublicClient` and never loads
private key material; live mode constructs a `SecureClient`. The five settings fields
`polymarket_api_key`, `polymarket_api_secret`, `polymarket_passphrase`,
`polymarket_funder`, and `polymarket_proxy_address` are replaced by two fields:
`POLYMARKET_PRIVATE_KEY` (the EVM private key for signing) and
`POLYMARKET_WALLET_ADDRESS` (the wallet to act for, passed as the `wallet` argument to
`SecureClient.create`; defaults to the signer address when omitted). `ApiCreds` are
derived automatically by the SDK during `SecureClient.create` and are not stored in
settings. The raw HTTP gamma call in `events/discovery.py` is replaced by
`PublicClient.list_events(slug=..., live=True)`.

**Considered option -- keep py_clob_client with a shim**: Maintain the old SDK and
write an adapter layer. Rejected because `py_clob_client` is unmaintained and its
`OrderArgs`/`ClobClient` API no longer matches the live Polymarket CLOB endpoints.

**Consequences**: `core/bot.py`, `core/exchange/clob.py`, `data/clob/orderbook.py`,
`events/discovery.py`, and `config/settings.py` all require mechanical rewrites. The
orderbook `bids` field ordering changes from descending to ascending (lowest bid
first) -- any code that assumed the first bid is the best bid must be corrected.
