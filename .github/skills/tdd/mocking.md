# When to Mock

Mock at **system boundaries** only:

- External APIs (payment, email, etc.)
- Databases (sometimes - prefer test DB)
- Time/randomness
- File system (sometimes)
- PolyWeatherBot live boundaries: HRRR/IEM network calls, CLOB clients, real
  order submission, Telegram

Don't mock:

- Your own classes/modules
- Internal collaborators
- Anything you control

## Designing for Mockability

At system boundaries, design interfaces that are easy to mock:

**1. Use dependency injection**

Pass external dependencies in rather than creating them internally:

```python
# Easy to fake in tests.
class TradingLoop:
    def __init__(self, probability_store, alert_bus, exchange):
        self.probability_store = probability_store
        self.alert_bus = alert_bus
        self.exchange = exchange


# Hard to test: creates the live boundary internally.
class TradingLoop:
    def __init__(self):
        self.exchange = ClobExchange.from_env()
```

**2. Prefer SDK-style interfaces over generic fetchers**

Create specific functions for each external operation instead of one generic function with conditional logic:

```python
# GOOD: each method has one shape.
class OrderbookFetcher:
    def get_book(self, token_id: str) -> OrderBook:
        ...


# BAD: tests must branch on operation strings.
class ClobGateway:
    def call(self, operation: str, payload: dict) -> dict:
        ...
```

The SDK approach means:
- Each mock returns one specific shape
- No conditional logic in test setup
- Easier to see which endpoints a test exercises
- Type safety per endpoint
