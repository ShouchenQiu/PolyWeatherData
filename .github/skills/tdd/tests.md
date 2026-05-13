# Good and Bad Tests

## Good Tests

**Integration-style**: Test through real interfaces, not mocks of internal parts.

```python
def test_order_can_be_recorded_and_counted(position_tracker):
    order_id = "order-1"

    position_tracker.record_order(
        order_id=order_id,
        token_id="token-1",
        condition_id="condition-1",
        side="BUY",
        size_usd=10.0,
        price=0.42,
    )

    assert position_tracker.open_positions_by_token() == {"token-1": 1}
```
```

Characteristics:

- Tests behavior users/callers care about
- Uses public API only
- Survives internal refactors
- Describes WHAT, not HOW
- One logical assertion per test

## Bad Tests

**Implementation-detail tests**: Coupled to internal structure.

```python
def test_trading_loop_calls_strategy_private_helper(trading_loop, mocker):
    helper = mocker.patch.object(trading_loop, "_build_signals")

    trading_loop.tick_once()

    helper.assert_called_once()
```
```

Red flags:

- Mocking internal collaborators
- Testing private methods
- Asserting on call counts/order
- Test breaks when refactoring without behavior change
- Test name describes HOW not WHAT
- Verifying through external means instead of interface

```python
# BAD: verifies storage detail instead of behavior.
def test_record_order_inserts_row(db, position_tracker):
    position_tracker.record_order(...)
    assert db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


# GOOD: verifies behavior through the PositionTracker interface.
def test_record_order_makes_token_open(position_tracker):
    position_tracker.record_order(...)
    assert position_tracker.open_positions_by_token()["token-1"] == 1
```
```
