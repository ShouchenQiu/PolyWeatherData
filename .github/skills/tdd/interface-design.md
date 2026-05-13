# Interface Design for Testability

Good interfaces make testing natural:

1. **Accept dependencies, don't create them**

   ```python
   # Testable
   def __init__(self, probability_store, alert_bus, exchange):
       self.probability_store = probability_store
       self.alert_bus = alert_bus
       self.exchange = exchange

   # Hard to test
   def __init__(self):
       self.exchange = ClobExchange.from_env()
   ```

2. **Return results, don't produce side effects**

   ```python
   # Testable
   def bucket_payloads_for_event(event, model_output) -> dict:
       ...

   # Hard to test
   def write_bucket_payloads_for_event(event, model_output) -> None:
       ...
   ```

3. **Small surface area**
   - Fewer methods = fewer tests needed
   - Fewer params = simpler test setup
