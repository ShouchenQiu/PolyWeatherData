# Settlement uses IEM observed max-temp confirmation (Plan B); Relayer path stubbed for future

`settlement_loop` determines that an Event is settled by reading the IEM observation
for the event's station and date from ParquetStore and comparing the daily maximum
temperature against the event's bucket boundaries. Once the correct bucket is confirmed,
the Event status is written to `"settled"` in the DB. This replaces the original
intent to listen for Polymarket's on-chain Relayer settlement event, which requires
reverse-engineering undocumented SDK internals.

**Considered option (A ¡ª Relayer path)**: Subscribe to the Polymarket Relayer contract
events to detect the on-chain resolution and derive payout from the resolved outcome.
Deferred because the Relayer interface is undocumented and no open-source reference
implementation was available at decision time. A `SettlementRelayer` abstract base class
(single method `resolve_on_chain(event_id) -> bool`) is stubbed in
`core/exchange/settlement.py` so Plan A can be wired in without changing
`settlement_loop`'s interface once a suitable implementation is found.

**Consequence**: Plan B settlement is correct as long as IEM data for the event date is
available in ParquetStore before `settlement_loop` processes the Event. If IEM data
is missing or delayed, the Event stays in `"pending_settle"` until the next
`settlement_loop` run. The `SettlementRelayer` stub must raise `NotImplementedError`
with a message pointing to this ADR.
