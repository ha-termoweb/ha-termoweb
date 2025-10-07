# Developer notes

## Ducaheat accumulator write semantics

Ducaheat accumulators (and heaters served by the same backend) do not expose a monolithic
`/acm/.../settings` endpoint. The mobile apps persist state by issuing targeted POSTs to the
segmented endpoints listed in [`docs/ducaheat_api.md`](./ducaheat_api.md). Use the table there as the
source of truth when mapping Home Assistant services to cloud writes.

Key observations from traffic captures:

- **Temperature payloads must be strings** formatted with exactly one decimal place (e.g. `"22.0"`).
  Sending floats or integers causes the backend to reject the request.
- The `units` field is **always uppercase** (`"C"` or `"F"`); lowercase variants fail validation.
- Some operations require a `select: true` POST before the substantive write and `select: false`
  afterwards. Keep the claim short-lived to avoid stepping on app sessions.
- Boost (accumulators only), lock, and similar toggles are literal booleans; do not send quoted values.

These semantics apply to both heater (`htr`) and accumulator (`acm`) nodes within the Ducaheat API.

### Boost presets, helper services, and duration rules

Accumulator boost defaults are stored under the `/setup` namespace as `extra_options.boost_time`
and `extra_options.boost_temp`. The integration keeps an in-memory cache of these values so
entity services can honour user-defined presets even when the REST payload omits a duration. When
`termoweb.start_boost` is invoked without an explicit runtime the coordinator falls back to the
cached preset, which in turn synchronises with the `select.termoweb_…_boost_duration` entity.

Three Home Assistant services wrap the Ducaheat boost workflow:

- `termoweb.configure_boost_preset` updates the default runtime and/or boost setpoint via
  `/setup`. This service validates the payload and refuses to write unless at least one field is
  present.
- `termoweb.start_boost` issues a `/status` write with `boost: true` (and an optional
  `boost_time`) and immediately applies optimistic state updates so Lovelace feedback remains
  responsive while the REST call is in flight.
- `termoweb.cancel_boost` performs the inverse `/status` write (`boost: false`) and clears cached
  runtime metadata.

Ducaheat limits boost sessions to **1–120 minutes**. Validation is enforced centrally:

- The public services reject values outside this range and log `ERROR` messages for developer
  diagnostics.
- Button helpers and select entities derive their choices from the same validation logic so UI
  affordances never generate invalid durations.
