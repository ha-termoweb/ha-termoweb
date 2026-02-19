# Developer notes

## Architecture reference

Review the authoritative architecture overview before making structural
changes: [`docs/architecture.md`](./architecture.md).

## Backend factory API change

The `create_backend` helper in `custom_components.termoweb.backend.factory` no longer accepts a
`ws_impl` keyword argument. Custom tooling that previously passed this placeholder parameter must
drop it and rely on the default websocket implementation that ships with the integration. Future
overrides, if needed, will be exposed through a different configuration surface.

## Ducaheat accumulator write semantics

Ducaheat accumulators (and heaters served by the same backend) do not expose a monolithic
`/acm/.../settings` endpoint. The mobile apps persist state by issuing targeted POSTs to the
segmented endpoints listed in [`docs/ducaheat_api.md`](./ducaheat_api.md). Use the table there as the
source of truth when mapping Home Assistant services to cloud writes.

Key observations from traffic captures:

- **Temperature payloads must be strings** formatted with exactly one decimal place (e.g. `"22.0"`).
  Sending floats or integers causes the backend to reject the request.
- The `units` field is **always uppercase** (`"C"` or `"F"`); lowercase variants fail validation.
- `/select` is an **optional identify helper**, not a write prerequisite. Use it only when a user
  needs to physically locate a node on-site.
- Boost (accumulators only), lock, and similar toggles are literal booleans; do not send quoted values.

These semantics apply to both heater (`htr`) and accumulator (`acm`) nodes within the Ducaheat API.

### Selection (physical identify helper)

- **Endpoint:** `POST /api/v2/devs/{dev_id}/{type}/{addr}/select`
- **Identify on:** `{ "select": true }` → `201 {}`
- **Identify off:** `{ "select": false }` → `201 {}`
- This can trigger visible hardware cues (for example display flash/backlight or showing the node
  address) so installers can identify the physical device.

### Boost (start/stop)

- **Endpoint:** `POST /api/v2/devs/{dev_id}/{type}/{addr}/boost`
- **Start example:**
  ```json
  { "boost": true, "boost_time": 60, "stemp": "7.5", "units": "C" }
  ```
- **Stop example:**
  ```json
  { "boost": false }
  ```
- `boost_time` is measured in minutes and must fall within **60–600** (1–10 hours).
- `stemp` is a string with exactly one decimal place that satisfies the regex `^[0-9]+\.[0-9]$`.
- `units` must be uppercase (`"C"` or `"F"`).

### WebSocket notifications

- **Namespace:** `/api/v2/socket_io` (Socket.IO v2 over Engine.IO v3)
- **Representative update:**
  ```json
  {
    "path": "/{type}/{addr}/status",
    "body": {
      "boost": true,
      "boost_end_day": 0,
      "boost_end_min": 945,
      "stemp": "7.5",
      "units": "C"
    }
  }
  ```
- A `dev_data` snapshot with the same state often follows shortly after the incremental update.

### Canonical sequence

1. `POST /boost` (start or stop) → `201 {}`
2. WebSocket `update` on `/{type}/{addr}/status`

### QA checklist

- Start Boost for each duration 1–10 hours (60–600 minutes) and confirm the WebSocket update includes
  `boost_end_day` and `boost_end_min`.
- Stop Boost and confirm both the `boost` flag and end fields flip to the expected values.
- Validate client-side guards reject invalid `stemp` values (`"7"`, `7.5`, `"7.53"`) and lowercase
  `units`.

### PMO specifics
- No selection (`/select`) step is required before reading `/pmo/{addr}` or `/pmo/{addr}/samples`.
- Samples use epoch-second windows. Empty ranges may return HTTP 204 or `{ "samples": [] }`.
- Treat sample payload keys as vendor-controlled. Only assert the top-level `{ "samples": [...] }` shape before persisting data.
