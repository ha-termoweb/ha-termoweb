# Ducaheat (Ducasa) Cloud API — Tevolve v2 (Unofficial)

This document summarizes the Ducaheat backend as implemented by the mobile app (v1.40.1) and verified against captured WebSocket traffic. It diverges from the consolidated **TermoWeb** contract: Ducaheat uses **segmented endpoints** per function and a **different Socket.IO path**.

**Base host:** `https://api-tevolve.termoweb.net`  
**Frontend:** `https://ducaheat.termoweb.net`  
**Auth client (Basic):** `5c49dce977510351506c42db:tevolve` → Base64 `NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=`

---

## Auth

**POST** `/client/token`  (HTTP Basic + form)

Form fields:
```
grant_type=password&username=<email>&password=<password>
```

Response:
```json
{ "access_token":"…", "token_type":"Bearer", "expires_in":3600, "refresh_token":"…" }
```

Use the `access_token` as `Authorization: Bearer …` for all API calls.

---

## Snapshot payload (`dev_data`)

The Socket.IO channel emits a `dev_data` event that delivers the complete gateway snapshot. The object mirrors what the mobile
app fetches via REST:

- `geoData` / `geo_data` — duplicated structures with coarse location metadata (country, administrative division, time zone,
  optional coarse coordinates). Clients should not rely on either variant being preferred.
- `away_status` — `{ "enabled": <bool>, "away": <bool>, "forced": <bool> }`.
- `nodes` — array of node descriptions (see below).
- `htr_system` — heater fleet level settings. Observed keys include `power_limit` and `setup.power_limit`,
  `setup.refresh_period`, and `setup.extra_nrg_conf.enabled`.
- `pmo_system` — power-management metadata with `main_circuit_pmos`, `max_power_config.profiles`, and
  `max_power_config.slots` (each slot: `{ "m": <minute_offset>, "i": <profile_index> }`).
- `discovery` — e.g., `{ "discovery": "off" }`.
- `connected` — gateway connection boolean.

Every node object includes `name`, `addr`, `type`, `installed`, `lost`, `uid`, `level`, and `parent`. Additional sections vary
by node type and match the REST resources for that node.

## Heater and accumulator node model (read)

**GET** `/api/v2/devs/{dev_id}/{node_type}/{addr}`

Applies to heater (`htr`) and accumulator (`acm`) nodes. The response is a consolidated object with nested sections such as
`status`, `setup`, `prog`, `prog_temps`, and `version`. Keys are model‑dependent; the app treats unknown keys leniently.

- `prog.sync_status` reflects synchronization with the cloud. `prog.prog` is an object keyed by weekday (`"0"` … `"6"`). Each
  weekday value is an array of slot states. Recent captures show 49 integer entries per day, which map 30-minute resolution
  slots to mode identifiers.
- `setup` contains operational metadata. Observed keys include:
  - `operational_mode` (integer)
  - `control_mode` (integer)
  - `units` (`"C"` / `"F"`)
  - `offset`, `priority`, `away_offset`, `min_stemp`, `max_stemp` as strings (one decimal)
  - `window_mode_enabled`, `true_radiant_enabled`, `frost_protect` (booleans)
  - `resistor_mode`, `prog_resolution` (integers)
  - `charging_conf.slot_1` / `slot_2` with `start`/`end` minute offsets and `active_days` array (seven booleans)
  - `factory_options` describing hardware capabilities (`resistor_available`, `ventilation_available`, `ventilation_type`).
- `status` contains real-time telemetry, all keyed as strings or booleans where appropriate. Observed keys: `mode`, `heating`,
  `ventilation`, `charging`, `ice_temp`, `eco_temp`, `comf_temp`, `units`, `stemp`, `mtemp`, `pcb_temp`, `power`, `locked`,
  `window_open`, `true_radiant`, `presence`, `current_charge_per`, `target_charge_per`, and `error_code`. Accumulators also
  expose `boost`, `boost_end_day`, and `boost_end_min`.
- `version` enumerates firmware and hardware revisions (`hw_version`, `fw_version`, `pid`, `uid`).

---

## Writes are segmented

Unlike TermoWeb’s consolidated `/settings` endpoint, Ducaheat exposes discrete resources per write. There is **no**
`/acm/.../settings` endpoint; the mobile apps persist accumulator state through the endpoints below.

| Endpoint | Purpose | Example body | Value types / notes |
| --- | --- | --- | --- |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/status` | Change operating mode, setpoint, boost flag, and display units. | `{ "mode": "manual", "stemp": "22.0", "units": "C" }`<br>`{ "boost": true }` | Temperatures must be strings with one decimal place (`"22.0"`). Units are uppercase (`"C"` / `"F"`). Boolean toggles (for example `select`, `lock`) are literal booleans. Boost writes are valid only when `node_type` is `"acm"`. Returns `201 {}`. |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/mode` | Explicitly set the operating mode when no setpoint change is required. | `{ "mode": "manual" }` | Use when the app issues a mode-only write. |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/prog` | Persist the weekly program definition. | *(mirrors GET payload)* | Send the `prog` object echoed by the read call; structure varies by model. |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/prog_temps` | Update named preset temperatures. | `{ "comfort": "21.0", "eco": "18.0", "antifrost": "7.0" }` | All temperature values are one-decimal strings in uppercase units context. |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/setup` | Write advanced configuration and feature toggles. | `{ "extra_options": { "boost_time": 60, "boost_temp": "22.0" } }` | Nested objects follow the GET schema; keep temperature values as strings with uppercase units. |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/lock` | Toggle the child lock. | `{ "lock": true }` | Boolean literal. |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/select` | Claim ownership prior to issuing writes. | `{ "select": true }` | Some writes require `select: true`; release with `select: false` afterwards. |

> **Temperature formatting:** Every captured request encodes degrees as strings with exactly one decimal place while keeping
> the `units` field uppercase (`"C"`/`"F"`). Back-end writes fail when the decimal precision or unit casing diverges from the
> mobile app.

---

## Samples (history / telemetry)

**GET** `/api/v2/devs/{dev_id}/{node_type}/{addr}/samples?start=<ms>&end=<ms>`

`start` and `end` are **epoch milliseconds**. The response shape varies by device and firmware; treat as opaque JSON until stabilized by capture.

## Power monitor node model (read)

**GET** `/api/v2/devs/{dev_id}/pmo/{addr}`

`pmo` nodes report:

- `power_limit` — wrapper around the configured limit (stringified value).
- `setup` — `{ "power_limit": <int>, "reverse": <bool>, "circuit_type": <int> }`.
- `version` — firmware and hardware identifiers (`hw_version`, `fw_version`, `pid`, `uid`).

No Socket.IO `status` event has been seen for `pmo` nodes, but the node appears in `dev_data` with `lost: true` when
connectivity drops.

---

## Boost / Runback behavior

- Timed override (accumulators only) that runs at a boost setpoint for a fixed duration, then reverts.
- Activation paths:
  - Immediate: `POST …/status { "boost": true }`
  - Defaults: set via `POST …/setup { "extra_options": { "boost_time": 60, "boost_temp": "22.0" } }`
- Related read keys often present under `setup.extra_options` and/or `status`:
  - `boost_time`, `boost_temp`, and `boost_active`.

---

## WebSocket (Socket.IO)

The Ducaheat app speaks **Engine.IO v3 + Socket.IO v2**. The mobile capture shows that the backend is strict about the upgrade
sequence, namespace, and heartbeat cadence; mirroring the flow below keeps the session alive indefinitely.

### Endpoint & headers

- **HTTP host:** `https://api-tevolve.termoweb.net`
- **Handshake path:** `/socket.io?token=<access_token>&dev_id=<dev_id>`
- **Namespace:** `/api/v2/socket_io`
- **Mandatory headers:**
  - `User-Agent: Ducaheat/<app-version>`
  - `X-Requested-With: net.termoweb.ducaheat.app`
  - `Origin: https://localhost` (even though no page lives at that URL; omit only if the integration can tolerate stricter
    CORS policies in the future).
  - Copy the app’s `Accept-Language` and `Accept` headers if available. The capture shows `Accept: */*`.

### Engine.IO handshake

1. **GET polling open** – `transport=polling&EIO=3&t=<random>`.
   - Expect HTTP `200` with a payload that contains multiple Engine.IO packets. The first packet is `0{"sid":"<sid>","upgrades":["websocket"],"pingInterval":<ms>,"pingTimeout":<ms>}`.
   - Persist the `sid`, `pingInterval`, and `pingTimeout`. The capture shows `pingInterval=25000` ms and `pingTimeout=50000` ms,
     but the client must trust the values returned by the server at runtime.
   - The HTTP response includes `Set-Cookie: io=<sid>` and `Set-Cookie: server_id=…`. Keep the same HTTP client session for all
     follow-up requests so those cookies are resent automatically.
   - Raw body example (101 bytes, no gzip):

     ```
     0{"sid":"f17Ht9TSWZcSC1Y1Vow7","upgrades":["websocket"],"pingInterval":25000,"pingTimeout":50000}
     ```
2. **POST namespace join (polling)** – repeat the query parameters above plus `sid=<sid>` and POST `40/api/v2/socket_io` as the
   body (content type `text/plain;charset=UTF-8`). This registers the namespace while still on HTTP long-polling.
   - Engine.IO polling frames must include a length prefix. The POST body in the capture is the ASCII text `18:40/api/v2/socket_io`.
   - Expect HTTP `200` with a 2-byte body (`ok`). Cookies from the GET are echoed back.
3. **GET polling drain** – GET once more with the same query (new `t` value). The server usually responds with a no-op packet.
   - Continue to send the `sid` query parameter. Skipping this step leaves unread packets queued on the polling transport.

> **Captured request headers:**
> ```
> GET /socket.io/?token=<token>&dev_id=<dev>&EIO=3&transport=polling&t=PcpSY7k HTTP/2
> Origin: https://localhost
> User-Agent: Mozilla/5.0 (Linux; Android 15; …)
> X-Requested-With: net.termoweb.ducaheat.app
> Accept: */*
> Accept-Language: en-US,en;q=0.9
> Accept-Encoding: gzip, deflate, br, zstd
> ```
>
> The POST uses the same header set plus `Content-Type: text/plain;charset=UTF-8` and the polling body described above.

### Upgrade to WebSocket

4. **WebSocket connect** – switch `transport=websocket` and reuse the `sid`. Re-send the headers listed above except for
   hop-by-hop values (drop `Connection`, `Accept-Encoding`, etc.).
   - The upgrade request mirrors the capture:

     ```
     GET /socket.io/?…&transport=websocket&sid=f17Ht9TSWZcSC1Y1Vow7 HTTP/1.1
     Connection: Upgrade
     Upgrade: websocket
     Sec-WebSocket-Version: 13
     Sec-WebSocket-Key: <base64>
     Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits
     ```
   - The server answers `101 Switching Protocols`, reissues the `io` and `server_id` cookies, and enables permessage-deflate. Do
     not set an Engine.IO heartbeat on the client socket; the server controls ping cadence.
5. **Probe** – immediately send the Engine.IO probe string `2probe` and wait for `3probe`.
6. **Upgrade confirm** – send `5` to finalize the upgrade. The server will reply with `40/api/v2/socket_io` to acknowledge the
   namespace. Treat absence of the `40` ack as a fatal error.
7. **Namespace enter (redundant but required)** – send `40/api/v2/socket_io` once more after the upgrade. The capture shows the
   mobile app doing this; reproducing it avoids sporadic “unknown namespace” errors on reconnect.

### Initial payloads & subscriptions

8. **Request snapshot** – send a Socket.IO event `42/api/v2/socket_io,["dev_data"]` immediately after the namespace join. The
   server responds with `42/api/v2/socket_io,["dev_data",{...}]` containing the full gateway state described earlier.
9. **Sample subscriptions** – for every heater or accumulator that exposes energy samples, emit `42/api/v2/socket_io,["subscribe","/<node_type>/<addr>/samples"]`. The backend acknowledges silently; subsequent `update` events carry sampled values.

### Runtime events

The active session surfaces the following Socket.IO events (all scoped to `/api/v2/socket_io`):

- `dev_handshake` — initial device permissions. Rare in recent builds but historically emitted before the first `dev_data`.
- `dev_data` — complete gateway snapshot. Treat the body as authoritative and refresh caches when received.
- `update` — incremental node delta. The payload is an object with:
  - `path` — e.g., `/htr/2/status` or `/acm/1/prog`
  - `body` — JSON structure matching the REST endpoint for that resource.
- `message` — vendor heartbeat. Payload `"ping"` must be answered with `"pong"` by sending `42/api/v2/socket_io,["message","pong"]`.
- `samples` — optional stream of historical datapoints for subscribed nodes. Payloads match the REST `/samples` response for
  that node type.

### Heartbeats & timeouts

Engine.IO and Socket.IO layer pings are independent, and both are mandatory:

| Incoming frame | Meaning | Required response |
| --- | --- | --- |
| `2` | Engine.IO ping | Reply `3` immediately. Failure triggers a disconnect after `pingTimeout`.
| `4` frame whose payload is `"2"` | Socket.IO ping (global namespace) | Reply with a bare `3` frame (Socket.IO pong).
| `4` frame whose payload starts with `"2/"` | Socket.IO ping (namespace) | Reply `3/<namespace>` (e.g., `3/api/v2/socket_io`).
| `42/api/v2/socket_io,["message","ping"]` | Application keep-alive | Respond with `42/api/v2/socket_io,["message","pong"]` within the Engine.IO interval.

Timers:

- Use the `pingInterval` from the handshake as the maximum idle period between Engine.IO pings. Start a watchdog at 80% of that
  value (e.g., 20s if the server advertises 25s). If no ping arrives within the watchdog window, proactively reconnect.
- Treat `pingTimeout` as the grace period to deliver the pong. The capture shows the server closing the socket roughly 5 s after
  the timeout elapses if the client stays silent.
- Expect a fresh Socket.IO ping every other Engine.IO cycle. Missing a namespace pong does not immediately drop the socket, but
  subsequent events stall until the pong is received.

### Session lifecycle

- The server does not emit an explicit “welcome” beyond the `40` ack. Consider the socket healthy only after receiving either
  `dev_data` or `update` with a non-empty body.
- On reconnect, redo the full polling + upgrade dance with a new token. Reusing an expired `sid` results in HTTP 400 responses
  from the polling endpoints.
- Send `close` frames with code `1001` (`GOING_AWAY`) when shutting down to match the app’s behavior.

---

## Example cURL flow

```bash
# 1) Token
TOK=$(curl -sS -u '5c49dce977510351506c42db:tevolve'   -d 'grant_type=password&username=EMAIL&password=PASS'   https://api-tevolve.termoweb.net/client/token | jq -r .access_token)

# 2) Read accumulator 2
curl -sS -H "Authorization: Bearer $TOK"   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/acm/2

# 3) (Optional) Select before write
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"select": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/acm/2/select

# 4) Set manual 22.0°C
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"mode":"manual","stemp":"22.0","units":"C"}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/acm/2/status

# 5) Start Boost now
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"boost": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/acm/2/status

# 6) Configure default Boost time
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"extra_options":{"boost_time":60,"boost_temp":"22.0"}}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/acm/2/setup

# 7) De-select after write
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"select": false}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/acm/2/select
```

---

## Notes and caveats

- REST and websocket calls include `User-Agent: Ducaheat/...` plus `X-Requested-With: net.termoweb.ducaheat.app` (and `Origin: https://localhost` for Socket.IO) to match the Android hybrid app environment.
- Treat unknown keys as forward‑compatible; the app is tolerant of additional fields.
- Program payload (`/prog`) structure varies by model/firmware. Capture the GET shape first and write it back unchanged after edits.
- Some deployments may accept `/acm/{addr}/boost` as a synonym for boosting, but `/status` with `{ "boost": true }` is consistently present in the app bundle.
