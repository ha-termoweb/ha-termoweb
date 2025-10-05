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
  `window_open`, `true_radiant`, `presence`, `current_charge_per`, `target_charge_per`, `boost`, `boost_end_day`,
  `boost_end_min`, and `error_code`.
- `version` enumerates firmware and hardware revisions (`hw_version`, `fw_version`, `pid`, `uid`).

---

## Writes are segmented

Unlike TermoWeb’s consolidated `/settings` endpoint, Ducaheat exposes discrete resources per write. There is **no**
`/acm/.../settings` endpoint; the mobile apps persist accumulator state through the endpoints below.

| Endpoint | Purpose | Example body | Value types / notes |
| --- | --- | --- | --- |
| `POST /api/v2/devs/{dev_id}/{node_type}/{addr}/status` | Change operating mode, setpoint, boost flag, and display units. | `{ "mode": "manual", "stemp": "22.0", "units": "C" }`<br>`{ "boost": true }` | Temperatures must be strings with one decimal place (`"22.0"`). Units are uppercase (`"C"` / `"F"`). Boolean toggles (`boost`, etc.) are literal booleans. Returns `201 {}`. |
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

- Timed override that runs at a boost setpoint for a fixed duration, then reverts.  
- Activation paths:
  - Immediate: `POST …/status { "boost": true }`
  - Defaults: set via `POST …/setup { "extra_options": { "boost_time": 60, "boost_temp": "22.0" } }`
- Related read keys often present under `setup.extra_options` and/or `status`:
  - `boost_time`, `boost_temp`, and `boost_active`.

---

## WebSocket (Socket.IO)

**Path:** `/socket.io?token=<access_token>&dev_id=<dev_id>` (handshake may be redirected to a session identifier URL fragment).
**Namespace:** `/api/v2/socket_io`.

The app listens for at least these events:
- `dev_handshake` — initial device list / permissions (not observed in this capture but present in prior reverse engineering).
- `dev_data` — full gateway snapshot (see above).
- `update` — incremental node changes. The payload contains `path` (e.g., `/acm/2/status`) and `body` matching the corresponding
  REST resource. Clients should route updates by node type and address using the `path` components.
- App heartbeat: the backend emits `"message"` events with payload `"ping"`. Reply with `"pong"` to avoid the server pausing
  updates after a few minutes.

---

## Example cURL flow

```bash
# 1) Token
TOK=$(curl -sS -u '5c49dce977510351506c42db:tevolve'   -d 'grant_type=password&username=EMAIL&password=PASS'   https://api-tevolve.termoweb.net/client/token | jq -r .access_token)

# 2) Read heater 2
curl -sS -H "Authorization: Bearer $TOK"   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2

# 3) (Optional) Select before write
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"select": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/select

# 4) Set manual 22.0°C
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"mode":"manual","stemp":"22.0","units":"C"}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/status

# 5) Start Boost now
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"boost": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/status

# 6) Configure default Boost time
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"extra_options":{"boost_time":60,"boost_temp":"22.0"}}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/setup

# 7) De-select after write
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"select": false}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/select
```

---

## Notes and caveats

- REST and websocket calls include `User-Agent: Ducaheat/...` plus `X-Requested-With: net.termoweb.ducaheat.app` (and `Origin: https://localhost` for Socket.IO) to match the Android hybrid app environment.
- Treat unknown keys as forward‑compatible; the app is tolerant of additional fields.
- Program payload (`/prog`) structure varies by model/firmware. Capture the GET shape first and write it back unchanged after edits.
- Some deployments may accept `/htr/{addr}/boost` as a synonym for boosting, but `/status` with `{ "boost": true }` is consistently present in the app bundle.
